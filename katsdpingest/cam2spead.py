"""Device server that listens to CAM events and passes updates to SPEAD stream."""

import logging
import threading
import time
import Queue
import copy

import spead
from katcp import DeviceServer, Sensor
from katcp.kattypes import request, return_reply, Str, Int

from katsdpingest import __version__


logger = logging.getLogger(__name__)


class SensorBridge(object):
    """Bridge between single KATCP sensor and corresponding SPEAD item in stream.

    Parameters
    ----------
    name : string
        Sensor name (used to name the corresponding SPEAD item)
    katcp_sensor : :class:`katcorelib.KATSensor` object
        Sensor object representing KATCP sensor
    server : :class:`Cam2SpeadDeviceServer` object
        Device server that serves SPEAD stream

    """

    # Pick a SPEAD id range that is not in use here
    next_available_spead_id = 0x7000

    def __init__(self, name, katcp_sensor, server):
        self.name, self.katcp_sensor, self.server = name, katcp_sensor, server
        self.spead_id = SensorBridge.next_available_spead_id
        SensorBridge.next_available_spead_id += 1
        self.strategy = 'none'
        self.param = ''
        self.listening = False
        self.last_update = ''
        # Store katcp.Sensor which will be used to parse KATCP string in listener
        sensor_type = Sensor.parse_type(self.katcp_sensor.type)
        params = ['unknown'] if sensor_type == Sensor.DISCRETE else None
        self._sensor = Sensor(sensor_type, self.katcp_sensor.name,
                              self.katcp_sensor.description, self.katcp_sensor.units,
                              params)

    def store_strategy(self, strategy, param):
        """Store sensor strategy if it has changed."""
        if strategy == self.strategy and param == self.param:
            return
        self.strategy = strategy
        self.param = param
        logger.info("Registered KATCP sensor %r with strategy (%r, %r) and SPEAD id 0x%x" %
                    (self.name, self.strategy, self.param, self.spead_id))

    def listen(self, update_seconds, value_seconds, status, value_string):
        """Callback that pushes KATCP sensor update to SPEAD stream.

        Parameters
        ----------
        update_seconds : float
            Unix timestamp indicating when update was received by local client
        value_seconds : float
            Unix timestamp indicating when sensor value was measured
        status : string
            Status of this update ('nominal' if all is well)
        value_string : string
            Sensor value encoded as a KATCP string

        """
        # Force value to be accepted by discrete sensor
        if self._sensor.stype == 'discrete':
            self._sensor._kattype._values.append(value_string)
            self._sensor._kattype._valid_values.add(value_string)
        # First convert value string to intended type to get appropriate repr()
        value = self._sensor.parse_value(value_string)
        # All KATCP events are sent as strings containing space-separated
        # value_timestamp + status + value, regardless of KATCP type
        # (consistent with the fact that KATCP data is string-based on the wire)
        update = "%r %s %r" % (value_seconds, status, value)
        logger.debug("Updating sensor %r: %s" % (self.name, update))
        # A lock is needed because each KATCP device client runs in its own
        # thread while calling this callback and the main SPEAD item group
        # of the server is shared among them (blame ig.get_heap()...)
        if self.server.streaming:
            with self.server._spead_lock:
                self.server.ig[self.name] = update
                # Transmit event-based updates immediately, while other updates
                # are periodically resampled in a separate thread
                if self.strategy == 'event':
                    self.server.transmit(self.server.ig.get_heap())
        self.last_update = update

    def start_listening(self):
        """Start listening to sensor and send updates to SPEAD stream."""
        if not self.listening:
            self.katcp_sensor.set_strategy(self.strategy, self.param)
            self.katcp_sensor.register_listener(self.listen)
            # This triggers the callback to obtain a valid last_update
            self.katcp_sensor.get_value()
            logger.debug("Start listening to sensor %r" % (self.name,))
            self.listening = True

    def stop_listening(self):
        """Stop listening to sensor and stop updates to SPEAD stream."""
        if self.listening:
            self.katcp_sensor.unregister_listener(self.listen)
            self.katcp_sensor.set_strategy('none')
            logger.debug("Stopped listening to sensor %r" % (self.name,))
            self.listening = False


class TransmitThread(threading.Thread):
    """Thread which transmits SPEAD heaps to a particular destination."""
    def __init__(self, spead_host, spead_port):
        threading.Thread.__init__(self)
        self.name = 'SpeadTxThread(%s:%d)' % (spead_host, spead_port)
        self.mailbox = Queue.Queue()
        self._transmit = spead.Transmitter(spead.TransportUDPtx(spead_host,
                                                                spead_port))
        self._thread_active = True

    def run(self):
        # wait for packets to be added to the queue, then transmit them
        while self._thread_active:
            try:
                # timeout necessary to stop waiting for queue item,
                # after the destination is removed
                heap = self.mailbox.get(block=True, timeout=0.5)
                self._transmit.send_heap(heap)
            except Queue.Empty:
                pass
        # send final spead "stop" packet
        self._transmit.end()

    def stop(self):
        self._thread_active = False


class Cam2SpeadDeviceServer(DeviceServer):
    """Device server that receives CAM events and pushes them to SPEAD stream.

    This receives a set of CAM attributes (static items) and listens to a set
    of KATCP sensors (dynamic items), assembles these as SPEAD items in a heap
    and pushes the heap onto a SPEAD stream via a separate transmitter thread.

    Parameters
    ----------
    attributes : dict mapping string to string
        Attributes as key-value string pairs which are streamed once upfront
    all_sensors : group of :class:`katcp.Sensor` objects
        Object (e.g. a :class:`katcorelib.ObjectGroup`) with all available
        sensors as attributes
    sensor_list : list of tuples of 3 strings
        List of sensors to listen to, and corresponding sensor strategy to be
        set as (name, strategy, param) tuple
    tx_period : float
        Non-event based sensor updates will be periodically resampled with
        this period in seconds and collated into a single SPEAD packet

    """

    VERSION_INFO = ("cam2spead", 0, 1)
    BUILD_INFO = ("cam2spead", 0, 1, __version__)

    def __init__(self, attributes, all_sensors, sensor_list, tx_period, *args, **kwargs):
        super(Cam2SpeadDeviceServer, self).__init__(*args, **kwargs)
        self.attributes = attributes
        self.sensors, self.sensor_strategies = all_sensors, sensor_list
        self._spead_lock = threading.Lock()
        self.sensor_bridges = {}
        self.streaming = False
        self.destinations = {}
        self.tx_period = float(tx_period)
        self.tx_thread = None
        self.init_heap = None
        self.ig = None

    def setup_sensors(self):
        """Populate sensor objects on server (none so far)."""
        pass

    def register_sensors(self):
        """Register all requested KATCP sensors, skipping the unknown ones."""
        for name, strategy, param in self.sensor_strategies:
            if name not in self.sensor_bridges:
                try:
                    sensor = getattr(self.sensors, name)
                except AttributeError:
                    logger.warning("Could not register unavailable KATCP sensor %r" % (name,))
                    continue
                self.sensor_bridges[name] = SensorBridge(name, sensor, self)
            # It is possible to change the strategy on an existing sensor bridge
            self.sensor_bridges[name].store_strategy(strategy, param)

    def start_listening(self):
        """Start listening to all registered sensors."""
        for bridge in self.sensor_bridges.itervalues():
            bridge.start_listening()

    def stop_listening(self):
        """Stop listening to all registered sensors."""
        for bridge in self.sensor_bridges.itervalues():
            bridge.stop_listening()

    def initial_spead_heap(self):
        """This creates the SPEAD item structure and fills in attributes."""
        self.ig = spead.ItemGroup()
        spead_id = SensorBridge.next_available_spead_id
        for name, value in self.attributes.items():
            logger.debug("Registering attribute %r with SPEAD id 0x%x and value %s" %
                         (name, spead_id, value))
            self.ig.add_item(name=name, id=spead_id, description='todo',
                             shape=-1, fmt=spead.mkfmt(('s', 8)), init_val=value)
            spead_id += 1
        SensorBridge.next_available_spead_id = spead_id
        for name, bridge in self.sensor_bridges.iteritems():
            logger.debug("Adding info for sensor %r (id 0x%x) to initial heap: %s" %
                         (name, bridge.spead_id, bridge.last_update))
            self.ig.add_item(name=name, id=bridge.spead_id,
                             description=bridge.katcp_sensor.description,
                             shape=-1, fmt=spead.mkfmt(('s', 8)),
                             init_val=bridge.last_update)
        return self.ig.get_heap()

    def start_destination(self, spead_host, spead_port):
        """Add destination for SPEAD stream and optionally start the thread."""
        dest = (spead_host, spead_port)
        thread = self.destinations.get(dest, None)
        # If the stream has already started, create thread and join the fun
        if not thread and self.streaming:
            thread = TransmitThread(spead_host, spead_port)
            thread.start()
            thread.mailbox.put(copy.deepcopy(self.init_heap))
            logger.debug("Started %s and sent initial SPEAD packet with %d items" %
                         (thread.name, len(self.ig.ids())))
        self.destinations[dest] = thread

    def stop_destination(self, spead_host, spead_port):
        """Remove destination for SPEAD stream and optionally stop the thread."""
        dest = (spead_host, spead_port)
        thread = self.destinations.get(dest, None)
        # Stop transmitter thread if running
        if thread and thread.is_alive():
            thread.stop()
            thread.join()
            logger.debug("Stopped %s" % (thread.name,))
        del self.destinations[dest]

    def transmit(self, heap):
        """Transmit SPEAD heap to all active destinations."""
        for thread in self.destinations.itervalues():
            thread.mailbox.put(copy.deepcopy(heap))

    def collate_and_transmit(self):
        """Periodically collate sensor updates and pass to transmitter threads."""
        transmit_time = 0.0
        while self.streaming:
            time_till_flush = max(self.tx_period - transmit_time, 0.0)
            time.sleep(time_till_flush)
            start = time.time()
            # Transmit current heap to all active destinations
            # This will contain only the latest sensor values and will ignore
            # other prior updates occurring after the previous periodic transmit
            with self._spead_lock:
                self.transmit(self.ig.get_heap())
            transmit_time = time.time() - start

    @return_reply(Str())
    def request_start_stream(self, req, msg):
        """Start the SPEAD stream of KATCP sensor data."""
        self.register_sensors()
        self.start_listening()
        self.init_heap = self.initial_spead_heap()
        self.streaming = True
        # Start all existing SPEAD transmitter threads and send initial heap
        for dest in self.destinations:
            self.start_destination(*dest)
        # Start periodic collation thread
        # (automatically terminates when stream is stopped)
        self.tx_thread = threading.Thread(target=self.collate_and_transmit,
                                          name='PeriodicCollationThread')
        self.tx_thread.start()
        smsg = "SPEAD stream started"
        logger.info(smsg)
        return ("ok", smsg)

    @request(Str(), Int())
    @return_reply(Str())
    def request_add_destination(self, req, spead_host, spead_port):
        """Add destination for SPEAD stream."""
        self.start_destination(spead_host, spead_port)
        smsg = "Added thread transmitting SPEAD to port %s on %s" \
               % (spead_port, spead_host)
        logger.info(smsg)
        return ("ok", smsg)

    @request(Str(), Int())
    @return_reply(Str())
    def request_remove_destination(self, req, spead_host, spead_port):
        """Remove destination for SPEAD stream."""
        self.stop_destination(spead_host, spead_port)
        smsg = "Removed thread transmitting SPEAD to port %s on %s" % \
               (spead_port, spead_host)
        logger.info(smsg)
        return ("ok", smsg)

    @return_reply(Str())
    def request_stop_stream(self, req, msg):
        """Stop the SPEAD stream of KATCP sensor data."""
        self.streaming = False
        self.stop_listening()
        # Ensure periodic collation thread is done
        if self.tx_thread:
            self.tx_thread.join()
        # Stop all SPEAD transmitter threads
        for dest in self.destinations.keys():
            self.stop_destination(*dest)
        smsg = "SPEAD stream stopped"
        logger.info(smsg)
        return ("ok", smsg)