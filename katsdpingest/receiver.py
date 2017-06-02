"""Receives from multiple SPEAD streams and combines heaps into frames."""

from __future__ import print_function, absolute_import, division
import logging
import multiprocessing
from collections import deque
import spead2
import spead2.recv.trollius
import trollius
from trollius import From, Return
import numpy as np
from . import utils
from .utils import Range


_logger = logging.getLogger(__name__)
# CBF SPEAD metadata items that should be stored as sensors rather than
# attributes. Don't use this directly; use :func:`is_cbf_sensor` instead,
# which handles cases that aren't fixed strings.
CBF_SPEAD_SENSORS = frozenset(["flags_xeng_raw"])
# Attributes that are required for data to be correctly ingested
CBF_CRITICAL_ATTRS = frozenset([
    'adc_sample_rate', 'n_chans', 'n_accs', 'bls_ordering',
    'bandwidth', 'center_freq',
    'sync_time', 'int_time', 'scale_factor_timestamp', 'ticks_between_spectra'])


def is_cbf_sensor(name):
    return name in CBF_SPEAD_SENSORS or name.startswith('eq_coef_')


class Frame(object):
    """A group of xeng_raw data with a common timestamp"""
    def __init__(self, timestamp, n_xengs):
        self.timestamp = timestamp
        self.items = [None] * n_xengs

    def ready(self):
        return all(item is not None for item in self.items)

    def empty(self):
        return all(item is None for item in self.items)

    @property
    def nbytes(self):
        return sum([(item.nbytes if item is not None else 0) for item in self.items])


class Receiver(object):
    """Class that receives from multiple SPEAD streams and combines heaps into
    frames. It also collects CBF metadata from the first stream and uses it to
    populate telescope state.

    Parameters
    ----------
    endpoints : list of :class:`katsdptelstate.Endpoint`
        Endpoints for SPEAD streams. These must be listed in order
        of increasing channel number.
    interface : str
        Name of interface to subscribe to for endpoints
    channel_range : :class:`katsdpingest.utils.Range`
        Channels to capture. These must be aligned to the stream boundaries.
    cbf_channels : int
        Total number of channels represented by `endpoints`.
    sensors : dict
        Dictionary mapping sensor names to sensor objects
    telstate : :class:`katsdptelstate.TelescopeState`, optional
        Telescope state to be populated with CBF attributes
    cbf_name : str
        Name to prepend to CBF metadata in telstate
    active_frames : int, optional
        Maximum number of incomplete frames to keep at one time
    loop : :class:`trollius.BaseEventLoop`, optional
        I/O loop used for asynchronous operations

    Attributes
    ----------
    telstate : :class:`katsdptelstate.TelescopeState`, optional
        Telescope state passed to constructor
    cbf_attr : dict
        Attributes read from CBF metadata when available. Otherwise populated from telstate.
    cbf_name : str
        Value of `cbf_name` passed to the constructor
    active_frames : int
        Value of `active_frames` passed to constructor
    _interval : int
        Expected timestamp change between successive frames. This is initially ``None``,
        and is computed once the necessary metadata is available.
    _frames : :class:`deque`
        Deque of :class:`Frame` objects representing incomplete frames. After
        initialization, it always contains exactly `active_frames`
        elements, with timestamps separated by the inter-dump interval.
    _frames_complete : :class:`trollius.Queue`
        Queue of complete frames of type :class:`Frame`. It may also contain
        integers, which are the numbers of finished streams.
    _running : int
        Number of streams still running
    _futures : list of :class:`trollius.Future`
        Futures associated with each call to :meth:`_read_stream`
    _streams : list of :class:`spead2.recv.trollius.Stream`
        Individual SPEAD streams
    """
    def __init__(self, endpoints, interface, channel_range, cbf_channels, sensors,
                 telstate=None, cbf_name='cbf', active_frames=2, loop=None):
        # Determine the endpoints to actually use
        if cbf_channels % len(endpoints):
            raise ValueError('cbf_channels not divisible by the number of endpoints')
        stream_channels = cbf_channels // len(endpoints)
        if not channel_range.isaligned(stream_channels):
            raise ValueError('channel_range is not aligned to the stream boundaries')
        use_endpoints = endpoints[channel_range.start // stream_channels :
                                  channel_range.stop // stream_channels]

        if loop is None:
            loop = trollius.get_event_loop()
        self.telstate = telstate
        self.cbf_attr = {}
        self.cbf_name = cbf_name
        self.n_chans = cbf_channels
        self.active_frames = active_frames
        self.channel_range = channel_range
        self.cbf_channels = cbf_channels
        self._interface_address = utils.get_interface_address(interface)
        self._streams = [None] * len(use_endpoints)
        self._frames = None
        self._frames_complete = trollius.Queue(maxsize=1, loop=loop)
        self._futures = []
        self._interval = None
        self._loop = loop
        self._input_bytes = 0
        self._input_heaps = 0
        self._input_dumps = 0
        self._input_bytes_sensor = sensors['input-bytes-total']
        self._input_bytes_sensor.set_value(0)
        self._input_heaps_sensor = sensors['input-heaps-total']
        self._input_heaps_sensor.set_value(0)
        self._input_dumps_sensor = sensors['input-dumps-total']
        self._input_dumps_sensor.set_value(0)
        # If we have a large number of streams, we avoid creating more
        # threads than CPU cores.
        thread_pool = spead2.ThreadPool(min(multiprocessing.cpu_count(), len(use_endpoints)))
        for i, endpoint in enumerate(use_endpoints):
            self._futures.append(trollius.async(self._read_stream(thread_pool, endpoint, i), loop=loop))
        self._running = len(use_endpoints)
        _logger.info("CBF SPEAD stream reception on %s via %s",
            [str(x) for x in use_endpoints],
            interface if interface is not None else 'default interface')

    def stop(self):
        """Stop all the individual streams."""
        for stream in self._streams:
            if stream is not None:
                stream.stop()

    @trollius.coroutine
    def join(self):
        """Wait for all the individual streams to stop. This must not
        be called concurrently with :meth:`get`.

        This is a coroutine.
        """
        while self._running > 0:
            frame = yield From(self._frames_complete.get())
            if isinstance(frame, int):
                yield From(self._futures[frame])
                self._futures[frame] = None
                self._running -= 1

    def _update_telstate(self, updated):
        """Updates the telescope state from new values in the item group."""
        for item_name, item in updated.iteritems():
            if item_name not in ['timestamp', 'frequency', 'xeng_raw']:
                # store as an attribute unless item is a sensor (e.g. flags_xeng_raw)
                utils.set_telstate_entry(self.telstate, item_name, item.value,
                                         prefix=self.cbf_name,
                                         attribute=not is_cbf_sensor(item_name))

    def _update_cbf_attr_from_telstate(self):
        """Look for any of the critical CBF sensors in telstate and use these to populate
        the cbf_attrs dict."""
        if self.telstate is not None:
            for critical_attr in CBF_CRITICAL_ATTRS:
                cval = self.telstate.get("{}_{}".format(self.cbf_name, critical_attr))
                if cval is not None and critical_attr not in self.cbf_attr:
                    self.cbf_attr[critical_attr] = cval
                    _logger.info("Set critical cbf attribute from telstate: {} => {}".format(critical_attr, cval))

    def _update_cbf_attr(self, updated):
        """Updates the internal cbf_attr dictionary from new values in the item group."""
        for item_name, item in updated.iteritems():
            if (item_name not in ['timestamp', 'frequency', 'xeng_raw'] and
                    not is_cbf_sensor(item_name) and
                    item.value is not None):
                if item_name not in self.cbf_attr:
                    self.cbf_attr[item_name] = item.value
                elif not np.array_equal(self.cbf_attr[item_name], item.value):
                    _logger.warning('Item %s is already set to %s, not setting to %s',
                                    item_name, self.cbf_attr[item_name], item.value)

    def _pop_frame(self):
        """Remove the oldest element of :attr:`_frames`, and replace it with
        a new frame at the other end.
        """
        xengs = len(self._frames[-1].items)
        next_timestamp = self._frames[-1].timestamp + self._interval
        self._frames.popleft()
        self._frames.append(Frame(next_timestamp, xengs))

    @trollius.coroutine
    def _put_frame(self, frame):
        """Put a frame onto :attr:`_frames_complete` and update the sensor."""
        self._input_dumps += 1
        self._input_dumps_sensor.set_value(self._input_dumps)
        yield From(self._frames_complete.put(frame))

    @trollius.coroutine
    def _flush_frames(self):
        """Remove any completed frames from the head of :attr:`_frames`."""
        while self._frames[0].ready():
            # Note: _pop_frame must be done *before* trying to put the
            # item onto the queue, because other coroutines may run and
            # operate on _frames while we're waiting for space in the
            # queue.
            frame = self._frames[0]
            _logger.debug('Flushing frame with timestamp %d', frame.timestamp)
            self._pop_frame()
            yield From(self._put_frame(frame))

    def _have_metadata(self, ig_cbf):
        if not CBF_CRITICAL_ATTRS.issubset(self.cbf_attr.keys()):
            return False
        if 'xeng_raw' not in ig_cbf:
            return False
        if 'timestamp' not in ig_cbf:
            return False
        # TODO: once CBF switches to new format, require 'frequency' too
        if self._interval is None:
            self._interval = self.cbf_attr['ticks_between_spectra'] * self.cbf_attr['n_accs']
        return True

    def _add_reader(self, stream, endpoint):
        if self._interface_address is None:
            stream.add_udp_reader(endpoint.port, bind_hostname=endpoint.host)
        else:
            stream.add_udp_reader(endpoint.host, endpoint.port,
                                  interface_address=self._interface_address)

    @trollius.coroutine
    def _read_stream(self, thread_pool, endpoint, stream_idx):
        """Co-routine that sucks data from a single stream and populates
        :attr:`_frames_complete` and metadata."""
        try:
            # We can't tell how big to make the memory pool or how many heaps
            # we will be receiving in parallel until we have the metadata, but
            # we need this to set up the stream. To handle this, we'll make a
            # temporary stream for reading the metadata, then replace it once
            # we have the metadata.
            # TODO: we can now get it all from telstate.
            ring_heaps = 4
            stream = spead2.recv.trollius.Stream(thread_pool, max_heaps=4,
                                                 ring_heaps=ring_heaps, loop=self._loop)
            self._add_reader(stream, endpoint)
            self._streams[stream_idx] = stream
            ig_cbf = spead2.ItemGroup()
            if self.telstate is None:
                _logger.warning("No connection to telescope state available. Critical metadata must be available in SPEAD stream.")
            # We may already have critical metadata available in telstate
            self._update_cbf_attr_from_telstate()
            while not self._have_metadata(ig_cbf):
                try:
                    heap = yield From(stream.get())
                    updated = ig_cbf.update(heap)
                    # Keep trying telstate in the hope that critical metadata will arrive
                    self._update_cbf_attr_from_telstate()
                    # Harvest any remaining metadata from the SPEAD stream. Neither source is treated
                    # as authoratitive, it is on a first come first served basis.
                    self._update_cbf_attr(updated)
                    # Finally, we try to put any meta data received via SPEAD into telstate.
                    # If this meta data is already in telstate it is not overwritten.
                    self._update_telstate(updated)
                    if 'timestamp' in updated:
                        _logger.warning('Dropping heap with timestamp %d because metadata not ready',
                                        updated['timestamp'].value)
                except spead2.Stopped:
                    return
            stream.stop()

            # We have the metadata, so figure out how many heaps will have the
            # same timestamp, and set up the new stream.
            heap_data_size = np.product(ig_cbf['xeng_raw'].shape) * ig_cbf['xeng_raw'].dtype.itemsize
            heap_channels = ig_cbf['xeng_raw'].shape[0]
            stream_channels = len(self.channel_range) // len(self._streams)
            if stream_channels % heap_channels != 0:
                raise ValueError('Number of channels in xeng_raw does not divide into per-stream channels')
            xengs = len(self.channel_range) // heap_channels
            stream_xengs = stream_channels // heap_channels
            # CBF currently send 2 metadata heaps in a row, hence the + 2
            # We assume that each xengine will not overlap packets between
            # heaps, and that there is enough of a gap between heaps that
            # reordering in the network is a non-issue.
            max_heaps = stream_xengs + 2
            # We need space in the memory pool for:
            # - live heaps (max_heaps, plus a newly incoming heap)
            # - ringbuffer heaps
            # - per X-engine:
            #   - heap that has just been popped from the ringbuffer (1)
            #   - active frames
            #   - complete frames queue (1)
            #   - frame being processed by ingest_session (which could be several, depending on
            #     latency of the pipeline, but assume 3 to be on the safe side)
            memory_pool_heaps = ring_heaps + max_heaps + stream_xengs * (self.active_frames + 5)
            stream = spead2.recv.trollius.Stream(
                thread_pool,
                max_heaps=max_heaps,
                ring_heaps=ring_heaps, loop=self._loop)
            memory_pool = spead2.MemoryPool(16384, heap_data_size + 512,
                                            memory_pool_heaps, memory_pool_heaps)
            stream.set_memory_allocator(memory_pool)
            self._add_reader(stream, endpoint)
            self._streams[stream_idx] = stream

            # Ready to receive data (using the same item group, since it has
            # the descriptors)
            prev_ts = None
            ts_wrap_offset = 0        # Value added to compensate for CBF timestamp wrapping
            ts_wrap_period = 2**48
            while True:
                try:
                    heap = yield From(stream.get())
                except spead2.Stopped:
                    break
                updated = ig_cbf.update(heap)
                if 'xeng_raw' not in updated:
                    _logger.debug("CBF non-data heap received on stream %d", stream_idx)
                    continue
                if 'timestamp' not in updated:
                    _logger.warning("CBF heap without timestamp received on stream %d", stream_idx)
                    continue
                channel0 = 0
                if 'frequency' in updated:
                    channel0 = updated['frequency'].value
                else:
                    # Old format, with only one engine per stream
                    if stream_xengs != 1:
                        _logger.warning('CBF heap without frequency received on stream %d', stream_idx)
                        continue
                    channel0 = stream_channels * stream_idx
                heap_channel_range = Range(channel0, channel0 + heap_channels)
                if not (heap_channel_range.isaligned(heap_channels) and
                        heap_channel_range.issubset(self.channel_range)):
                    _logger.warning("CBF heap with invalid channel %d on stream %d", channel0, stream_idx)
                    continue
                xeng_idx = (channel0 - self.channel_range.start) // heap_channels

                data_ts = ig_cbf['timestamp'].value + ts_wrap_offset
                data_item = ig_cbf['xeng_raw'].value
                if prev_ts is not None and data_ts < prev_ts - ts_wrap_period // 2:
                    # This happens either because packets ended up out-of-order,
                    # or because the CBF timestamp wrapped. Out-of-order should
                    # jump backwards a tiny amount while wraps should jump back by
                    # close to ts_wrap_period.
                    ts_wrap_offset += ts_wrap_period
                    data_ts += ts_wrap_period
                    _logger.warning('Data timestamps wrapped')
                elif prev_ts is not None and data_ts > prev_ts + ts_wrap_period // 2:
                    # This happens if we wrapped, then received another heap
                    # (probably from a different X engine) from before the
                    # wrap. We need to undo the wrap.
                    ts_wrap_offset -= ts_wrap_period
                    data_ts -= ts_wrap_period
                    _logger.warning('Data timestamps reverse wrapped')
                _logger.debug('Received heap with timestamp %d on stream %d, channel %d', data_ts, stream_idx, channel0)
                prev_ts = data_ts
                # we have new data...

                self._input_bytes += data_item.nbytes
                self._input_heaps += 1
                self._input_bytes_sensor.set_value(self._input_bytes)
                self._input_heaps_sensor.set_value(self._input_heaps)
                if self._frames is None:
                    self._frames = deque()
                    for i in range(self.active_frames):
                        self._frames.append(Frame(data_ts + self._interval * i, xengs))
                ts0 = self._frames[0].timestamp
                if data_ts < ts0:
                    _logger.warning('Timestamp %d on stream %d is too far in the past, discarding',
                                    data_ts, stream_idx)
                    continue
                elif (data_ts - ts0) % self._interval != 0:
                    _logger.warning('Timestamp %d on stream %d does not match expected period, discarding',
                                    data_ts, stream_idx)
                    continue
                while data_ts >= ts0 + self._interval * self.active_frames:
                    frame = self._frames[0]
                    self._pop_frame()
                    if frame.empty():
                        _logger.warning('Frame with timestamp %d is empty, discarding', ts0)
                    else:
                        expected = len(frame.items)
                        actual = sum(item is not None for item in frame.items)
                        _logger.warning('Frame with timestamp %d is %d/%d complete', ts0,
                                        actual, expected)
                        yield From(self._put_frame(frame))
                    del frame   # Free it up, particularly if discarded
                    yield From(self._flush_frames())
                    ts0 = self._frames[0].timestamp
                frame_idx = (data_ts - ts0) // self._interval
                self._frames[frame_idx].items[xeng_idx] = data_item
                yield From(self._flush_frames())
        finally:
            yield From(self._frames_complete.put(stream_idx))

    @trollius.coroutine
    def get(self):
        """Return the next frame.

        This is a coroutine.

        Raises
        ------
        spead2.Stopped
            if all the streams have stopped
        """
        while self._running > 0:
            frame = yield From(self._frames_complete.get())
            if isinstance(frame, int):
                # It's actually the index of a finished stream
                self._streams[frame].stop()   # In case the co-routine exited with an exception
                yield From(self._futures[frame])
                self._futures[frame] = None
                self._running -= 1
            else:
                raise Return(frame)
        # Check for frames still in the queue
        while self._frames:
            frame = self._frames[0]
            self._frames.popleft()
            if frame.ready():
                _logger.debug('Flushing frame with timestamp %d', frame.timestamp)
                raise Return(frame)
            elif not frame.empty():
                _logger.warning('Frame with timestamp %d is incomplete, discarding', frame.timestamp)
        raise spead2.Stopped('End of streams')