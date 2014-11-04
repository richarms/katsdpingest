#!/usr/bin/python

# Threads for ingesting data and meta-data in order to produce a complete HDF5 file for further
# processing.
#
# Currently has a CBFIngest and CAMIngest class
#
# Details on these are provided in the class documentation

import numpy as np
import threading
import spead
import time
import copy
import katsdpingest.sigproc as sp
import katsdpsigproc.accel as accel
import katsdpsigproc.rfi.device as rfi
import logging
import socket
import struct


timestamps_dataset = '/Data/timestamps'
flags_dataset = '/Data/flags'
cbf_data_dataset = '/Data/correlator_data'
sdisp_ips = {}
 # dict storing the configured signal destination ip addresses
MULTICAST_PREFIXES = ['224', '239']
 # list of prefixes used to determine a multicast address


class CAMIngest(threading.Thread):
    """The CAM Ingest class receives meta-data updates in the form
    of sensor information from the CAM via SPEAD. It uses these to
    update a model of the telescope that is specific to the current
    ingest configuration (subarray)."""
    def __init__(self, spead_host, spead_port, h5_file, model, logger):
        self.logger = logger
        self.spead_host = spead_host
        self.spead_port = spead_port
        self.h5_file = h5_file
        self.model = model
        self.ig = None
        threading.Thread.__init__(self)

    def enable_debug(self, debug):
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

    def run(self):
        self.ig = spead.ItemGroup()
        self.logger.debug("Initalising SPEAD transports at %f" % time.time())
        self.logger.info("CAM SPEAD stream reception on {0}:{1}".format(self.spead_host, self.spead_port))
        if self.spead_host[:self.spead_host.find('.')] in MULTICAST_PREFIXES:
         # if we have a multicast address we need to subscribe to the appropriate groups...
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if self.spead_host.rfind("+") > 0:
                host_base, host_number = self.spead_host.split("+")
                hosts = ["{0}.{1}".format(host_base[:host_base.rfind('.')],int(host_base[host_base.rfind('.')+1:])+x) for x in range(int(host_number)+1)]
            else:
                hosts = [self.spead_host]
            for h in hosts:
                mreq = struct.pack("4sl", socket.inet_aton(h), socket.INADDR_ANY)
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
                 # subscribe to each of these hosts
            self.logger.info("Subscribing to the following multicast addresses: {0}".format(hosts))
        rx_md = spead.TransportUDPrx(self.spead_port)

        for heap in spead.iterheaps(rx_md):
            self.ig.update(heap)
            self.model.update_from_ig(self.ig)

        self.logger.info("CAM ingest thread complete at %f" % time.time())


class CBFIngest(threading.Thread):
    @classmethod
    def _create_proc_template(cls, context):
        flag_value = 1 << sp.IngestTemplate.flag_names.index('detected_rfi')
        # TODO: these parameters should probably come from somewhere else
        # (particularly cont_factor).
        background_template = rfi.BackgroundMedianFilterDeviceTemplate(context, width=13)
        noise_est_template = rfi.NoiseEstMADTDeviceTemplate(context, max_channels=10240)
        threshold_template = rfi.ThresholdSimpleDeviceTemplate(
                context, n_sigma=11.0, transposed=True, flag_value=flag_value)
        flagger_template = rfi.FlaggerDeviceTemplate(
                background_template, noise_est_template, threshold_template)
        return sp.IngestTemplate(context, flagger_template, cont_factor=16)

    def __init__(self, spead_host, spead_port, h5_file, my_sensors, model, cbf_name, logger):
        ## TODO: remove my_sensors and rather use the model to drive local sensor updates
        self.logger = logger
        self.spead_port = spead_port
        self.spead_host = spead_host
        self.h5_file = h5_file
        self.model = model
        self.cbf_name = cbf_name
        self.cbf_component = self.model.components[self.cbf_name]
        self.cbf_attr = self.cbf_component.attributes

        self._process_log_idx = 0
        self._my_sensors = my_sensors
        self.pkt_sensor = self._my_sensors['packets-captured']
        self.status_sensor = self._my_sensors['status']
        self.status_sensor.set_value("init")
        self._sd_metadata = None
        self.sdisp_ips = {}
        self._sd_count = 0
        self.center_freq = 0
        self.ig_sd = spead.ItemGroup()
        self.timestamps = []
         # temporary timestamp store
        self.sd_frame = None
        self.baseline_mask = None
         # by default record all baselines
        self._script_ants = None
         # a reference to the antennas requested from the current script
        #### Initialise processing blocks used
        self.context = accel.create_some_context(interactive=False)
        self.command_queue = self.context.create_command_queue()
        self.proc_template = self._create_proc_template(self.context)
        self.proc = None    # Instantiation of the template delayed until data shape is known
        self.flags_description = zip(self.proc_template.flag_names, self.proc_template.flag_descriptions)
         # an array describing the flags produced by the rfi flagger
        self.h5_file['/Data'].create_dataset('flags_description',data=self.flags_description)
         # insert flags descriptions into output file
        #### Done with blocks
        threading.Thread.__init__(self)

    def enable_debug(self, debug):
        if debug: self.logger.setLevel(logging.DEBUG)
        else: self.logger.setLevel(logging.INFO)

    def send_sd_data(self, data):
        #if self._sd_count % 10 == 0:
        #    logger.debug("Sending metadata heartbeat...")
        #    self.send_sd_metadata()

        for tx in self.sdisp_ips.itervalues():
            tx.send_heap(data)

        self._sd_count += 1

    def _update_sd_metadata(self):
        """Update the itemgroup for the signal display metadata to include any changes since last sent..."""
        self.ig_sd = spead.ItemGroup()
         # we need to clear the descriptor so as not to accidently send a signal display frame twice...
        if self.sd_frame is not None:
            self.ig_sd.add_item(name=('sd_data'),id=(0x3501), description="Combined raw data from all x engines.", ndarray=(self.sd_frame.dtype,self.sd_frame.shape))
            self.ig_sd.add_item(name=('sd_flags'),id=(0x3503), description="8bit packed flags for each data point.", ndarray=(np.dtype(np.uint8), self.sd_frame.shape[:-1]))
        self.ig_sd.add_item(name="center_freq",id=0x1011, description="The center frequency of the DBE in Hz, 64-bit IEEE floating-point number.",
                            shape=[],fmt=spead.mkfmt(('f',64)), init_val=self.center_freq)
        self.ig_sd.add_item(name=('sd_timestamp'), id=0x3502, description='Timestamp of this sd frame in centiseconds since epoch (40 bit limitation).',
                            shape=[], fmt=spead.mkfmt(('u',spead.ADDRSIZE)))
        self.ig_sd.add_item(name=('bls_ordering'), id=0x100C, description="Mapping of antenna/pol pairs to data output products.", init_val=self.cbf_attr['bls_ordering'].value)
        self.ig_sd.add_item(name="bandwidth",id=0x1013, description="The analogue bandwidth of the digitally processed signal in Hz.",
                            shape=[],fmt=spead.mkfmt(('f',64)), init_val=self.cbf_attr['bandwidth'].value)
        self.ig_sd.add_item(name="n_chans",id=0x1009, description="The total number of frequency channels present in any integration.",
                            shape=[], fmt=spead.mkfmt(('u',spead.ADDRSIZE)), init_val=self.cbf_attr['n_chans'].value)
        return copy.deepcopy(self.ig_sd.get_heap())

    def set_baseline_mask(self, bls_ordering):
        """Uses the _script_ants variable to set a baseline mask.
        This only works if script_ants has been set by an external process between capture_done and capture_start."""
        new_bls = bls_ordering
        if self._script_ants is not None:
            logger.info("Using script-ants (%s) as a custom baseline mask..." % self._script_ants)
            ants = self._script_ants.replace(" ","").split(",")
            if len(ants) > 0:
                b = bls_ordering.tolist()
                self.baseline_mask = [b.index(pair) for pair in b if pair[0][:-1] in ants and pair[1][:-1] in ants]
                new_bls = np.array([b[idx] for idx in self.baseline_mask])
                 # we need to recalculate the bls ordering as well...
        return new_bls

    @classmethod
    def baseline_permutation(cls, bls_ordering):
        """Construct a permutation to place the baselines into the desired
        order for internal processing.

        Parameters
        ----------
        bls_ordering : list of pairs of strings
            Names of inputs in current ordering

        Returns
        -------
        (list, list)
            The permutation giving the reordering and the new ordering
        """
        def key(item):
            ant1, ant2 = item[1]
            pol1 = ant1[-1]
            pol2 = ant2[-1]
            return (ant1[:-1] != ant2[:-1], pol1 != pol2, pol1, pol2)
        reordered = sorted(enumerate(bls_ordering), key=key)
        permutation = [x[0] for x in reordered]
        # permutation contains the mapping from new position to original
        # position, but we need the inverse. np.argsort inverts a permutation
        permutation = np.argsort(permutation)
        return permutation, np.array([x[1] for x in reordered])

    def send_sd_metadata(self):
        self._sd_metadata = self._update_sd_metadata()
        if self._sd_metadata is not None:
            for tx in self.sdisp_ips.itervalues():
                mdata = copy.deepcopy(self._sd_metadata)
                tx.send_heap(mdata)

    def write_process_log(self, process, args, revision):
        """Write an entry into the process log."""
        if self._process_log_idx > 0:
            self.h5_file['/History/process_log'].resize(self._process_log_idx+1, axis=0)
        self.h5_file['/History/process_log'][self._process_log_idx] = (process, args, revision)
        self._process_log_idx += 1

    def write_timestamps(self):
        """Write the accumulated timestamps into a dataset.
        Previously these timestamps were written alongside each received data frame, but this
        results in a highly fragmented timestamp array. This in turns leads to exceptionally long load
        times for this dataset, even though it contains very little data. By deferring writing, we can
        instead locate the timestamp data contiguously on disk and thus obviate the read overhead.

        As this MUST be called before the file is closed, it may get called multiple times as security to
        ensure that it is done - it is therefore safe to call multiple times."""
        if self.h5_file is not None:
            if timestamps_dataset not in self.h5_file:
            # explicit check for existence of timestamp dataset - we could rely on h5py exceptions, but these change
            # regularly - hence this check.
                if self.timestamps:
                    self.h5_file.create_dataset(timestamps_dataset,data=np.array(self.timestamps))
                    # create timestamp array before closing file. This means that it will be contiguous and hence much faster to read than if it was
                    # distributed across the entire file.
                else:
                    self.logger.warning("H5 file contains no data and hence no timestamps")
                    # exception if there is no data (and hence no timestamps) in the file.
        else: self.logger.warning("Write timestamps called, but h5 file already closed. No timestamps will be written.")

    def finalise(self):
        """Write any final information to file and mark file as not current."""
        if self.h5_file is not None:
            self.write_timestamps()
        self.h5_file = None

    def drop_sdisp_ip(self, ip):
        self.logger.info("Removing ip %s from the signal display list." % (ip))
        del self.sdisp_ips[ip]

    def add_sdisp_ip(self, ip, port):
        self.logger.info("Adding %s:%s to signal display list. Starting transport..." % (ip,port))
        self.sdisp_ips[ip] = spead.Transmitter(spead.TransportUDPtx(ip, port))
        if self._sd_metadata is not None:
            mdata = copy.deepcopy(self._sd_metadata)
            self.sdisp_ips[ip].send_heap(mdata)
             # new connection requires headers...

    def run(self):
        self.logger.debug("Initalising SPEAD transports at %f" % time.time())
        self.logger.info("CBF SPEAD stream reception on {0}:{1}".format(self.spead_host, self.spead_port))
        if self.spead_host[:self.spead_host.find('.')] in MULTICAST_PREFIXES:
         # if we have a multicast address we need to subscribe to the appropriate groups...
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if self.spead_host.rfind("+") > 0:
                host_base, host_number = self.spead_host.split("+")
                hosts = ["{0}.{1}".format(host_base[:host_base.rfind('.')],int(host_base[host_base.rfind('.')+1:])+x) for x in range(int(host_number)+1)]
            else:
                hosts = [self.spead_host]
            for h in hosts:
                mreq = struct.pack("4sl", socket.inet_aton(h), socket.INADDR_ANY)
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
                 # subscribe to each of these hosts
            self.logger.info("Subscribing to the following multicast addresses: {0}".format(hosts))
        rx = spead.TransportUDPrx(self.spead_port, pkt_count=1024, buffer_size=51200000)
        ig = spead.ItemGroup()
        idx = 0
        self.status_sensor.set_value("idle")
        datasets = {}
        datasets_index = {}
        current_dbe_target = ''
        dbe_target_since = 0.0
        current_ant_activities = {}
        ant_activities_since = {}
         # track the current DBE target and antenna activities via sensor updates
        sd_slots = None
        sd_timestamp = None

        for heap in spead.iterheaps(rx):
            st = time.time()
            if idx == 0:
                self.status_sensor.set_value("capturing")

            #### Update the telescope model

            ig.update(heap)
            self.model.update_from_ig(ig, proxy_path=self.cbf_name)
             # any interesting attributes will now end up in the model
             # this means we are only really interested in actual data now
            if not ig._names.has_key('xeng_raw'): self.logger.warning("CBF Data received but either no metadata or xeng_raw group is present"); continue
            if not ig._names.has_key('timestamp'): self.logger.warning("No timestamp received for current data frame - discarding"); continue
            data_ts = ig['timestamp']
            data_item = ig.get_item('xeng_raw')
            if not data_item._changed: self.logger.debug("Xeng_raw is unchanged"); continue
             # we have new data...

             # check to see if our CBF model is valid
             # i.e. make sure any attributes marked as critical are present
            if not self.cbf_component.is_valid(check_sensors=False):
                self.logger.warning("CBF Component Model is not currently valid as critical attribute items are missing. Data will be discarded until these become available.")
                continue

            ##### Configure datasets and other items now that we have complete metedata

            if idx == 0:
                # set up baseline mask
                self.baseline_mask = range(self.cbf_attr['n_bls'].value)
                 # default mask is to include all known baselines
                if self._script_ants is not None:
                 # we need to calculate a baseline_mask to match the specified script_ants
                    self.cbf_attr['bls_ordering'].value = self.set_baseline_mask(self.cbf_attr['bls_ordering'].value)

                 # we need to create the raw and timestamp datasets.
                new_shape = list(data_item.shape)
                new_shape[-2] = len(self.baseline_mask)
                self.logger.info("Creating cbf_data dataset with shape: {0}, dtype: {1}".format(str(new_shape),np.float32))
                self.h5_file.create_dataset(cbf_data_dataset, [1] + new_shape, maxshape=[None] + new_shape, dtype=np.float32)
                self.h5_file.create_dataset(flags_dataset, [1] + new_shape[:-1], maxshape=[None] + new_shape[:-1], dtype=np.uint8)

                # configure the signal processing blocks
                channels = self.cbf_attr['n_chans'].value
                baselines = len(self.baseline_mask)
                self.proc = self.proc_template.instantiate(
                        self.command_queue, channels, (0, channels), baselines)
                self.proc.set_scale(1.0 / self.cbf_attr['n_accs'].value)
                self.proc.ensure_all_bound()
                permutation, self.cbf_attr['bls_ordering'] = self.baseline_permutation(self.cbf_attr['bls_ordering'].value)
                self.proc.buffer('permutation').set(self.command_queue, np.asarray(permutation, dtype=np.uint16))

                # TODO: configure van_vleck once implemented
                for description in self.proc.descriptions():
                    self.write_process_log(*description)
            else:
                 # resize datasets
                self.h5_file[cbf_data_dataset].resize(idx+1, axis=0)
                self.h5_file[flags_dataset].resize(idx+1, axis=0)

            if self.sd_frame is None:
                self.sd_frame = np.zeros((self.cbf_attr['n_chans'].value,len(self.baseline_mask),2),dtype=np.float32)
                 # initialise the signal display data frame

            if sd_slots is None:
                self.sd_frame.dtype = np.dtype(np.float32)
                 # make sure we have the right dtype for the sd data
                sd_slots = np.zeros(self.cbf_attr['n_chans'].value/data_item.shape[0])
                self.send_sd_metadata()

            ##### Generate timestamps
            current_ts = self.cbf_attr['sync_time'].value + (data_ts / self.cbf_attr['scale_factor_timestamp'].value)
            self._my_sensors["last-dump-timestamp"].set_value(current_ts)
            self.timestamps.append(current_ts)

            ##### Perform data processing

            masked_data = data_item.get_value()[...,self.baseline_mask,:]
            self.proc.buffer('vis_in').set(self.command_queue, masked_data)
            self.start_sum()
            self.proc()
            self.end_sum()

            flags = self.proc.buffer('spec_flags').get(self.command_queue)
            vis = self.proc.buffer('spec_vis').get(self.command_queue)
            # Complex values are written to file as an extra dimension of size 2,
            # rather than as structs
            vis = vis.view(np.float32).reshape(list(vis.shape) + [2])
            self.h5_file[flags_dataset][idx] = flags
            self.h5_file[cbf_data_dataset][idx] = vis

            #### Send signal display information
            self.ig_sd['sd_timestamp'] = int(current_ts * 100)
            self.ig_sd['sd_data'] = vis
             # send out a copy of the data we are writing to disk. In the future this will need to be rate limited to some extent
            self.ig_sd['sd_flags'] = flags
             # send out RFI flags with the data
            self.send_sd_data(self.ig_sd.get_heap())

            #### Done with writing this frame
            idx += 1
            if self.h5_file is not None: self.h5_file.flush()
            self.pkt_sensor.set_value(idx)
            tt = time.time() - st
            self.logger.info("Captured CBF dump with timestamp %i (local: %.3f, process_time: %.2f, index: %i)" % (current_ts, tt+st, tt, idx))

        #### Stop received.

        self.logger.info("CBF ingest complete at %f" % time.time())
        self.logger.debug("\nProcessing Blocks\n=================\n%s\n%s\n" % (self.scale,self.rfi))
        self.status_sensor.set_value("complete")
