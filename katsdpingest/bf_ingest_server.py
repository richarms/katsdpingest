from __future__ import print_function, division, absolute_import
import spead2
import spead2.recv
import spead2.recv.trollius
import katcp
from katcp.kattypes import request, return_reply
from katsdpfilewriter import telescope_model, ar1_model, file_writer
from ._bf_ingest_session import Session, SessionConfig
import h5py
import numpy as np
import time
import os
import os.path
import trollius
from trollius import From
import tornado
import logging
import katsdpingest
import psutil
import ipaddress
import netifaces
import socket
import contextlib


_logger = logging.getLogger(__name__)


def _get_interface_address(interface):
    return netifaces.ifaddresses(interface)[netifaces.AF_INET][0]['addr']


class _CaptureSession(object):
    """Object encapsulating a co-routine that runs for a single capture session
    (from ``capture-init`` to end of stream or ``capture-done``.

    Parameters
    ----------
    args : :class:`argparse.Namespace`
        Command-line arguments. See :class:`CaptureServer`.
    loop : :class:`trollius.BaseEventLoop`
        IO Loop for the coroutine

    Attributes
    ----------
    _args : :class:`argparse.Namespace`
        Command-line arguments passed to the constructor
    _loop : :class:`trollius.BaseEventLoop`
        Event loop passed to the constructor
    _filename : :class:`str`
        Filename of the HDF5 file written
    _session : :class:`katsdpingest._bf_ingest_session.Session`
        C++-driven capture session
    """
    def __init__(self, args, loop):
        self._loop = loop
        self._args = args
        self._filename = os.path.join(args.file_base, '{}.h5'.format(int(time.time())))

        endpoint = args.cbf_spead[0]  # TODO: make it a singleton
        address = socket.gethostbyname(endpoint.host)
        # ipaddress module requires a unicode string
        address_obj = ipaddress.ip_address(unicode(address, 'us-ascii'))
        if not address_obj.is_multicast:
            raise ValueError('Only multicast IP addresses are supported (not {})'.format(address))
        interface_address = _get_interface_address(args.interface)
        config = SessionConfig(self._filename, address, endpoint.port, interface_address)
        if args.cbf_channels:
            config.total_channels = args.cbf_channels
        else:
            _logger.info('Assuming %d PFB channels; if not, pass --cbf-channels', config.total_channels)
        if args.affinity:
            config.disk_affinity = args.affinity[0]
            config.network_affinity = args.affinity[1]

        self._session = Session(config)

    def _write_metadata(self):
        telstate = self._args.telstate
        try:
            sync_time = telstate['cbf_sync_time']
            scale_factor_timestamp = telstate['cbf_scale_factor_timestamp']
            first_timestamp = sync_time + self._session.first_timestamp / scale_factor_timestamp
        except KeyError:
            _logger.warn('Failed to get timestamp conversion items, so skipping metadata')
            return
        antenna_mask = telstate.get('config', {}).get('antenna_mask', '').split(',')
        model = ar1_model.create_model(antenna_mask)
        model_data = telescope_model.TelstateModelData(model, telstate, first_timestamp)
        h5file = h5py.File(self._filename, 'r+')
        with contextlib.closing(h5file):
            file_writer.set_telescope_model(h5file, model_data)
            file_writer.set_telescope_state(h5file, telstate)

    def _finalise(self):
        self._session.stop()
        if self._session.n_dumps > 0:
            # Write the metadata to file
            if self._args.telstate is not None:
                self._write_metadata()
        # TODO: log statistics
        _logger.info('Capture complete, %d heaps', self._session.n_dumps)
        self._session = None

    @trollius.coroutine
    def stop(self):
        """Shut down the stream and wait for the session. This
        is a coroutine.
        """
        yield From(self._loop.run_in_executor(None, self._finalise))


class CaptureServer(object):
    """Beamformer capture. This contains all the core functionality of the
    katcp device server, without depending on katcp. It is split like this
    to facilitate unit testing.

    Parameters
    ----------
    args : :class:`argparse.Namespace`
        Command-line arguments. The following arguments are required. Refer to
        the script for documentation of these options.

        - cbf_channels
        - cbf_spead
        - file_base
        - buffer
        - affinity

    loop : :class:`trollius.BaseEventLoop`
        IO Loop for running coroutines

    Attributes
    ----------
    capturing : :class:`bool`
        Whether a capture session is in progress. Note that a session is
        considered to be in progress until explicitly stopped with
        :class:`stop_capture`, even if the stream has terminated.
    _args : :class:`argparse.Namespace`
        Command-line arguments passed to constructor
    _loop : :class:`trollius.BaseEventLoop`
        IO Loop passed to constructor
    _capture : :class:`_CaptureSession`
        Current capture session, or ``None`` if not capturing
    """
    def __init__(self, args, loop):
        self._args = args
        self._loop = loop
        self._capture = None

    @property
    def capturing(self):
        return self._capture is not None

    def start_capture(self):
        """Start capture to file, if not already in progress."""
        if self._capture is None:
            self._capture = _CaptureSession(self._args, self._loop)

    @trollius.coroutine
    def stop_capture(self):
        """Stop capture to file, if currently running. This is a co-routine."""
        if self._capture is not None:
            yield From(self._capture.stop())
            self._capture = None


class KatcpCaptureServer(CaptureServer, katcp.DeviceServer):
    """katcp device server for beamformer capture.

    Parameters
    ----------
    args : :class:`argparse.Namespace`
        Command-line arguments (see :class:`CaptureServer`).
        The following additional arguments are required:

        host
          Hostname to bind to ('' for none)
        port
          Port number to bind to
    loop : :class:`trollius.BaseEventLoop`
        IO Loop for running coroutines
    """

    VERSION_INFO = ('bf-ingest', 1, 0)
    BUILD_INFO = ('katsdpingest',) + tuple(katsdpingest.__version__.split('.', 1)) + ('',)

    def __init__(self, args, loop):
        CaptureServer.__init__(self, args, loop)
        katcp.DeviceServer.__init__(self, args.host, args.port)

    def setup_sensors(self):
        pass

    @request()
    @return_reply()
    def request_capture_init(self, sock):
        """Start capture to file."""
        if self.capturing:
            return ('fail', 'already capturing')
        stat = os.statvfs(self._args.file_base)
        if stat.f_bavail / stat.f_blocks < 0.05:
            return ('fail', 'less than 5% disk space free on {}'.format(os.path.abspath(self._args.file_base)))
        self.start_capture()
        return ('ok',)

    @tornado.gen.coroutine
    def _stop_capture(self):
        """Tornado variant of :meth:`stop_capture`"""
        stop_future = trollius.async(self.stop_capture(), loop=self._loop)
        yield tornado.platform.asyncio.to_tornado_future(stop_future)

    @request()
    @return_reply()
    @tornado.gen.coroutine
    def request_capture_done(self, sock):
        """Stop a capture that is in progress."""
        if not self.capturing:
            raise tornado.gen.Return(('fail', 'not capturing'))
        yield self._stop_capture()
        raise tornado.gen.Return(('ok',))

    @tornado.gen.coroutine
    def stop(self):
        yield self._stop_capture()
        yield katcp.DeviceServer.stop(self)

    stop.__doc__ = katcp.DeviceServer.stop.__doc__


__all__ = ['CaptureServer', 'KatcpCaptureServer']
