import time
import logging
import threading

import tornado.ioloop

from concurrent.futures import Future
from tornado import gen


log = logging.getLogger(__name__)


def with_relative_timeout(timeout, future, io_loop=None):
    return gen.with_timeout(timeout + time.time(), future, io_loop)


class IOLoopManager(object):
    """Manages an IOLoop instance, optionally in a separate thread."""

    def __init__(self, managed_default=True, logger=log):
        # True if we manage the ioloop. Will be updated by self.set_ioloop()
        self._ioloop_managed = managed_default
        self._logger = logger
        # Thread object that a managed ioloop is running in
        self._ioloop_thread = None
        self._ioloop = None
        # Event that indicates that the ioloop is running.
        self._running = threading.Event()
        self._start_lock = threading.Lock()
        self._daemonize = False

    @property
    def managed(self):
        return self._ioloop_managed

    def get_ioloop(self):
        if not self._ioloop:
            if self._ioloop_managed:
                self.set_ioloop(tornado.ioloop.IOLoop())
            else:
                self.set_ioloop(tornado.ioloop.IOLoop.current())
        return self._ioloop

    def set_ioloop(self, ioloop, managed=None):
        if managed is not None:
            self._ioloop_managed = managed
        if self._ioloop:
            raise RuntimeError('IOLoop instance already set')
        self._ioloop = ioloop

    def _run_managed_ioloop(self):
        assert self._ioloop_managed

        def run_ioloop():
            try:
                self._ioloop.start()
                self._ioloop.close()
            except Exception:
                self._logger.error('Error running tornado IOloop: ',
                                   exc_info=True)
            finally:
                ioloop = self._ioloop
                self._ioloop = None
                self._running.clear()
                self._logger.info('Managed tornado IOloop {0} stopped'
                                  .format(ioloop))

        with self._start_lock:
            t = threading.Thread(target=run_ioloop)
            t.setDaemon(self._daemonize)
            try:
                if self._ioloop_thread.isAlive():
                    raise RuntimeError('Seems that managed ioloop has already been '
                                       'started, can only restart after stop()')
            except AttributeError:
                pass
            self._ioloop_thread = t
            self._ioloop_thread.start()

    def setDaemon(self, daemonic):
        """Set daemonic state of the managed ioloop thread to True / False

        Calling this method for a non-managed ioloop has no effect. Must be called before
        start(), or it will also have no effect
        """
        self._daemonize = bool(daemonic)

    def start(self, timeout=None):
        """Start managed ioloop thread, or do nothing if not managed.

        If a timeout is passed, it will block until the the event loop is alive
        (or the timeout expires) even if the ioloop is not managed.

        """
        if not self._ioloop:
            raise RuntimeError('Call get_ioloop() or set_ioloop() first')

        self._ioloop.add_callback(self._running.set)

        if self._ioloop_managed:
            self._run_managed_ioloop()
        else:            #  TODO this seems inconsistent with what the docstring describes
            self._running.set()

        if timeout:
            return self._running.wait(timeout)

    def stop(self, timeout=None, callback=None):
        """Stop ioloop (if managed) and call callback in ioloop before close.

        Parameters
        ----------
        timeout : float or None
            Seconds to wait for ioloop to have *started*.

        Returns
        -------
        stopped : thread-safe Future
            Resolves when the callback() is done

        """
        if timeout:
            self._running.wait(timeout)

        stopped_future = Future()

        @gen.coroutine
        def _stop():
            if callback:
                try:
                    yield gen.maybe_future(callback())
                except Exception:
                    self._logger.exception('Unhandled exception calling stop callback')
            if self._ioloop_managed:
                self._logger.info('Stopping ioloop {0!r}'.format(self._ioloop))
                # Allow ioloop to run once before stopping so that callbacks
                # scheduled by callback() above get a chance to run.
                yield gen.moment
                self._ioloop.stop()
            self._running.clear()

        try:
            self._ioloop.add_callback(
                lambda: gen.chain_future(_stop(), stopped_future))
        except AttributeError:
            # Probably we have been shut-down already
            pass

        return stopped_future

    def join(self, timeout=None):
        """Join managed ioloop thread, or do nothing if not managed."""
        if not self._ioloop_managed:
            # Do nothing if the loop is not managed
            return
        try:
            self._ioloop_thread.join(timeout)
        except AttributeError:
            raise RuntimeError('Cannot join if not started')
