import logging
import time
import collections
import unittest2 as unittest

import tornado
import mock

import katcp

from concurrent.futures import Future

from katcp import Sensor, Message
from katcp.testutils import (DeviceTestServer, start_thread_with_cleanup,
                             DeviceTestSensor)
from katcp import inspecting_client
from katcp.inspecting_client import InspectingClientAsync

logger = logging.getLogger(__name__)

class TestICAClass(tornado.testing.AsyncTestCase):

    def setUp(self):
        super(TestICAClass, self).setUp()
        self.client = InspectingClientAsync('', 0, initial_inspection=False,
                                            ioloop=self.io_loop)

        self.client.set_state_callback(self._cb_state)

    def test_initial_state(self):
        self.assertEqual(self.client.state,
                         inspecting_client.InspectingClientStateType(
            connected=False, synced=False, model_changed=False, data_synced=False))

    def _cb_state(self, state, model_changes):
        """A callback used in the test."""
        self.stop((state, model_changes))

    def test_util_method_difference_add(self):
        """Test the _difference utility method on add."""

        name = None
        original_keys = ['B', 'C']
        updated_keys = ['B', 'C', 'D']
        self.client._sensors_index = {}
        for sen in original_keys:
            data = {'description': "This is {0}.".format(sen)}
            self.client._update_index(self.client._sensors_index, sen, data)
        added, removed = self.client._difference(
            original_keys, updated_keys, name, self.client._sensors_index)
        self.assertIn('D', added)
        self.assertIn('B', self.client.sensors)
        self.assertIn('C', self.client.sensors)

    def test_util_method_difference_rem(self):
        """Test the _difference utility method on remove."""

        name = None
        original_keys = ['A', 'B', 'C']
        updated_keys = ['B', 'C']
        self.client._sensors_index = {}
        for sen in original_keys:
            data = {'description': "This is {0}.".format(sen)}
            self.client._update_index(self.client._sensors_index, sen, data)

        added, removed = self.client._difference(
            original_keys, updated_keys, name, self.client._sensors_index)
        self.assertIn('A', removed)
        self.assertNotIn('A', self.client.sensors)
        self.assertIn('B', self.client.sensors)
        self.assertIn('C', self.client.sensors)

    def test_util_method_difference_rem_named(self):
        """Test the _difference utility method on remove with name set."""

        name = 'A'
        original_keys = ['B', 'C']
        updated_keys = []
        self.client._sensors_index = {}
        for sen in original_keys:
            data = {'description': "This is {0}.".format(sen)}
            self.client._update_index(self.client._sensors_index, sen, data)
        added, removed = self.client._difference(
            original_keys, updated_keys, name, self.client._sensors_index)
        self.assertNotIn('A', removed)
        self.assertNotIn('A', self.client.sensors)
        self.assertIn('B', self.client.sensors)
        self.assertIn('C', self.client.sensors)

    def test_util_method_difference_changed(self):
        """Test the _difference utility method on changed."""

        name = None
        original_keys = ['B']
        updated_keys = ['B']

        self.client._sensors_index = {}
        for sen in original_keys:
            data = {'description': "This is {0}.".format(sen), '_changed': True}
            self.client._update_index(self.client._sensors_index, sen, data)

        added, removed = self.client._difference(original_keys, updated_keys,
                                                 name, self.client._sensors_index)
        self.assertIn('B', self.client.sensors)

    def test_update_index(self):
        """Test the update_index method."""

        index = self.client._sensors_index = {}

        data = {'description': "This is {0}.".format('A')}
        self.client._update_index(index, 'A', data)

        data = {'description': "This is {0}.".format('B')}
        self.client._update_index(index, 'B', data)

        data = {'description': "This is {0}.".format('A')}
        self.client._update_index(index, 'A', data)

        data = {'description': "This is new {0}.".format('B')}
        self.client._update_index(index, 'B', data)

        self.assertIn('new', index['B'].get('description'))
        self.assertFalse(index['A'].get('_changed', False))
        self.assertTrue(index['B'].get('_changed'))


class TestInspectingClientAsync(tornado.testing.AsyncTestCase):

    def setUp(self):
        super(TestInspectingClientAsync, self).setUp()
        self.server = DeviceTestServer('', 0)
        start_thread_with_cleanup(self, self.server, start_timeout=1)
        self.host, self.port = self.server.bind_address

        self.client = InspectingClientAsync(self.host, self.port,
                                            ioloop=self.io_loop)
        self.io_loop.add_callback(self.client.connect)

    @tornado.testing.gen_test
    def test_timeout_of_until_synced(self):
        # Test for timing out
        with self.assertRaises(tornado.gen.TimeoutError):
            yield self.client.until_synced(timeout=0.001)
        # Test for NOT timing out
        yield self.client.until_synced(timeout=0.5)

    @tornado.testing.gen_test
    def test_simple_request(self):
        """Perform a basic request."""
        yield self.client.until_synced()
        reply, informs = yield self.client.simple_request('help', 'watchdog')
        self.assertIn('ok', str(reply))
        self.assertEquals(len(informs), 1)

    @tornado.testing.gen_test
    def test_sensor(self):
        """Access the sensor with the Async client."""
        yield self.client.until_synced()
        sensor_name = 'an.int'
        sensor = yield self.client.future_get_sensor(sensor_name)
        self.assertEquals(sensor.name, sensor_name)
        self.assertEquals(sensor.stype, 'integer')

        # Unknown sensor requests return a None.
        sensor_name = 'thing.unknown_sensor'
        sensor = yield self.client.future_get_sensor(sensor_name)
        self.assertIsNone(sensor)

    @tornado.testing.gen_test
    def test_request_access(self):
        """Test access to requests."""
        yield self.client.until_synced()

        request_name = 'watchdog'
        self.assertIn(request_name, self.client.requests)
        request = yield self.client.future_get_request(request_name)
        self.assertEqual(request.name, request_name)
        self.assertTrue(request.description,
                        'Expected an description: got nothing.')

        # Unknown request return a None.
        request_name = 'watchcat'
        self.assertNotIn(request_name, self.client.requests)
        request = yield self.client.future_get_request(request_name)
        self.assertIsNone(request)

    @tornado.testing.gen_test
    def test_sensor_add_remove(self):
        """Test a sensor being added and then remove it."""
        yield self.client.until_synced()

        sensor = DeviceTestSensor(Sensor.INTEGER, "another.int",
                                  "An Integer.",
                                  "count", [-5, 5], timestamp=time.time(),
                                  status=Sensor.NOMINAL, value=3)
        # Check that the sensor does not exist currently
        self.assertNotIn(sensor.name, self.client.sensors)

        # Add a sensor.
        self.server.add_sensor(sensor)
        self.server.mass_inform(Message.inform('interface-changed'))
        # Do a blocking request to ensure #interface-changed has been received
        yield self.client.simple_request('watchdog')
        yield self.client.until_synced()
        self.assertIn('another.int', self.client.sensors)

        # Remove a sensor.
        self.server.remove_sensor(sensor)
        self.server.mass_inform(Message.inform('interface-changed'))
        # Do a blocking request to ensure #interface-changed has been received
        yield self.client.simple_request('watchdog')

        yield self.client.until_synced()
        self.assertNotIn('another.int', self.client.sensors)


    @tornado.testing.gen_test
    def test_request_add_remove(self):
        """Test a request being added and then remove it."""
        yield self.client.until_synced()

        def request_sparkling_new(self, req, msg):
            """A new command."""
            return Message.reply(msg.name, "ok", "bling1", "bling2")

        # Check that the request did not exist before
        self.assertNotIn('sparkling-new', self.client.requests)

        # Add a request.
        self.server.request_sparkling_new = request_sparkling_new
        self.server._request_handlers['sparkling-new'] = request_sparkling_new
        self.server.mass_inform(Message.inform('interface-changed'))
        # Do a blocking request to ensure #interface-changed has been received
        yield self.client.simple_request('watchdog')

        yield self.client.until_synced()
        self.assertIn('sparkling-new', self.client.requests)
        req = yield self.client.future_get_request('sparkling-new')
        self.assertEqual(req.name, 'sparkling-new')

        # Remove a request.
        self.server.request_sparkling_new = None
        del(self.server._request_handlers['sparkling-new'])
        self.server.mass_inform(Message.inform('interface-changed'))
        # Do a blocking request to ensure #interface-changed has been received
        self.client.simple_request('watchdog')
        yield self.client.until_synced()
        self.assertNotIn('sparkling_new', self.client.requests)

    @tornado.testing.gen_test
    def test_send_request(self):
        """Very high level test.

        Calling methods to insure they do not raise exception.

        """
        client = InspectingClientAsync(self.host, self.port,
                                       ioloop=self.io_loop,
                                       initial_inspection=False)

        yield client.connect()
        yield client.until_connected()
        yield client.until_synced()
        self.assertEquals(len(client.sensors), 0)

        self.assertEquals(len(client.requests), 0)
        self.assertTrue(client.synced)
        self.assertTrue(client.is_connected())
        self.assertTrue(client.connected)
        yield client.simple_request('sensor-sampling', 'an.int', 'event')
        # Wait for sync and check if the sensor was automaticaly added.
        # Get the sensor object and see if it has data.
        sensor = yield client.future_get_sensor('an.int')
        self.assertEquals(len(client.sensors), 1)
        self.assertTrue(sensor.read())
        self.assertEquals(len(client.requests), 0)

    @tornado.testing.gen_test
    def test_handle_sensor_value(self):
        yield self.client.until_connected()
        # Test that #sensor-value informs are handles like #sensor-inform informs if
        # handle_sensor_value() is called.
        sens = yield self.client.future_get_sensor('an.int')
        test_val = 1911
        # Check that the sensor doesn't start out with our test value
        self.assertNotEqual(sens.read().value, test_val)
        # Now set the value of the sensor on the server
        server_sens = self.server.get_sensor('an.int')
        server_sens.set_value(test_val)
        # Do a request to ensure we are synced with the server
        yield self.client.simple_request('watchdog')
        # Test that the value has not yet been set on the client sensor (there are no
        # strategies set)
        self.assertNotEqual(sens.read().value, test_val)
        # Now do a ?sensor-value request, and check that our object is NOT YET updated,
        # since we have not called handle_sensor_value() yet
        yield self.client.simple_request('sensor-value', 'an.int')
        self.assertNotEqual(sens.read().value, test_val)

        # Test call
        self.client.handle_sensor_value()

        # Now do a ?sensor-value request, and check that our object is updated
        yield self.client.simple_request('sensor-value', 'an.int')
        self.assertEqual(sens.read().value, test_val)

    @tornado.testing.gen_test
    def test_factories(self):
        yield self.client.until_connected()
        # Test that the correct factories are used to construct sensor and request
        # objects, and that the factories are called with the correct parameters.
        sf = self.client.sensor_factory = mock.Mock()
        rf = self.client.request_factory = mock.Mock()

        sen = yield self.client.future_get_sensor('an.int')
        req = yield self.client.future_get_request('watchdog')
        self.assertIs(sen, sf.return_value)
        sf.assert_called_once_with(
            units='count', sensor_type=0, params=[-5, 5],
            description='An Integer.', name='an.int')
        self.assertIs(req, rf.return_value)
        rf.assert_called_once_with('watchdog', mock.ANY)

class TestInspectingClientAsyncStateCallback(tornado.testing.AsyncTestCase):

    def setUp(self):
        super(TestInspectingClientAsyncStateCallback, self).setUp()
        self.server = DeviceTestServer('', 0)
        start_thread_with_cleanup(self, self.server, start_timeout=1)
        self.host, self.port = self.server.bind_address
        self.state_cb_future = tornado.concurrent.Future()
        self.client = InspectingClientAsync(self.host, self.port,
                                            ioloop=self.io_loop)
        self.client.set_state_callback(self._test_state_cb)
        self.done_state_cb_futures = []
        self.cnt_state_cb_futures = collections.defaultdict(tornado.concurrent.Future)

    def _test_state_cb(self, state, model_changes):
        f = self.state_cb_future
        self.state_cb_future = tornado.concurrent.Future()
        self.done_state_cb_futures.append(f)
        num_calls = len(self.done_state_cb_futures)
        f.set_result((state, model_changes))
        self.cnt_state_cb_futures[num_calls].set_result(None)

    @tornado.gen.coroutine
    def _check_no_cb(self, no_expected):
        """Let the ioloop run and assert that the callback was not called"""
        yield tornado.gen.moment
        self.assertEqual(len(self.done_state_cb_futures), no_expected)

    @tornado.testing.gen_test(timeout=1)
    def test_from_connect(self):
        # Hold back #version-connect informs
        num_calls_before = len(self.done_state_cb_futures)
        logger.debug('before starting client, num_calls_before:{}'.format(num_calls_before))

        self.server.proceed_on_client_connect.clear()
        self.client.connect()

        state, model_changes = yield self.state_cb_future
        self.assertEqual(state, inspecting_client.InspectingClientStateType(
            connected=True, synced=False, model_changed=False, data_synced=False))
        self.assertIs(model_changes, None)
        # Due to the structure of the state loop the initial state may be sent twice
        # before we get her, and was the case for the initial implementation. Changes made
        # on 2015-01-26 caused it to happy only once, hence + 1. If the implementation
        # changes having + 2 would also be OK.
        yield self._check_no_cb(num_calls_before + 1)

        # Now let the server send #version-connect informs
        num_calls_before = len(self.done_state_cb_futures)
        self.server.ioloop.add_callback(self.server.proceed_on_client_connect.set)
        # We're expecting two calls hard on each other's heels, so lets wait for them
        yield self.cnt_state_cb_futures[num_calls_before + 2]
        # We expected two status callbacks, and no more after
        yield self._check_no_cb(num_calls_before + 2)
        state, model_changes = yield self.done_state_cb_futures[-2]
        state2, model_changes2 = yield self.done_state_cb_futures[-1]
        self.assertEqual(state, inspecting_client.InspectingClientStateType(
            connected=True, synced=False, model_changed=True, data_synced=True))
        server_sensors = self.server._sensors.keys()
        server_requests = self.server._request_handlers.keys()
        self.assertEqual(model_changes, dict(
            sensors=dict(added=set(server_sensors), removed=set()),
            requests=dict(added=set(server_requests), removed=set())))
        self.assertEqual(state2, inspecting_client.InspectingClientStateType(
            connected=True, synced=True, model_changed=False, data_synced=True))
        self.assertEqual(model_changes2, None)

    @tornado.testing.gen_test(timeout=1)
    def test_reconnect(self):
        self.client.connect()
        yield self.client.until_synced()
        yield tornado.gen.moment   # Make sure the ioloop is 'caught up'

        # cause a disconnection and check that the callback is called
        num_calls_before = len(self.done_state_cb_futures)
        self.server.stop()
        self.server.join()
        state, model_changes = yield self.state_cb_future
        self.assertEqual(state, inspecting_client.InspectingClientStateType(
            connected=False, synced=False, model_changed=False, data_synced=False))
        self.assertIs(model_changes, None)
        yield self._check_no_cb(num_calls_before + 1)


class Test_InformHookDeviceClient(tornado.testing.AsyncTestCase):
    def setUp(self):
        super(Test_InformHookDeviceClient, self).setUp()
        self.server = DeviceTestServer('', 0)
        start_thread_with_cleanup(self, self.server, start_timeout=1)
        self.host, self.port = self.server.bind_address
        self.client = katcp.inspecting_client._InformHookDeviceClient(
            self.host, self.port)
        self.client.set_ioloop(self.io_loop)
        self.io_loop.add_callback(self.client.start)

    @tornado.testing.gen_test()
    def test_hook_inform(self):
        h1_calls, h2_calls = [], []
        h1 = lambda msg: h1_calls.append(msg)
        # Specifically test that method callback functions are handled properly. See
        # http://stackoverflow.com/questions/13348031/python-bound-and-unbound-method-object
        class MH(object):
            def h2(self, msg):
                h2_calls.append(msg)
        mh = MH()
        # Add same hook multiple times to check that duplicates are removed
        self.client.hook_inform('help', h1)
        self.client.hook_inform('help', h1)
        self.client.hook_inform('sensor-list', mh.h2)
        self.client.hook_inform('sensor-list', mh.h2)

        yield self.client.until_protocol()
        yield self.client.future_request(katcp.Message.request('help', 'watchdog'))
        yield self.client.future_request(katcp.Message.request('sensor-list', 'an.int'))

        self.assertEqual(len(h1_calls), 1)
        self.assertEqual(h1_calls[0].name, 'help')
        self.assertEqual(len(h2_calls), 1)
        self.assertEqual(h2_calls[0].name, 'sensor-list')
