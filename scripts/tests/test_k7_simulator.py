import unittest
import os.path
import subprocess
import katcp
import spead
import time
import socket
import ConfigParser
import numpy as np

def retry(fn, exception, timeout=1, sleep_interval=0.025):
    trytime = 0
    while trytime < timeout:
        try:
            retval = fn()
        except exception:
            time.sleep(sleep_interval)
            trytime += sleep_interval
            if trytime >= timeout:
                raise
            continue
        break
    return retval


class TestCorrelatorData(unittest.TestCase):
    def setUp(self):
        # simulator host and port
        device_host = '127.0.0.1' ; device_port = 2041
        # Test that there is not already a service listening on that
        # port
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect((device_host, device_port))
            raise Exception('Service already running on port %d. Perhaps run kat-stop?')
        except: pass
        self.k7_simulator_logfile = open('k7_simulator.log', 'w')
        #Find the k7_simulator executable
        k7_simulator_file = os.path.join(
            os.path.dirname(__file__), '..', 'k7_simulator.py')
        k7_conf_file = os.path.join(
            os.path.dirname(__file__), '..', '..',
            'katcapture', 'conf', 'k7-local.conf')
        # Set up a process running the k7_simulator
        self.k7_simulator_proc = subprocess.Popen(
            [k7_simulator_file, '-p', '%d' % device_port, '-c', k7_conf_file],
            stdout=self.k7_simulator_logfile, stderr=self.k7_simulator_logfile)

        # Set up katcp client
        self.k7_simulator_katcp = katcp.BlockingClient(device_host, device_port)
        self.k7_simulator_katcp.start()
        # Wait for katcp client to become alive
        help_reply, help_informs = retry(
            lambda : self.k7_simulator_katcp.blocking_request(
                katcp.Message.request('help')),
            katcp.KatcpClientError)
        # Get spead udp port
        cfg = ConfigParser.SafeConfigParser()
        cfg.read(k7_conf_file)
        spead_udp_port = cfg.getint('receiver', 'rx_udp_port')


        # Set up spead client
        self.k7_simulator_speadrx = spead.TransportUDPrx(
            spead_udp_port, pkt_count=1024, buffer_size=51200000)

    def katcp_req(self, *req):
        return self.k7_simulator_katcp.blocking_request(
            katcp.Message.request(*req))
        
    def tearDown(self):
        # Stop the spead client
        self.k7_simulator_speadrx.stop()
        
        # Stop the katcp client
        self.k7_simulator_katcp.stop()
        self.k7_simulator_katcp.join()

        # Kill the k7_simulator process
        self.k7_simulator_proc.terminate()
        time.sleep(0.1)         # Give it a chance to exit gracefully
        self.k7_simulator_proc.kill()
        self.k7_simulator_logfile.close()
        
    def test_data(self):
        # Set test target at az 165, el 45 deg with flux of 200
        self.katcp_req('test-target', '165.', '45.', '200.')
        # Point 'antenna' away from test target
        self.katcp_req('pointing-az', '280.')
        self.katcp_req('pointing-el', '80.')
        away_group = self.get_spead_itemgroup(time_after=time.time())
        self.verify_shapes(away_group)
        
        # Point 'antenna' at the test target
        self.katcp_req('pointing-az', '165.')
        self.katcp_req('pointing-el', '45.')
        target_group_200 = self.get_spead_itemgroup(time_after=time.time())
        self.verify_shapes(target_group_200)

        # Increase flux of test target to 1000
        self.katcp_req('test-target', '165.', '45.', '1000.')
        target_group_1000 = self.get_spead_itemgroup(time_after=time.time())
        self.verify_shapes(target_group_1000)

        away_max = np.max(np.abs(self.get_xeng(away_group)).flat)
        target_200_max = np.max(np.abs(self.get_xeng(target_group_200)).flat)
        target_1000_max = np.max(np.abs(self.get_xeng(target_group_1000)).flat)
        self.assertTrue(target_200_max > 2*away_max)
        self.assertTrue(target_1000_max > 4*target_200_max)
        
    def get_xeng(self, group):
        xeng_raw = group['xeng_raw']
        xeng = np.zeros(xeng_raw.shape[0:2], dtype=np.complex64)
        xeng[:] = (xeng_raw[:,:,0] / group['n_accs'] +
                  1j*xeng_raw[:,:,1] / group['n_accs'])
        return xeng


    def verify_shapes(self, group):
        n_stokes = 4            # Number of stokes polarisation parameters
        n_chans = 512           # Number of frequency channels
        self.assertEqual(group['n_chans'], n_chans)
        n_ants = group['n_ants']
        self.assertEqual(n_ants, 8) # Number of antennas supported by DBE7
        # Number of baselines (including self-correlations)
        n_bls = (n_ants)*(n_ants+1)/2*n_stokes
        self.assertEqual(group['n_bls'], n_bls)
        self.assertEqual(group['xeng_raw'].shape, (n_chans, n_bls, 2))

        
    def get_spead_itemgroup(self, time_after=None):
        max_spead_iters = 10
        itemgroup = spead.ItemGroup()
        meta_req = ['n_bls', 'n_chans', 'n_accs', 'xeng_raw',
                    'timestamp', 'scale_factor_timestamp', 'sync_time']
        self.katcp_req('spead-issue')
        for i, heap in enumerate(spead.iterheaps(self.k7_simulator_speadrx)):
            itemgroup.update(heap)
            speadkeys = itemgroup.keys()
            if all(
                k in speadkeys and itemgroup[k] is not None for k in meta_req):
                if time_after is None:
                    break
                timestamp = (itemgroup['timestamp'] /
                             itemgroup['scale_factor_timestamp'] +
                             itemgroup['sync_time'])
                if timestamp > time_after + itemgroup['int_time']:
                    break
            if i > max_spead_iters:
                raise Exception(
                    'Unable to obtain required spead data in %d iterations'
                    % max_spead_iters)
        return itemgroup