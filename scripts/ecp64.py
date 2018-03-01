#!/usr/bin/env python

from __future__ import print_function

import logging
from collections import OrderedDict
import sys

import numpy as np
import spead2
from spead2 import recv


def channel_ordering(num_chans):
    """Ordering of spectrometer channels in an ECP-64 SPEAD item.

    Parameters
    ----------
    num_chans : int
        Number of spectrometer channels

    Returns
    -------
    spead_index_per_channel : array of int, shape (`num_chans`,)
        Index into SPEAD item of each spectrometer channel, allowing i'th
        channel to be accessed as `spead_data[spead_index_per_channel[i]]`
    """
    pairs = np.arange(num_chans).reshape(-1, 2)
    first_half = pairs[:num_chans // 4]
    second_half = pairs[num_chans // 4:]
    return np.c_[first_half, second_half].ravel()


def unpack_bits(x, partition):
    """Extract a series of bit fields from an integer.

    Parameters
    ----------
    x : uint
        Unsigned integer to be interpreted as a series of bit fields
    partition : sequence of int
        Bit fields to extract from `x` as indicated by their size in bits,
        with the last field ending at the LSB of `x` (as per ECP-64 document)

    Returns
    -------
    fields : list of uint
        The value of each bit field as an unsigned integer
    """
    out = []
    for size in reversed(partition):  # Grab fields starting from LSB
        out.append(x & ((1 << size) - 1))
        x >>= size
    return out[::-1]    # Put back into MSB-to-LSB order


if __name__ == '__main__':

    ip = sys.argv[1]
    ports = (7150, 7151, 7152, 7153)
    num_chans = 128
    sampling_rate = 1712e6
    bandwidth = 0.5 * sampling_rate
    centre_freq = 0.75 * sampling_rate

    chans = channel_ordering(num_chans)
    freqs = centre_freq + (bandwidth / num_chans) * (np.arange(num_chans) - num_chans / 2)

    logging.basicConfig(level='DEBUG')
    rx = recv.Stream(spead2.ThreadPool())
    for port in ports:
        rx.add_udp_reader(port, bind_hostname=ip)
    ig = spead2.ItemGroup()

    raw_data = OrderedDict()
    for heap in rx:
        new_items = ig.update(heap)
        if 'timestamp' not in new_items:
            continue
        timestamp = ig['timestamp'].value / sampling_rate
        dig_id = ig['digitiser_id'].value
        dig_status = ig['digitiser_status'].value
        dig_serial, dig_type, receptor, pol = unpack_bits(dig_id, (24, 8, 14, 2))
        saturation, nd_on = unpack_bits(dig_status, (8, 1))
        stream = [s[5:] for s in new_items if s.startswith('data_')][0]
        print(receptor, timestamp, nd_on, stream)
        key = (receptor[0], timestamp[0])
        fields = raw_data.get(key, {})
        fields['saturation'] = saturation
        fields['nd_on'] = nd_on
        if stream == 'vh':
            fields['revh'] = ig['data_' + stream].value[:num_chans][chans]
            fields['imvh'] = ig['data_' + stream].value[num_chans:][chans]
        else:
            fields[stream] = ig['data_' + stream].value[chans]
        raw_data[key] = fields

    timestamps = []
    saturation = []
    nd_on = []
    data_hh_0 = []
    data_vv_0 = []
    data_vh_0 = []
    data_hh_1 = []
    data_vv_1 = []
    data_vh_1 = []

    for key, value in raw_data.items():
        receptor = 'm%03d' % (key[0],)
        if len(value) < 6:
            continue
        timestamps.append(key[1])
        nd_on.append(value['nd_on'])
        saturation.append(value['saturation'])
        if value['nd_on'] == 0 :
            data_hh_0.append(value['hh'])
            data_vv_0.append(value['vv'])
            data_vh_0.append(value['revh'] + 1.0j * value['imvh'])
        else:
            data_hh_1.append(value['hh'])
            data_vv_1.append(value['vv'])
            data_vh_1.append(value['revh'] + 1.0j * value['imvh'])

    np.savez('spectrometer_data.npz', ts=timestamps, sat=saturation, nd=nd_on,
             hh_0=data_hh_0, vv_0=data_vv_0, vh_0=data_vh_0, hh_1=data_hh_1, vv_1=data_vv_1, vh_1=data_vh_1, freqs=freqs,
             ds=dig_serial, dt=dig_type, receptor=receptor)
