# Copyright (C) 2018 Michele Guerini Rocco
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# How to use this program
#
# 1. To preview a file run
#
#    $ python3 dtlg.py 180412001 -p
#    samples: 75000
#    sampling rate: 1.504e+05 Hz
#    time: [0.000e+00 6.650e-06 1.330e-05 ... 4.987e-01 4.987e-01 4.987e-01]
#    curve 1: [ 1.221e-03  0.000e+00 -3.662e-03 ... -1.482e+00 -1.504e+00 -1.512e+00]
#    curve 2: [0.011 0.01  0.021 ... 0.002 0.012 0.004]
#    curve 3: [0.016 0.018 0.022 ... 0.013 0.017 0.011]
#
# 2. To convert a file into text
#
#    $ python3 dtlg.py 180412001 -e 180412001.txt -f txt
#
# 3. To convert a file to pickle (numpy format)
#
#    $ python3 dtlg.py 180412001 -e 180412001.npy
#
# This program can also be imported and used to load a DTLG file
# directly into memory as numpy arrays. For example
#
#    import dtlg
#    with open('180412001', 'rb') as data:
#       time, col1, col2, col3 = dtlg.parse(data)
#
# Run `python3 dtlg.py -h` for more information.

import argparse
import logging
import struct
import numpy as np


def parse_file_header(dtlg):
    """
    Parse the DTLG file header.
    Returns a dictionary of the header.
    """
    log = logging.getLogger('dtlg')

    magic = struct.unpack('>4s', dtlg.read(4))  # DTLG
    magic_string = magic[0].decode('ascii')

    if magic_string != 'DTLG':
        log.warning('magic string is not "DTLG". wrong file?')

    version = struct.unpack('>2B', dtlg.read(2))  # labVIEW version
    version_string = '.'.join(str(i) for i in version)

    dtlg.read(2)  # ??? 80 05

    n_events = struct.unpack('>I', dtlg.read(4))[0]  # number of events

    dtlg.read(4)  # ??? 00 00 00 3a

    dtlg.read(557)  # skip data description

    n_curves = struct.unpack('>b', dtlg.read(1))[0]  # number of data sets

    log.info('magic string: ' + magic_string)
    log.info('labVIEW version: ' + version_string)
    log.info('events: {}'.format(n_events))
    log.info('curves: {}'.format(n_curves))

    dtlg.read(16)  # ???

    return {'magic':   magic_string,
            'version': version_string,
            'events':  n_events,
            'curves':  n_curves}


def parse_curve_header(curve, i=0):
    """
    Parse a curve header.
    Returns sampling interval and number of points.
    """
    Δt = struct.unpack('>d', curve.read(8))[0]  # sampling interval
    n  = struct.unpack('>I', curve.read(4))[0]  # number of points

    log = logging.getLogger('dtlg')
    log.debug('parsing curve {}...'.format(i))
    log.debug('  sampling rate: {:.3e} Hz'.format(1/Δt))
    log.debug('  samples: {}'.format(n))
    return Δt, n


def parse_points(curve, n):
    """
    Parse curve points as an n-sized 1D array.
    """
    points = np.empty(n, dtype=float)
    for i in range(n):
        points[i] = struct.unpack('>d', curve.read(8))[0]
    curve.read(159)  # ???

    log = logging.getLogger('dtlg')
    log.debug('done.')
    return points


def parse(dtlg):
    """
    Parse a DTLG file.
    Takes a file-like object and returns an array
    of arrays with the following columns:
      | time (s) | curve 1 | curve 2 | ... | curve n |
    """
    header = parse_file_header(dtlg)
    curves = []
    for i in range(header['curves']):
        Δt, n = parse_curve_header(dtlg, i)
        curve = parse_points(dtlg, n)
        curves.append(curve)
    time = np.arange(0, n*Δt, Δt)
    return np.stack([time, *curves])


def cli_args():
    parser = argparse.ArgumentParser(
        description='Convert DTLG files into numpy or text format.')
    parser.add_argument('input', metavar='DTLG',
                        type=argparse.FileType('rb'),
                        help='input file in DTLG format')
    parser.add_argument('-f', '--format', metavar='EXT', type=str,
                        default='npy', choices=['npy', 'txt'],
                        help='output file format')
    parser.add_argument('-e', '--export',
                        type=argparse.FileType('wb'),
                        help='output file path')
    parser.add_argument('-p', '--preview',
                        action='store_true', default=False,
                        help='preview file content')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='increase logging verbosity')
    args = parser.parse_args()

    return args


def main(opts):
    # setup logger
    logging.basicConfig(
        level=logging.WARNING,
        format='%(levelname)s: %(message)s')
    log = logging.getLogger('dtlg')
    log.setLevel(max(3 - opts.verbose, 0) * 10)

    data = parse(opts.input)

    # preview data
    if opts.preview:
        time = data[0]
        rate = 1/(time[1] - time[0])
        print('samples:', len(time))
        print('sampling rate: {:.3e} Hz\n'.format(rate))

        np.set_printoptions(precision=3)
        print('time:', time)
        for i, curve in enumerate(data[1:]):
            print('curve {}:'.format(i + 1), curve)

    # export data
    if opts.export:
        if opts.format == "npy":
            np.save(opts.export, data)
        elif opts.format == "txt":
            np.savetxt(opts.export, data.T, fmt='%.12g')


if __name__ == "__main__":
    main(cli_args())
