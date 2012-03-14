#!/usr/bin/env python

import sys
import hddm
import os

try:
    import argparse
except:
    # fall back to version not using argparse
    if __name__=='__main__':
        if len(sys.argv) == 1 or sys.argv[1] == '-h' or sys.argv[1] == '--help':
            print "hddmfit model.cfg [data.csv]"
            sys.exit(-1)

        config_fname = sys.argv[1]
        data_fname = None

        if len(sys.argv) == 3:
            data_fname = sys.argv[2]

        hddm.utils.parse_config_file(config_fname, data=data_fname)
        sys.exit(0)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='HDDM: Hierarchical estimation of drift diffusion models.', add_help=True, version='0.2a')

    parser.add_argument('--verbose', action="store_true", dest="verbose", help='more verbosity')
    parser.add_argument('-m', '--map', action="store_true", dest="map", help='find a starting point via MAP estimation')
    parser.add_argument('-s', '--samples', action="store", dest="samples", type=int, help='samples to generate')
    parser.add_argument('-b', '--burn', action="store", dest="burn", type=int, help='initial samples to discard as burn-in')
    parser.add_argument('-t', '--thin', action="store", dest="thin", type=int, help='keep every nth sample')

    parser.add_argument('-og', action="store_true", dest="only_group_stats", default=False, help='Output only group stats.')
    parser.add_argument('-np', action='store_true', dest='no_plots', default=False, help='Do not generate output plots.')

    parser.add_argument('model')
    parser.add_argument('data')

    results = parser.parse_args()

    hddm.utils.parse_config_file(results.model, data=results.data,
                                 samples=results.samples,
                                 burn=results.burn, thin=results.thin,
                                 map=results.map,
                                 only_group_stats=results.only_group_stats,
                                 plot=not(results.no_plots),
                                 verbose=results.verbose)

    sys.exit(0)
