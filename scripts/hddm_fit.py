#!/usr/bin/python

if __name__=='__main__':
    import hddm
    import sys
    import os
    if len(sys.argv) == 1 or sys.argv[1] == '-h' or sys.argv[1] == '--help':
        print "hddmfit model.cfg [data.csv]"
        sys.exit(-1)

    config_fname = sys.argv[1]
    data_fname = None

    if len(sys.argv) == 3:
        data_fname = sys.argv[2]
        
    hddm.utils.parse_config_file(config_fname, data=data_fname)
