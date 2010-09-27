#!/bin/sh
cython lba.pyx
gcc -c lba.c -I/usr/include/python2.6 -I/usr/lib/python2.6/dist-packages/numpy/core/include
gcc -shared -o lba.so lba.o -lblas -lm -lpython2.6 -O3 -fPIC -fno-strict-aliasing -fwrapv -pthread
