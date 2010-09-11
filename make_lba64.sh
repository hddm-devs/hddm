#!/bin/sh
cp lba.pyx lba64.pyx
cython lba64.pyx
gcc -c lba64.c -fPIC -I/usr/include/python2.6 -I/usr/lib/python2.6/dist-packages/numpy/core/include
gcc -shared -o lba64.so lba64.o -lblas -lm -lpython2.6 -O3 -fPIC -fno-strict-aliasing -fwrapv -pthread
