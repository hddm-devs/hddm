#!/bin/sh
cython wfpt.pyx
gcc -c wfpt.c -I/usr/include/python2.6 -I/usr/lib/python2.6/dist-packages/numpy/core/include
gcc -shared -o wfpt.so wfpt.o -lblas -lm -lpython2.6 -O3 -fPIC -fno-strict-aliasing -fwrapv -pthread
