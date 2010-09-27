#!/bin/sh
cp wfpt.pyx wfpt64.pyx
cython wfpt64.pyx
gcc -c wfpt64.c -fPIC -I/usr/include/python2.6 -I/usr/lib/python2.6/dist-packages/numpy/core/include
gcc -shared -o wfpt64.so wfpt64.o -lblas -lm -lpython2.6 -O3 -fPIC -fno-strict-aliasing -fwrapv -pthread
