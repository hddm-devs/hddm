#cython: embedsignature=True
#cython: cdivision=True
#cython: wraparound=False
#cython: boundscheck=False

#!/usr/bin/python
# Cython wrapper for the fast-dm code by Voss & Voss
# (C) by Thomas Wiecki (thomas_wiecki@brown.edu), 2010
# GPLv2 or later.

from libc.stdlib cimport malloc, free

ctypedef np.int_t DTYPE_int

# Define external functions of fast-dm
cdef extern from "fast-dm.h":
    cdef struct F_calculator:
        pass
    void F_start (F_calculator*, int)
    int  F_get_N (F_calculator*)
    double F_get_z (F_calculator*, int)
    double  F_get_val (F_calculator*, double, double)
    void  F_delete (F_calculator*)
    cdef F_calculator* F_new(double*)
    void  set_precision (double)

@cython.boundscheck(False) # turn off bounds-checking for entire function
cdef double* cdf(double v, double sv, double a, double z, double sz, double t, double st, double precision, int N, double time, double *output):
    """Compute the CDF of the drift diffusion model using the method
    and implementation of fast-dm by Voss&Voss.
    """
    cdef F_calculator *fc
    cdef double dt = time/N
    cdef double *params = <double *> malloc(sizeof(double) * 6)

    set_precision(precision)

    params[0] = a; params[1]=v; params[2]=t; params[3]=a*sz;
    params[4] = sv; params[5]=st;


    fc = F_new(params)

    F_start (fc, 1)
    for i from 0 <= i <= N:
        output[N+i] = F_get_val(fc, i*dt, a*z)

    F_start (fc, 0)
    for i from 1 <= i <= N:
        output[N-i] = F_get_val(fc, i*dt, a*z)


    #make sure that order of the output is not affected by random errors
    for i from 1 <= i <= (2*N):
        if output[i] < output[i-1]:
            output[i] = output[i-1]

    #free memory
    F_delete (fc)
    free(params)

    return output
