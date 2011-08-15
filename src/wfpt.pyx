#!/usr/bin/python 
#
# Cython version of the Navarro & Fuss, 2009 DDM PDF. Based directly
# on the following code by Navarro & Fuss:
# http://www.psychocmath.logy.adelaide.edu.au/personalpages/staff/danielnavarro/resources/wfpt.m
#
# This implementation is about 170 times faT than the matlab
# reference version.
#
# Copyleft Thomas Wiecki (thomas_wiecki[at]brown.edu), 2010 
# GPLv3

from copy import copy
import numpy as np
cimport numpy as np

cimport cython

from cython.parallel import *

include "pdf.pxi"

@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def pdf_array(np.ndarray[double, ndim=1] x, double v, double a, double z, double t, double err, bint logp=0):
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef np.ndarray[double, ndim=1] y = np.empty(size, dtype=np.double)

    for i in prange(size, nogil=True):
        y[i] = pdf_sign(x[i], v, a, z, t, err)

    if logp==1:
        return np.log(y)
    else:
        return y

@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def wiener_like_simple(np.ndarray[double, ndim=1] x, double v, double a, double z, double t, double err):
    cdef Py_ssize_t i
    cdef double p
    cdef double sum_logp = 0
    for i in prange(x.shape[0], nogil=True):
        p = pdf_sign(x[i], v, a, z, t, err)
        # If one probability = 0, the log sum will be -Inf
        if p == 0:
            with gil:
                return -np.inf
        sum_logp += log(p)

    return sum_logp

@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def wiener_like_simple_multi(np.ndarray[double, ndim=1] x, v, a, z, t, double err, multi=None):
    cdef unsigned int size = x.shape[0]
    cdef unsigned int i
    cdef double p = 0

    if multi is None:
        return wiener_like_simple(x, v, a, z, t, err)
    else:
        params = {'v':v, 'z':z, 't':t, 'a':a}
        params_iter = copy(params)
        for i from 0 <= i < size:
            for param in multi:
                params_iter[param] = params[param][i]
                
            p += log(pdf_sign(x[i], params_iter['v'], params_iter['a'], params_iter['z'], params_iter['t'], err))
                
        return p
