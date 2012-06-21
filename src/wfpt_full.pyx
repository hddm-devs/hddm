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
def wiener_like_full_intrp(np.ndarray[double, ndim=1] x, double v, double V, double a, double z, double sz, double t,
                           double st, double err, int n_st= 10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-8):
    cdef Py_ssize_t i
    cdef double p
    cdef double sum_logp = 0

    for i in prange(x.shape[0], nogil=True):
        p = full_pdf(x[i], v, V, a, z, sz, t, st, err, n_st, n_sz, use_adaptive, simps_err)
        # If one probability = 0, the log sum will be -Inf
        if p == 0:
            with gil:
                return -np.inf
        sum_logp += log(p)

    return sum_logp


@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def wiener_like_full(np.ndarray[double, ndim=1] x,
                     np.ndarray[double, ndim=1] v, np.ndarray[double, ndim=1] a,
                     np.ndarray[double, ndim=1] z, np.ndarray[double, ndim=1] t, err):
    cdef Py_ssize_t i
    cdef double p
    cdef double sum_logp = 0

    for i in prange(x.shape[0], nogil=True):
        p = pdf_sign(x[i], v[i], a[i], z[i], t[i], err)
        # If one probability = 0, the log sum will be -Inf
        if p == 0:
            with gil:
                return -np.inf
        sum_logp += log(p)

    return sum_logp


@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def wiener_like_full_multi(np.ndarray[double, ndim=1] x, v, V, a, z, sz, t, st, double err, multi=None):
    cdef unsigned int size = x.shape[0]
    cdef unsigned int i
    cdef double p = 0

    if multi is None:
        return wiener_like_full_intrp(x, v, V, a, z, sz, t, st, err)
    else:
        params = {'v':v, 'z':z, 't':t, 'a':a, 'sv':V, 'sz':sz, 'st':st}
        params_iter = copy(params)
        for i from 0 <= i < size:
            for param in multi:
                params_iter[param] = params[param][i]

            p += log(full_pdf(x[i], params_iter['v'],
                              params_iter['sv'], params_iter['a'], params_iter['z'],
                              params_iter['sz'], params_iter['t'], params_iter['st'],
                              err))
        return p

@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def gen_rts_from_cdf(double v, double V, double a, double z, double sz, double t, \
                           double st, int samples=1000, double cdf_lb = -6, double cdf_ub = 6, double dt=1e-2):

    cdef np.ndarray[double, ndim=1] x = np.arange(cdf_lb, cdf_ub, dt)
    cdef np.ndarray[double, ndim=1] l_cdf = np.empty(x.shape[0], dtype=np.double)
    cdef double pdf, rt
    cdef Py_ssize_t i, j
    cdef int idx

    l_cdf[0] = 0
    for i in range(x.shape[0]):
        pdf = full_pdf(x[i], v, V, a, z, sz, 0, 0, 1e-4)
        l_cdf[i] = l_cdf[i-1] + pdf

    l_cdf /= l_cdf[x.shape[0]-1]

    cdef np.ndarray[double, ndim=1] rts = np.empty(samples, dtype=np.double)
    cdef np.ndarray[double, ndim=1] f = np.random.rand(samples)
    cdef np.ndarray[double, ndim=1] delay

    if st!=0:
        delay = (np.random.rand(samples)*st + (t - st/2.))
    for i from 0 <= i < samples:
        idx = np.searchsorted(l_cdf, f[i])
        rt = x[idx]
        if st==0:
            rt = rt + np.sign(rt)*t
        else:
            rt = rt + np.sign(rt)*delay[i]
        rts[i] = rt
    return rts

@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def wiener_like_full_contaminant(np.ndarray[double, ndim=1] value, np.ndarray[int, ndim=1] cont_x, double gamma, double v, double V, double a, double z, double sz, double t, double st, double t_min, double t_max, double err):
    """Wiener likelihood function where RTs could come from a
    separate, uniform contaminant distribution.

    Reference: Lee, Vandekerckhove, Navarro, & Tuernlinckx (2007)
    """
    cdef Py_ssize_t i
    cdef double p
    cdef double sum_logp = 0
    cdef int n_cont = np.sum(cont_x)
    cdef int pos_cont = 0

    for i in prange(value.shape[0], nogil=True):
        if cont_x[i] == 0:
            p = full_pdf(value[i], v, V, a, z, sz, t, st, err)
            sum_logp += log(p)
        elif value[i]>0:
            pos_cont += 1

    # add the log likelihood of the contaminations
    #first the guesses
    sum_logp += n_cont*log(gamma*(0.5 * 1./(t_max-t_min)))
    #then the positive prob_boundary
    sum_logp += pos_cont*log((1-gamma) * prob_ub(v, a, z) * 1./(t_max-t_min))
    #and the negative prob_boundary
    sum_logp += (n_cont - pos_cont)*log((1-gamma)*(1-prob_ub(v, a, z)) * 1./(t_max-t_min))

    return sum_logp

def wiener_like_full_single(double x, double v, double V, double a,
                            double z, double sz, double t, double st,
                            double err, int n_st=2, int n_sz=2, bint
                            use_adaptive=1, double simps_err=1e-3):

    # Wrapper for c-only full_pdf function, only required for unittest
    return full_pdf(x, v, V, a, z, sz, t, st, err, n_st, n_sz, use_adaptive, simps_err)
