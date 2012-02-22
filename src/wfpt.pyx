#cython: embedsignature=True
#cython: cdivision=True
#cython: wraparound=False
#cython: boundscheck=False
#
# Cython version of the Navarro & Fuss, 2009 DDM PDF. Based on the following code by Navarro & Fuss:
# http://www.psychocmath.logy.adelaide.edu.au/personalpages/staff/danielnavarro/resources/wfpt.m
#
# This implementation is about 170 times faster than the matlab
# reference version.
#
# Copyleft Thomas Wiecki (thomas_wiecki[at]brown.edu) & Imri Sofer, 2011
# GPLv3

import hddm

from copy import copy
import numpy as np
cimport numpy as np

cimport cython

from cython.parallel import *
#cimport openmp

include "pdf.pxi"

def pdf_array(np.ndarray[double, ndim=1] x, double v, double V, double a, double z, double Z, double t, double T, double err, bint logp=0, int nT=2, int nZ=2, bint use_adaptive=1, double simps_err=1e-3):
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef np.ndarray[double, ndim=1] y = np.empty(size, dtype=np.double)

    for i in prange(size, nogil=True):
        y[i] = full_pdf(x[i], v, V, a, z, Z, t, T, err, nT, nZ, use_adaptive, simps_err)

    if logp==1:
        return np.log(y)
    else:
        return y

def wiener_like(np.ndarray[double, ndim=1] x, double v, double V, double a, double z, double Z, double t,
                double T, double err, int nT=10, int nZ=10, bint use_adaptive=1, double simps_err=1e-8):
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef double p
    cdef double sum_logp = 0

    #if num_threads != 0:
    #    openmp.omp_set_num_threads(num_threads)

    for i in prange(size, nogil=True, schedule='dynamic'):
        p = full_pdf(x[i], v, V, a, z, Z, t, T, err, nT, nZ, use_adaptive, simps_err)
        # If one probability = 0, the log sum will be -Inf
        if p == 0:
            with gil:
                return -np.inf

        sum_logp += log(p)

    return sum_logp

def wiener_like_array(np.ndarray[double, ndim=1] x, double v, double V, double a, double z, double Z, double t,
                double T, double err, int nT=10, int nZ=10, bint use_adaptive=1, double simps_err=1e-8):
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef double sum_logp = 0

    cdef np.ndarray[double, ndim=1] p_array = np.empty(size, dtype=np.float)

    #if num_threads != 0:
    #    openmp.omp_set_num_threads(num_threads)

    for i in prange(size, nogil=True, schedule='dynamic'):
        p_array[i] = full_pdf(x[i], v, V, a, z, Z, t, T, err, nT, nZ, use_adaptive, simps_err)
        # If one probability = 0, the log sum will be -Inf
        if p_array[i] == 0:
            with gil:
                return -np.inf

    for i in prange(size, nogil=True, schedule='dynamic'):
        sum_logp += log(p_array[i])

    return sum_logp

def wiener_like_multi(np.ndarray[double, ndim=1] x, v, V, a, z, Z, t, T, double err, multi=None):
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef double p = 0

    if multi is None:
        return full_pdf(x, v, V, a, z, Z, t, T, err)
    else:
        params = {'v':v, 'z':z, 't':t, 'a':a, 'V':V, 'Z':Z, 'T':T}
        params_iter = copy(params)
        for i from 0 <= i < size:
            for param in multi:
                params_iter[param] = params[param][i]

            p += log(full_pdf(x[i], params_iter['v'],
                              params_iter['V'], params_iter['a'], params_iter['z'],
                              params_iter['Z'], params_iter['t'], params_iter['T'],
                              err))
        return p

def gen_rts_from_cdf(double v, double V, double a, double z, double Z, double t, \
                     double T, int samples=1000, double cdf_lb=-6, double cdf_ub=6, double dt=1e-2):

    cdef np.ndarray[double, ndim=1] x = np.arange(cdf_lb, cdf_ub, dt)
    cdef np.ndarray[double, ndim=1] l_cdf = np.empty(x.shape[0], dtype=np.double)
    cdef double pdf, rt
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i, j
    cdef int idx

    l_cdf[0] = 0
    for i in range(size):
        pdf = full_pdf(x[i], v, V, a, z, Z, 0, 0, 1e-4)
        l_cdf[i] = l_cdf[i-1] + pdf

    l_cdf /= l_cdf[x.shape[0]-1]

    cdef np.ndarray[double, ndim=1] rts = np.empty(samples, dtype=np.double)
    cdef np.ndarray[double, ndim=1] f = np.random.rand(samples)
    cdef np.ndarray[double, ndim=1] delay

    if T!=0:
        delay = (np.random.rand(samples)*T + (t - T/2.))
    for i from 0 <= i < samples:
        idx = np.searchsorted(l_cdf, f[i])
        rt = x[idx]
        if T==0:
            rt = rt + np.sign(rt)*t
        else:
            rt = rt + np.sign(rt)*delay[i]
        rts[i] = rt
    return rts

def wiener_like_contaminant(np.ndarray[double, ndim=1] x, np.ndarray[int, ndim=1] cont_x, double v, \
                                 double V, double a, double z, double Z, double t, double T, double t_min, \
                                 double t_max, double err, int nT= 10, int nZ=10, bint use_adaptive=1, \
                                 double simps_err=1e-8):
    """Wiener likelihood function where RTs could come from a
    separate, uniform contaminant distribution.

    Reference: Lee, Vandekerckhove, Navarro, & Tuernlinckx (2007)
    """
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef double p
    cdef double sum_logp = 0
    cdef int n_cont = np.sum(cont_x)
    cdef int pos_cont = 0

    for i in prange(size, nogil=True):
        if cont_x[i] == 0:
            p = full_pdf(x[i], v, V, a, z, Z, t, T, err, nT, nZ, use_adaptive, simps_err)
            if p == 0:
                with gil:
                    return -np.inf
            sum_logp += log(p)
        # If one probability = 0, the log sum will be -Inf


    # add the log likelihood of the contaminations
    sum_logp += n_cont*log(0.5 * 1./(t_max-t_min))

    return sum_logp

