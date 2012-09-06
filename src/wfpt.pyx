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

#include "pdf.pxi"
include 'integrate.pxi'
include 'cdf.pxi'

def pdf_array(np.ndarray[double, ndim=1] x, double v, double sv, double a, double z, double sz, double t, double st, double err, bint logp=0, int n_st=2, int n_sz=2, bint use_adaptive=1, double simps_err=1e-3):
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef np.ndarray[double, ndim=1] y = np.empty(size, dtype=np.double)

    for i in prange(size, nogil=True):
        y[i] = full_pdf(x[i], v, sv, a, z, sz, t, st, err, n_st, n_sz, use_adaptive, simps_err)

    if logp==1:
        return np.log(y)
    else:
        return y

def wiener_like(np.ndarray[double, ndim=1] x, double v, double sv, double a, double z, double sz, double t,
                double st, double err, int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-8):
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef double p
    cdef double sum_logp = 0

    #if num_threads != 0:
    #    openmp.omp_set_num_threads(num_threads)

    for i in prange(size, nogil=True, schedule='dynamic'):
        p = full_pdf(x[i], v, sv, a, z, sz, t, st, err, n_st, n_sz, use_adaptive, simps_err)
        # If one probability = 0, the log sum will be -Inf
        if p == 0:
            with gil:
                return -np.inf

        sum_logp += log(p)

    return sum_logp

def wiener_like_array(np.ndarray[double, ndim=1] x, double v, double sv, double a, double z, double sz, double t,
                double st, double err, int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-8):
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef double sum_logp = 0

    cdef np.ndarray[double, ndim=1] p_array = np.empty(size, dtype=np.float)

    #if num_threads != 0:
    #    openmp.omp_set_num_threads(num_threads)

    for i in prange(size, nogil=True, schedule='dynamic'):
        p_array[i] = full_pdf(x[i], v, sv, a, z, sz, t, st, err, n_st, n_sz, use_adaptive, simps_err)
        # If one probability = 0, the log sum will be -Inf
        if p_array[i] == 0:
            with gil:
                return -np.inf

    for i in prange(size, nogil=True, schedule='dynamic'):
        sum_logp += log(p_array[i])

    return sum_logp

def wiener_like_multi(np.ndarray[double, ndim=1] x, v, sv, a, z, sz, t, st, double err, multi=None):
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef double p = 0

    if multi is None:
        return full_pdf(x, v, sv, a, z, sz, t, st, err)
    else:
        params = {'v':v, 'z':z, 't':t, 'a':a, 'sv':sv, 'sz':sz, 'st':st}
        params_iter = copy(params)
        for i from 0 <= i < size:
            for param in multi:
                params_iter[param] = params[param][i]

            p += log(full_pdf(x[i], params_iter['v'],
                              params_iter['sv'], params_iter['a'], params_iter['z'],
                              params_iter['sz'], params_iter['t'], params_iter['st'],
                              err))
        return p

def gen_rts_from_cdf(double v, double sv, double a, double z, double sz, double t, \
                     double st, int samples=1000, double cdf_lb=-6, double cdf_ub=6, double dt=1e-2):

    cdef np.ndarray[double, ndim=1] x = np.arange(cdf_lb, cdf_ub, dt)
    cdef np.ndarray[double, ndim=1] l_cdf = np.empty(x.shape[0], dtype=np.double)
    cdef double pdf, rt
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i, j
    cdef int idx

    l_cdf[0] = 0
    for i from 1 <= i < size:
        pdf = full_pdf(x[i], v, sv, a, z, sz, 0, 0, 1e-4)
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

def wiener_like_contaminant(np.ndarray[double, ndim=1] x, np.ndarray[int, ndim=1] cont_x, double v, \
                                 double sv, double a, double z, double sz, double t, double st, double t_min, \
                                 double t_max, double err, int n_st= 10, int n_sz=10, bint use_adaptive=1, \
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
            p = full_pdf(x[i], v, sv, a, z, sz, t, st, err, n_st, n_sz, use_adaptive, simps_err)
            if p == 0:
                with gil:
                    return -np.inf
            sum_logp += log(p)
        # If one probability = 0, the log sum will be -Inf


    # add the log likelihood of the contaminations
    sum_logp += n_cont*log(0.5 * 1./(t_max-t_min))

    return sum_logp

def gen_cdf(double v, double sv, double a, double z, double sz, double t, double st, double precision=3.,
            int N=500, double time=5., np.ndarray[double, ndim=1] cdf_array=None):

    if (sv < 0) or (a <=0 ) or (z < 0) or (z > 1) or (sz < 0) or (sz > 1) or (z+sz/2.>1) or \
    (z-sz/2.<0) or (t-st/2.<0) or (t<0) or (st < 0):
        raise ValueError("at least one of the parameters is out of the support")

    if cdf_array is None:
        cdf_array = np.empty(2*N+1, dtype=np.double)

    cdef np.ndarray[double, ndim=1] x = np.linspace(-time, time, 2*N+1)

    cdef double* cdf_ptr = cdf(v, sv, a, z, sz, t, st, precision, N, time, <double *> cdf_array.data)

    return x, cdf_array

def split_cdf(np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] data):

    #get length of data
    cdef int N = (len(data) -1) / 2

    # lower bound is reversed
    cdef np.ndarray[double, ndim=1] x_lb = -x[:N][::-1]
    cdef np.ndarray[double, ndim=1] lb = data[:N][::-1]
    # lower bound is cumulative in the wrong direction
    lb = np.cumsum(np.concatenate([np.array([0]), -np.diff(lb)]))

    cdef np.ndarray[double, ndim=1] x_ub = x[N+1:]
    cdef np.ndarray[double, ndim=1] ub = data[N+1:]
    # ub does not start at 0
    ub -= ub[0]

    return (x_lb, lb, x_ub, ub)