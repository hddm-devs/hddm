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

include "pdf.pxi"

@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def wiener_like_full_intrp(np.ndarray[double, ndim=1] x, double v, double V, double a, double z, double Z, double t, 
                           double T, double err, int nT= 10, int nZ=10, bint use_adaptive=1, double simps_err=1e-8):
    cdef Py_ssize_t i
    cdef double p
    cdef sum_logp = 0
    
    for i from 0 <= i < x.shape[0]:
        p = full_pdf(x[i], v, V, a, z, Z, t, T, err, nT, nZ, use_adaptive, simps_err)
        # If one probability = 0, the log sum will be -Inf
        if p == 0:
            return -np.inf
        sum_logp += log(p)
        
    return sum_logp


@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def wiener_like_full(np.ndarray[double, ndim=1] x,
                     np.ndarray[double, ndim=1] v, np.ndarray[double, ndim=1] a,
                     np.ndarray[double, ndim=1] z, np.ndarray[double, ndim=1] t, err):
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef double p
    cdef double sum_logp = 0

    for i from 0 <= i < size:
        p = pdf_sign(x[i], v[i], a[i], z[i], t[i], err)
        # If one probability = 0, the log sum will be -Inf
        if p == 0:
            return -np.inf
        sum_logp += log(p)

    return sum_logp


@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def wiener_like_full_collCont(np.ndarray[double, ndim=1] x,
                              np.ndarray[bint, ndim=1] cont_x, double gamma,
                              double v, double V, double a, double z, double Z,
                              double t, double T, double t_min, double t_max,
                              double err=1e-4, int nT=2, int nZ=2, bint
                              use_adaptive = 1, double simps_err = 1e-3):
    """Wiener likelihood function where RTs could come from a
    separate, uniform contaminant distribution.

    Reference: Lee, Vandekerckhove, Navarro, & Tuernlinckx (2007)
    """
    return 0
#    cdef Py_ssize_t i
#    cdef double p
#    cdef sum_logp = 0
#    for i from 0 <= i < x.shape[0]:
#        if cont_x[i] == 1:
#            p = full_pdf(x[i], v, V, a, z, Z, t, T, err, nT, nZ, use_adaptive, simps_err)
#        elif cont_y[i] == 0:
#            p = prob_boundary(x[i], v, a, z, t, err) * 1./(t_max-t_min)
#        else:
#            p = .5 * 1./(t_max-t_min)
#        #print p, x[i], v, a, z, t, err, t_max, t_min, cont_x[i], cont_y[i]
#        # If one probability = 0, the log sum will be -Inf
#        if p == 0:
#            return -infinity
#
#        sum_logp += log(p)
#        
#    return sum_logp



@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def wiener_like_full_multi(np.ndarray[double, ndim=1] x, v, V, a, z, Z, t, T, double err, multi=None):
    cdef unsigned int size = x.shape[0]
    cdef unsigned int i
    cdef double p = 0

    if multi is None:
        return wiener_like_full_intrp(x, v, V, a, z, Z, t, T, err)
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
    
@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def gen_rts_from_cdf(double v, double V, double a, double z, double Z, double t, \
                           double T, int samples=1000, double cdf_lb = -6, double cdf_ub = 6, double dt=1e-2):
    
    cdef np.ndarray[double, ndim=1] x = np.arange(cdf_lb, cdf_ub, dt)
    cdef np.ndarray[double, ndim=1] l_cdf = np.empty(x.shape[0], dtype=np.double)
    cdef double pdf, rt
    cdef Py_ssize_t i, j
    cdef int idx
    
    l_cdf[0] = 0
    for i from 1 <= i < x.shape[0]:
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

@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def wiener_like_full_contaminant(np.ndarray[double, ndim=1] value, np.ndarray[int, ndim=1] cont_x, double gamma, double v, double V, double a, double z, double Z, double t, double T, double t_min, double t_max, double err):
    """Wiener likelihood function where RTs could come from a
    separate, uniform contaminant distribution.

    Reference: Lee, Vandekerckhove, Navarro, & Tuernlinckx (2007)
    """
    cdef Py_ssize_t i
    cdef double p
    cdef double sum_logp = 0
    cdef int n_cont = np.sum(cont_x)
    cdef int pos_cont = 0
    
    for i from 0 <= i < value.shape[0]:
        if cont_x[i] == 0:
            p = full_pdf(value[i], v, V, a, z, Z, t, T, err)
            if p == 0:
                return -np.inf
            sum_logp += log(p)      
        elif value[i]>0:
            pos_cont += 1
        # If one probability = 0, the log sum will be -Inf
        
    
    # add the log likelihood of the contaminations
    #first the guesses
    sum_logp += n_cont*log(gamma*(0.5 * 1./(t_max-t_min)))     
    #then the positive prob_boundary 
    sum_logp += pos_cont*log((1-gamma) * prob_ub(v, a, z) * 1./(t_max-t_min))
    #and the negative prob_boundary
    sum_logp += (n_cont - pos_cont)*log((1-gamma)*(1-prob_ub(v, a, z)) * 1./(t_max-t_min))

    return sum_logp
