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
def pdf_array(np.ndarray[double, ndim=1] x, double v, double a, double z, double t, double err, bint logp=0):
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef np.ndarray[double, ndim=1] y = np.empty(size, dtype=np.double)

    for i from 0 <= i < size:
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
    for i from 0 <= i < x.shape[0]:
        p = pdf_sign(x[i], v, a, z, t, err)
        # If one probability = 0, the log sum will be -Inf
        if p == 0:
            return -np.inf
        sum_logp += log(p)
        
    return sum_logp

@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def wiener_like_simple_contaminant(np.ndarray[double, ndim=1] value, np.ndarray[int, ndim=1] cont_x, double gamma, double v, double a, double z, double t, double t_min, double t_max, double err):
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
            p = pdf_sign(value[i], v, a, z, t, err)
            if p == 0:
                return -np.inf
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

