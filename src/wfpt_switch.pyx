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

include "gsl/gsl.pxi"
include "wfpt.pyx"

ctypedef double * double_ptr
ctypedef void * void_ptr

cdef double wfpt_gsl(double x, void * params):
    cdef double rt, v, v_switch, a, z, t, t_switch, f
    rt = (<double_ptr> params)[0]
    v = (<double_ptr> params)[1]
    v_switch = (<double_ptr> params)[2]
    a = (<double_ptr> params)[3]
    z = (<double_ptr> params)[4]
    t = (<double_ptr> params)[5]
    t_switch = (<double_ptr> params)[6]
    
    f = pdf_sign(rt, v_switch, a, x, t+t_switch, 1e-4) * drift_dens(x*a, t_switch, v, a, z*a) * a
    #f = pdf_sign(rt, v, a, x, t, 1e-4) * (gsl_ran_gaussian_pdf(x, sqrt(t_switch)) + (t_switch * v + (z*a)))
    return f

cpdef inline double drift_dens_term_small(double x, double t, double v, double a, double z, int n):
    # Ratcliff 1980 Equation 13
    cdef double x1 = 2*n*a
    cdef double x2 = 2*a - 2*z - x1

    return exp(v*x1 - ((x - z - x1 - v*t)**2 / 2*t)) - exp(v*x2 - ((x - z - x2 - v*t)**2 / 2*t))

cpdef double drift_dens_small(double x, double t, double v, double a, double z):
    cdef int N=200
    cdef int i,idx=0
    #cdef double terms[2*40+1]
    cdef double summed = 0
    
    for i from -N <= i <= N:
        summed += drift_dens_term_small(x, t, v, a, z, i)
        #summed += terms[idx]
        #idx+=1

    return 1/(2*PIs*t)**.5 * summed

cpdef inline double drift_dens_term(double x, double t, double v, double a, double z, int n):
    # Ratcliff 1980 Equation 12
    return 2/a * sin(n*PI*z/a) * sin(n*PI*x/a) * exp(-.5*(v**2 + (n**2*PIs)/a**2)*t)

cpdef double drift_dens(double x, double t, double v, double a, double z):
    cdef int N=35
    cdef int i
    #cdef double terms[35]
    #cdef double sum_accel, err
    cdef double summed = 0
    #cdef gsl_sum_levin_u_workspace * w = gsl_sum_levin_u_alloc(N)
    
    for i from 1 <= i <= N:
        #terms[i-1] = drift_dens_term(x, t, v, a, z, i)
        #summed += terms[i-1]
        summed += drift_dens_term(x, t, v, a, z, i)

    #gsl_sum_levin_u_accel(terms, N, w, &sum_accel, &err)
    #gsl_sum_levin_u_free(w)

    #print summed,sum_accel
    #print err

    return exp(v*(x-z)) * summed
    
cdef double pdf_Z_norm_sign(double rt, double v, double v_switch, double a, double z, double t, double t_switch, double err):
    cdef double alpha, result, error, expected
    cdef gsl_integration_workspace * W
    W = gsl_integration_workspace_alloc(1000)
    cdef gsl_function F
    cdef double params[7]
    cdef size_t neval
    params[0] = rt
    params[1] = v
    params[2] = v_switch
    params[3] = a
    params[4] = z
    params[5] = t
    params[6] = t_switch

    F.function = &wfpt_gsl
    F.params = params

    #gsl_integration_qag(&F, 0, 1, 1e-2, 1e-2, 1000, GSL_INTEG_GAUSS15, W, &result, &error)
    gsl_integration_qng(&F, 0, 1, 1e-2, 1e-2, &result, &error, &neval)
    gsl_integration_workspace_free(W)

    return result

cpdef switch_pdf(DTYPE_t rt, double v, double v_switch, double a, double z, double t, double t_switch, double err):
    cdef double p

    if (fabs(rt) <= t+t_switch):
        # Pre switch
        p = pdf_sign(rt, v, a, z, t, err)
    else:
        # Post switch
        p = pdf_Z_norm_sign(rt, v, v_switch, a, z, t, t_switch, err)
    
    return p

@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def wiener_like_antisaccade(np.ndarray[DTYPE_t, ndim=1] rt, np.ndarray[int, ndim=1] instruct, double v, double v_switch, double a, double z, double t, double t_switch, double err):
    cdef Py_ssize_t size = rt.shape[0]
    cdef Py_ssize_t i
    cdef double p
    cdef double sum_logp = 0

    for i from 0 <= i < size:
        if instruct[i] == 0: # Prosaccade
            p = pdf_sign(rt[i], v, a, z, t, err)
        else: # Antisaccade
            p = switch_pdf(rt[i], v, v_switch, a, z, t, t_switch, err)
        if p == 0:
            return -infinity
        sum_logp += log(p)

    return sum_logp