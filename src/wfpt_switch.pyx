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

import scipy.interpolate

cimport cython

include "gsl/gsl.pxi"
include "wfpt.pyx"

ctypedef double * double_ptr
ctypedef void * void_ptr

cdef extern from "stdlib.h":
    void free(void* ptr)
    void* malloc(size_t size)
    void* realloc(void* ptr, size_t size)
    
cdef double wfpt_gsl(double x, void * params):
    cdef double rt, v, v_switch, a, z, t, t_switch, f
    rt = (<double_ptr> params)[0]
    v = (<double_ptr> params)[1]
    v_switch = (<double_ptr> params)[2]
    a = (<double_ptr> params)[3]
    z = (<double_ptr> params)[4]
    t = (<double_ptr> params)[5]
    t_switch = (<double_ptr> params)[6]
    
    f = pdf_sign(rt, v_switch, a, x, t+t_switch, 1e-4) * calc_drift_dens(x*a, t_switch, v, a, z*a) * a
    #f = pdf_sign(rt, v, a, x, t, 1e-4) * (gsl_ran_gaussian_pdf(x, sqrt(t_switch)) + (t_switch * v + (z*a)))
    return f

cdef double calc_drift_dens(double x, double t, double v, double a, double z):
    cdef int N=35
    cdef int n
    cdef double summed = 0

    for n from 1 <= n <= N:
        summed += 2/a * sin(n*PI*z/a) * sin(n*PI*x/a) * exp(-.5*(v**2 + (n**2*PIs)/a**2)*t) # Ratcliff 1980 Equation 12

    return exp(v*(x-z)) * summed
    
cpdef double pdf_Z_norm_sign(double rt, double v, double v_switch, double a, double z, double t, double t_switch, double err):
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

    gsl_integration_qag(&F, 0, 1, 1e-3, 1e-3, 1000, GSL_INTEG_GAUSS15, W, &result, &error)
    gsl_integration_workspace_free(W)

    return result

@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def wiener_like_antisaccade(np.ndarray[DTYPE_t, ndim=1] rt, np.ndarray instruct, double v, double v_switch, double a, double z, double t, double t_switch, double err):
    cdef Py_ssize_t size = rt.shape[0]
    cdef Py_ssize_t i
    cdef double p
    cdef double sum_logp = 0
        
    for i from 0 <= i < size:
        if instruct[i] == 0 or (fabs(rt[i]) <= t+t_switch): # Prosaccade or pre-switch antisaccade
            p = pdf_sign(rt[i], v, a, z, t, err)
        else: # post-switch Antisaccade
            p = pdf_Z_norm_sign(rt[i], v, v_switch, a, z, t, t_switch, err)
        if p == 0:
            return -infinity
        sum_logp += log(p)

    return sum_logp


# Global variable for density
cdef double *drift_density
cdef double *eval_dens
cdef gsl_interp_accel *acc 
cdef gsl_spline *spline

@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def wiener_like_antisaccade_precomp(np.ndarray[DTYPE_t, ndim=1] rt, np.ndarray instruct, double v, double v_switch, double a, double z, double t, double t_switch, double err, int evals=20):
    cdef Py_ssize_t size = rt.shape[0]
    cdef Py_ssize_t i
    cdef Py_ssize_t x
    cdef double p
    cdef double sum_logp = 0

    #############################
    # Precompute drift density

    # Initialize cubic spline gsl variables
    global drift_density, eval_dens, acc, spline
    drift_density = <double_ptr> malloc(evals * sizeof(double))
    eval_dens = <double_ptr> malloc(evals * sizeof(double))
    acc = gsl_interp_accel_alloc()
    spline = gsl_spline_alloc(gsl_interp_cspline, evals)

    # Compute density
    for x from 0 <= x < evals:
        eval_dens[x] = a * (<double>x/(evals-1))
        drift_density[x] = calc_drift_dens(eval_dens[x], t_switch, v, a, z*a)

    # Init spline
    gsl_spline_init(spline, eval_dens, drift_density, evals)

    for i from 0 <= i < size:
        if instruct[i] == 0 or (fabs(rt[i]) <= t+t_switch): # Prosaccade or pre-switch
            p = pdf_sign(rt[i], v, a, z, t, err)
        else: # post-switch antisaccade
            p = pdf_switch_precomp(rt[i], v, v_switch, a, z, t, t_switch, err)
        if p == 0:
            return -infinity
        sum_logp += log(p)

    gsl_spline_free (spline)
    gsl_interp_accel_free (acc)
    
    return sum_logp

cdef double pdf_switch_precomp(double rt, double v, double v_switch, double a, double z, double t, double t_switch, double err):
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

    F.function = &wfpt_gsl_precomp
    F.params = params

    gsl_integration_qag(&F, 0, 1, 1e-3, 1e-3, 1000, GSL_INTEG_GAUSS15, W, &result, &error)
    gsl_integration_workspace_free(W)

    return result

cdef double wfpt_gsl_precomp(double x, void * params):
    cdef double rt, v, v_switch, a, z, t, t_switch, f
    rt = (<double_ptr> params)[0]
    v = (<double_ptr> params)[1]
    v_switch = (<double_ptr> params)[2]
    a = (<double_ptr> params)[3]
    z = (<double_ptr> params)[4]
    t = (<double_ptr> params)[5]
    t_switch = (<double_ptr> params)[6]

    global drift_density, eval_dens, acc, spline

    f = pdf_sign(rt, v_switch, a, x, t+t_switch, 1e-4) * gsl_spline_eval(spline, x*a, acc) * a

    return f
