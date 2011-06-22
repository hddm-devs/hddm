#!/usr/bin/python 
#
# Cython version of the Navarro & Fuss, 2009 DDM PDF. Based directly
# on the following code by Navarro & Fuss:
# http://www.psychocmath.logy.adelaide.edu.au/personalpages/staff/danielnavarro/resources/wfpt.m
#
# This implementation is about 170 times faster than the matlab
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
include "pdf.pxi"

ctypedef double * double_ptr
ctypedef void * void_ptr

cdef extern from "stdlib.h":
    void free(void* ptr)
    void* malloc(size_t size)
    void* realloc(void* ptr, size_t size)
    
cdef double wfpt_gsl(double x, void * params):
    cdef double rt, v, v_switch, a, z, t, t_switch, f, T
    rt = (<double_ptr> params)[0]
    v = (<double_ptr> params)[1]
    v_switch = (<double_ptr> params)[2]
    V_switch = (<double_ptr> params)[3]
    a = (<double_ptr> params)[4]
    z = (<double_ptr> params)[5]
    t = (<double_ptr> params)[6]
    t_switch = (<double_ptr> params)[7]
    T = (<double_ptr> params)[8]

    f = pdf_V_sign(rt, v_switch, V_switch, a, x, t+t_switch, 1e-4) * calc_drift_dens_T(x*a, t_switch, v, a, z*a, T)

    return f


cpdef double calc_drift_dens_T(double x, double t, double v, double a, double z, double T):
    if T < 1e-4:
        return calc_drift_dens(x,t,v,a,z,False)
    else:
        return 1/T * (calc_drift_dens(x,t+T/2,v,a,z,True) - calc_drift_dens(x,t-T/2,v,a,z,True))

cpdef double calc_drift_dens(double x, double t, double v, double a, double z, bint integrate_t):
    cdef int N=30
    cdef int n=0
    cdef int got_zero = 0
    cdef double term = 0
    cdef double summed = 0

    #for n from 1 <= n <= N:
    while(got_zero < 5):
        if not integrate_t:
            # Ratcliff 1980 Equation 12
            term = sin(n*M_PI*z/a) * sin(n*M_PI*x/a) * exp(-.5*(v**2 + (n**2*M_PI**2)/a**2)*t)
        else:
            # Indefinite integral over t
            term = sin(n*M_PI*z/a) * sin(n*M_PI*x/a) * (exp(-.5*(v**2 + (n**2*M_PI**2)/a**2)*t) / (-0.5*M_PI**2*n**2/a**2 - 0.5*v**2))
        # Start counting after N terms
        if fabs(term) < 1e-6 and n > N:
            got_zero+=1
        
        summed += term
        n+=1

    return 2 * exp(v*(x-z)) * summed
    
cpdef double pdf_post_switch(double rt, double v, double v_switch,
                             double V_switch, double a, double z, double t,
                             double t_switch, double T, double err):
    cdef double alpha, result, error, expected
    cdef gsl_integration_workspace * W
    W = gsl_integration_workspace_alloc(1000)
    cdef gsl_function F
    cdef double params[9]
    cdef size_t neval
    params[0] = rt
    params[1] = v
    params[2] = v_switch
    params[3] = V_switch
    params[4] = a
    params[5] = z
    params[6] = t
    params[7] = t_switch
    params[8] = T

    F.function = &wfpt_gsl
    F.params = params

    gsl_integration_qag(&F, 0, 1, 1e-2, 1e-2, 1000, GSL_INTEG_GAUSS15, W, &result, &error)
    gsl_integration_workspace_free(W)

    return result

cpdef pdf_switch(double rt, int instruct, double v, double v_switch, double V_switch, double a, double z, double t, double t_switch, double T, double err):
    cdef double p

    if instruct == 0 or (fabs(rt) <= t+t_switch): # Prosaccade or pre-switch
        p = full_pdf(rt, v, 0, a, z, 0, t, T, 1e-4, 2, 2, True, 1e-3)
    elif t_switch < 1e-2:
        # Use regular likelihood
        p = full_pdf(rt, v_switch, 0, a, z, 0, t, T, 1e-4, 2, 2, True, 1e-3)
    else: # post-switch antisaccade
        p = pdf_post_switch(rt, v, v_switch, V_switch, a, z, t, t_switch, T, err)

    return p

@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def wiener_like_antisaccade(np.ndarray[double, ndim=1] rt, np.ndarray instruct, double v, double v_switch, double V_switch, double a, double z, double t, double t_switch, double T, double err):
    cdef Py_ssize_t size = rt.shape[0]
    cdef Py_ssize_t i
    cdef double p
    cdef double sum_logp = 0
        
    for i from 0 <= i < size:
        p = pdf_switch(rt[i], instruct[i], v, v_switch, V_switch, a, z, t, t_switch, T, err)
        if p == 0:
            return -np.inf
        sum_logp += log(p)

    return sum_logp

####################################################################
# Functions to compute antisaccade likelihood with precomputing the
# drift density

# Global variables for density and interpolation
cdef double *drift_density
cdef double *eval_dens
cdef gsl_interp_accel *acc 
cdef gsl_spline *spline

cdef pdf_switch_precomp(double rt, int instruct, double v, double v_switch, double V_switch, double a, double z, double t, double t_switch, double T, double err):
    cdef double p

    if instruct == 0 or (fabs(rt) <= t+t_switch): # Prosaccade or pre-switch
        p = full_pdf(rt, v, 0, a, z, 0, t, T, 1e-4, 2, 2, True, 1e-3)
    elif t_switch < 1e-2:
        # Use regular likelihood
        p = full_pdf(rt, v_switch, 0, a, z, 0, t, T, 1e-4, 2, 2, True, 1e-3)
    else: # post-switch antisaccade
        p = pdf_post_switch_precomp(rt, v, v_switch, V_switch, a, z, t, t_switch, err)

    return p

@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def wiener_like_antisaccade_precomp(np.ndarray[double, ndim=1] rt, np.ndarray instruct, double v, double v_switch, double V_switch, double a, double z, double t, double t_switch, double T, double err, int evals=40):
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
        drift_density[x] = calc_drift_dens_T(eval_dens[x], t_switch, v, a, z*a, T)

    # Init spline
    gsl_spline_init(spline, eval_dens, drift_density, evals)

    for i from 0 <= i < size:
        p = pdf_switch_precomp(rt[i], instruct[i], v, v_switch, V_switch, a, z, t, t_switch, T, err)
        if p == 0:
            return -np.inf
        sum_logp += log(p)

    gsl_spline_free (spline)
    gsl_interp_accel_free (acc)
    
    return sum_logp

cdef double pdf_post_switch_precomp(double rt, double v, double v_switch, double V_switch, double a, double z, double t, double t_switch, double err):
    cdef double alpha, result, error, expected
    cdef gsl_integration_workspace * W
    W = gsl_integration_workspace_alloc(1000)
    cdef gsl_function F
    cdef double params[8]
    cdef size_t neval
    params[0] = rt
    params[1] = v
    params[2] = v_switch
    params[3] = V_switch
    params[4] = a
    params[5] = z
    params[6] = t
    params[7] = t_switch

    F.function = &wfpt_gsl_precomp
    F.params = params

    gsl_integration_qag(&F, 0, 1, 1e-3, 1e-3, 1000, GSL_INTEG_GAUSS15, W, &result, &error)
    gsl_integration_workspace_free(W)

    return result

cdef double wfpt_gsl_precomp(double x, void * params):
    cdef double rt, v, v_switch, V_switch, a, z, t, t_switch, f
    rt = (<double_ptr> params)[0]
    v = (<double_ptr> params)[1]
    v_switch = (<double_ptr> params)[2]
    V_switch = (<double_ptr> params)[3]
    a = (<double_ptr> params)[4]
    z = (<double_ptr> params)[5]
    t = (<double_ptr> params)[6]
    t_switch = (<double_ptr> params)[7]

    global acc, spline

    f = pdf_V_sign(rt, v_switch, V_switch, a, x, t+t_switch, 1e-4) * gsl_spline_eval(spline, x*a, acc)

    return f
