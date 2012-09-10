#cython: embedsignature=True
#cython: cdivision=True
#cython: wraparound=False
#cython: boundscheck=False
#
# Copyleft Thomas Wiecki (thomas_wiecki[at]brown.edu), 2012
# GPLv3

from copy import copy
import numpy as np
cimport numpy as np

from cython_gsl cimport *

include 'integrate.pxi'
include 'cdf.pxi'

ctypedef double * double_ptr
ctypedef void * void_ptr

from libc.math cimport *

cdef extern from "stdlib.h":
    void free(void* ptr)
    void* malloc(size_t size)
    void* realloc(void* ptr, size_t size)

cdef double wfpt_gsl(double x, void * params) nogil:
    cdef double rt, v, v_switch, a, z, t, t_switch, f, st
    rt = (<double_ptr> params)[0]
    v = (<double_ptr> params)[1]
    v_switch = (<double_ptr> params)[2]
    V_switch = (<double_ptr> params)[3]
    a = (<double_ptr> params)[4]
    z = (<double_ptr> params)[5]
    t = (<double_ptr> params)[6]
    t_switch = (<double_ptr> params)[7]
    st = (<double_ptr> params)[8]

    f = full_pdf(rt, v_switch, V_switch, a, x, 0, t+t_switch, 0, 1e-4) * calc_drift_dens_st(x*a, t_switch, v, a, z*a, st)

    return f

cdef double calc_drift_dens_st(double x, double t, double v, double a, double z, double st) nogil:
    if st < 1e-4:
        return calc_drift_dens(x, t, v, a, z,False)
    else:
        return 1/st * (calc_drift_dens(x, t+st/2, v, a, z, True) - calc_drift_dens(x, t-st/2, v, a, z, True))

cpdef double calc_drift_dens(double x, double t, double v, double a, double z, bint integrate_t) nogil:
    cdef int N=40
    cdef int n=0
    cdef int got_zero = 0
    cdef double term = 0
    cdef double summed = 0
    cdef double divisor

    # Compute sum minimum of 60 evals or until 5 zeroes have been encountered.
    while(got_zero < 10 and n < N):
        if not integrate_t:
            # Ratcliff 1980 Equation 12
            term = sin(n*M_PI*z/a) * sin(n*M_PI*x/a) * exp(-.5*(v**2 + (n**2*M_PI**2)/a**2)*t)
        else:
            # Indefinite integral over t
            divisor = (-0.5*M_PI**2*n**2/a**2 - 0.5*v**2)
            #if divisor == 0:
            #    print n, a, v
            term = sin(n*M_PI*z/a) * sin(n*M_PI*x/a) * (exp(-.5*(v**2 + (n**2*M_PI**2)/a**2)*t) / divisor)
        # Start counting after N terms
        if fabs(term) < 1e-6 and n > N:
            got_zero+=1

        summed += term
        n+=1
        #if term == -np.inf or term == +np.inf:
            #print x, t, v, a, z, n, term
            #return 0

    return 2 * exp(v*(x-z)) * summed

cdef double pdf_post_switch(double rt, double v, double v_switch,
                             double V_switch, double a, double z, double t,
                             double t_switch, double st, double err) nogil:
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
    params[8] = st

    F.function = &wfpt_gsl
    F.params = params

    gsl_integration_qag(&F, 0, 1, 1e-2, 1e-2, 1000, GSL_INTEG_GAUSS15, W, &result, &error)
    gsl_integration_workspace_free(W)

    return result

cpdef double pdf_switch(double rt, double v, double v_switch, double V_switch, double a, double z, double t, double t_switch, double st, double err) nogil:
    cdef double p

    if fabs(rt) < t-st/2 or t < st/2 or t_switch < st/2 or t<0 or t_switch<0 or st<0 or a<=0 or z<=0 or z>=1 or st>.5:
        return 0

    if (fabs(rt) <= t+t_switch): # Prosaccade or pre-switch
        p = full_pdf(rt, v, 0, a, z, 0, t, st, 1e-4, 2, 2, True, 1e-3)
    elif t_switch < 0.001:
        # Use regular likelihood
        p = full_pdf(rt, v_switch, 0, a, z, 0, t, st, 1e-4, 2, 2, True, 1e-3)
    else: # post-switch antisaccade
        p = pdf_post_switch(rt, v, v_switch, V_switch, a, z, t, t_switch, st, err)

    return p

def wiener_like_antisaccade(np.ndarray[double, ndim=1] rt, double v, double v_switch, double V_switch, double a, double z, double t, double t_switch, double st, double err):
    cdef Py_ssize_t size = rt.shape[0]
    cdef Py_ssize_t i
    cdef double p
    cdef double sum_logp = 0

    if np.any(np.abs(rt) < t-st/2) or t < st/2 or t_switch < st/2 or t<0 or t_switch<0 or st<0 or a<=0 or z<=0 or z>=1 or st>.5:
        return -np.inf

    for i in range(size):
        p = pdf_switch(rt[i], v, v_switch, V_switch, a, z, t, t_switch, st, err)
        if p == 0:
            return -np.inf
        #if p < 0:
        #    print rt[i], instruct[i], v, v_switch, V_switch, a, z, t, t_switch, st
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

@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def wiener_like_antisaccade_precomp(np.ndarray[double, ndim=1] rt, double v, double v_switch, double V_switch, double a, double z, double t, double t_switch, double st, double err, int evals=40, double t_switch_cutoff=.02):
    cdef Py_ssize_t size = rt.shape[0]
    cdef Py_ssize_t i
    cdef Py_ssize_t x
    cdef double p
    cdef double sum_logp = 0

    if np.any(np.abs(rt) < t-st/2) or t < st/2 or t_switch == 0 or t_switch < st/2 or t<0 or t_switch<0 or st<0 or a<=0 or z<=0 or z>=1 or st>.5:
        return -np.inf

    #############################
    # Precompute drift density

    # Initialize cubic spline gsl variables
    global drift_density, eval_dens, acc, spline
    drift_density = <double_ptr> malloc(evals * sizeof(double))
    eval_dens = <double_ptr> malloc(evals * sizeof(double))
    acc = gsl_interp_accel_alloc()
    spline = gsl_spline_alloc(gsl_interp_cspline, evals)

    # Compute density
    for x in range(evals):
        eval_dens[x] = a * (<double>x/(evals-1))
        #if x == evals-1:
        #    eval_dens[x] = a - 1e-3
        if t_switch < t_switch_cutoff:
            # If too small, approximate drift-density with normal distribution
            drift_density[x] = gsl_ran_gaussian_pdf(eval_dens[x] - (t_switch*v + z*a), sqrt(t_switch))
        else:
            drift_density[x] = calc_drift_dens_st(eval_dens[x], t_switch, v, a, z*a, st)
        if np.isnan(drift_density[x]) or drift_density[x] < 0 or fabs(drift_density[x]) > 100:

            if (drift_density[x] < 0 and drift_density[x] > -1e-2) or x == 1:
                drift_density[x] = 0
            else:
                print eval_dens[x], drift_density[x], x, t_switch, v, a, z*a, st
                # Ran into numerical stability issues, abort.
                free(drift_density)
                free(eval_dens)
                gsl_spline_free (spline)
                gsl_interp_accel_free (acc)
                return -np.inf
        #    return 0

    # Init spline
    gsl_spline_init(spline, eval_dens, drift_density, evals)

    for i in range(size):
        p = pdf_switch_precomp(rt[i], v, v_switch, V_switch, a, z, t, t_switch, st, err)
        if np.isnan(log(p)):
            print p, rt[i], v, v_switch, V_switch, a, z, t, t_switch, st, err
        if p == 0:
            sum_logp = -np.inf
            break
        sum_logp += log(p)

    free(drift_density)
    free(eval_dens)
    gsl_spline_free (spline)
    gsl_interp_accel_free (acc)

    return sum_logp

cdef double pdf_switch_precomp(double rt, double v, double v_switch, double V_switch, double a, double z, double t, double t_switch, double st, double err) nogil:
    cdef double p

    if fabs(rt) < t-st/2 or t < st/2 or t_switch < st/2 or t<0 or t_switch<0 or st<0 or a<=0 or z<=0 or z>=1 or st>.5:
        return 0

    if (fabs(rt) <= t+t_switch): # Prosaccade or pre-switch
        p = full_pdf(rt, v, 0, a, z, 0, t, st, 1e-4, 2, 2, True, 1e-3)
    #elif t_switch < 0.05:
        # Use regular likelihood
        #p = full_pdf(rt, v_switch, 0, a, z, 0, t+t_switch, st, 1e-4, 2, 2, True, 1e-3)
    else: # post-switch antisaccade
        p = pdf_post_switch_precomp(rt, v, v_switch, V_switch, a, z, t, t_switch, err)

    return p

cdef double pdf_post_switch_precomp(double rt, double v, double v_switch, double V_switch, double a, double z, double t, double t_switch, double err) nogil:
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

cdef double wfpt_gsl_precomp(double x, void * params) nogil:
    cdef double rt, v, v_switch, V_switch, a, z, t, t_switch, f, interp
    rt = (<double_ptr> params)[0]
    v = (<double_ptr> params)[1]
    v_switch = (<double_ptr> params)[2]
    V_switch = (<double_ptr> params)[3]
    a = (<double_ptr> params)[4]
    z = (<double_ptr> params)[5]
    t = (<double_ptr> params)[6]
    t_switch = (<double_ptr> params)[7]

    global acc, spline

    interp = gsl_spline_eval(spline, x*a, acc)
    if interp <= 0:
        return 0
    f = full_pdf(rt, v_switch, V_switch, a, x, 0, t+t_switch, 0, 1e-4) * interp

    return f
