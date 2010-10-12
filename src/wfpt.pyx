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
from __future__ import division
from copy import copy
import numpy as np
cimport numpy as np

cimport cython

cdef extern from "math.h":
    double sin(double)
    double log(double)
    double exp(double)
    double sqrt(double)
    double fmax(double, double)
    double pow(double, double)
    double ceil(double)
    double floor(double)
    double fabs(double)

# Define data type
DTYPE = np.double
ctypedef double DTYPE_t

cpdef double pdf(double t, double v, double a, double z, double err, unsigned int logp=0):
    """Compute the likelihood of the drift diffusion model using the method
    and implementation of Navarro & Fuss, 2009.
    """
    if t <= 0:
        if logp == 0:
            return 0
        else:
            return -np.Inf
    cdef double tt = t/(pow(a,2)) # use normalized time
    cdef double w = z/a # convert to relative start point
    
    cdef double kl, ks, p
    cdef double PI = 3.1415926535897
    cdef double PIs = 9.869604401089358 # PI^2
    cdef int k, K, lower, upper

    # calculate number of terms needed for large t
    if PI*tt*err<1: # if error threshold is set low enough
        kl=sqrt(-2*log(PI*tt*err)/(PIs*tt)) # bound
        kl=fmax(kl,1./(PI*sqrt(tt))) # ensure boundary conditions met
    else: # if error threshold set too high
        kl=1./(PI*sqrt(tt)) # set to boundary condition


    # calculate number of terms needed for small t
    if 2*sqrt(2*PI*tt)*err<1: # if error threshold is set low enough
        ks=2+sqrt(-2*tt*log(2*sqrt(2*PI*tt)*err)) # bound
        ks=fmax(ks,sqrt(tt)+1) # ensure boundary conditions are met
    else: # if error threshold was set too high
        ks=2 # minimal kappa for that case

    # compute f(tt|0,1,w)
    p=0 #initialize density
    if ks<kl: # if small t is better (i.e., lambda<0)
        K=<int>(ceil(ks)) # round to smallest integer meeting error
        lower = <int>(-floor((K-1)/2.))
        upper = <int>(ceil((K-1)/2.))
        for k from lower <= k <= upper: # loop over k
            p=p+(w+2*k)*exp(-(pow((w+2*k),2))/2/tt) # increment sum
        p=p/sqrt(2*PI*pow(tt,3)) # add constant term
  
    else: # if large t is better...
        K=<int>(ceil(kl)) # round to smallest integer meeting error
        for k from 1 <= k <= K:
            p=p+k*exp(-(pow(k,2))*(PIs)*tt/2)*sin(k*PI*w) # increment sum
        p=p*PI # add constant term

    # convert to f(t|v,a,w)
    if logp == 0:
        return p*exp(-v*a*w -(pow(v,2))*t/2.)/(pow(a,2))
    else:
        return log(p) + (-v*a*w -(pow(v,2))*t/2.) - 2*log(a)

cpdef double pdf_sign(double t, double v, double a, double z, double ter, double err, int logp=0):
    """Wiener likelihood function for two response types. Lower bound
    responses have negative t, upper boundary response have positive t"""
    if a<z or z<=0 or z<0 or a<0:
        return -np.Inf

    if t<0:
        return pdf(fabs(t)-ter, v, a, z, err, logp)
    else:
        return pdf(t-ter, -v, a, a-z, err, logp)
    
    
@cython.boundscheck(False) # turn of bounds-checking for entire function
def pdf_array(np.ndarray[DTYPE_t, ndim=1] x, double v, double a, double z, double ter, double err, int logp=0):
    cdef unsigned int size = x.shape[0]
    cdef unsigned int i
    cdef np.ndarray[DTYPE_t, ndim=1] y = np.empty(size, dtype=DTYPE)
    for i from 0 <= i < size:
        y[i] = pdf_sign(x[i], v, a, z, ter, err, logp)
    return y

@cython.boundscheck(False) # turn of bounds-checking for entire function
def pdf_array_multi(np.ndarray[DTYPE_t, ndim=1] x, v, a, z, ter, double err, int logp=0, multi=None):
    cdef unsigned int size = x.shape[0]
    cdef unsigned int i
    cdef np.ndarray[DTYPE_t, ndim=1] y = np.empty(size, dtype=DTYPE)

    if multi is None:
        return pdf_array(x, v=v, a=a, z=z, ter=ter, err=err, logp=logp)
    else:
        params = {'v':v, 'z':z, 'ter':ter, 'a':a}
        params_iter = copy(params)
        for i from 0 <= i < size:
            for param in multi:
                params_iter[param] = params[param][i]
                
            y[i] = pdf_sign(x[i], params_iter['v'], params_iter['a'], params_iter['z'], params_iter['ter'], err=err, logp=logp)

        return y
    

@cython.boundscheck(False) # turn of bounds-checking for entire function
def wiener_like_full_avg_interp(np.ndarray[DTYPE_t, ndim=1] x, double v, double sv, double z, double sz, double ter, double ster, double a, double err=0.0001, int logp=0, int logspace=0, double samples=50., unsigned int reps=10, unsigned int k=2):
    cdef double max_rt = np.max(np.abs(x))
    cdef np.ndarray[DTYPE_t, ndim=1] out = np.empty_like(x)
    
    import scipy.interpolate
    if logspace:
        x_pos = (np.logspace(0,1,samples)-1)*(4./9.)
        x_interp = np.concatenate((-x_pos[::-1], x_pos))
    else:
        x_lower = np.linspace(-max_rt, 0, samples/2.)
        x_upper = np.linspace(0, max_rt, samples/2.)
            
    wfpt_lower = wiener_like_full_avg(x=x_lower,
                                      v=v,
                                      sv=sv,
                                      z=z,
                                      sz=sz,
                                      ter=ter,
                                      ster=ster,
                                      a=a, err=err, reps=reps, logp=0)
    wfpt_upper = wiener_like_full_avg(x=x_upper,
                                      v=v,
                                      sv=sv,
                                      z=z,
                                      sz=sz,
                                      ter=ter,
                                      ster=ster,
                                      a=a, err=err, reps=reps, logp=out)

    out[x<0] = scipy.interpolate.InterpolatedUnivariateSpline(x_lower, wfpt_lower,k=k)(x[x<0])
    out[x>0] = scipy.interpolate.InterpolatedUnivariateSpline(x_upper, wfpt_upper,k=k)(x[x>0])

    if logp == 1:
        return np.log(out)
    else:
        return out
    
@cython.boundscheck(False) # turn of bounds-checking for entire function
def wiener_like_full_avg(np.ndarray[DTYPE_t, ndim=1] x, double v, double sv, double z, double sz, double ter, double ster, double a, double err=.0001, int logp=0, unsigned int reps=10):
    cdef unsigned int num_resps = x.shape[0]
    cdef unsigned int rep, i

    if logp == 1:
        zero_prob = -np.Inf
    else:
        zero_prob = 0
        
    # Create samples
    cdef np.ndarray[DTYPE_t, ndim=1] ter_samples = np.random.uniform(size=reps, low=ter-ster/2., high=ter+ster/2.)
    cdef np.ndarray[DTYPE_t, ndim=1] z_samples = np.random.uniform(size=reps, low=z-sz/2., high=z+sz/2.)
    cdef np.ndarray[DTYPE_t, ndim=1] v_samples = np.random.normal(size=reps, loc=v, scale=sv)
    cdef np.ndarray[DTYPE_t, ndim=2] probs = np.empty((reps,num_resps), dtype=DTYPE)

    for rep from 0 <= rep < reps:
        for i from 0 <= i < num_resps:
            if (fabs(x[i])-ter_samples[rep]) < 0:
                probs[rep,i] = zero_prob
            elif a <= z_samples[rep]:
                probs[rep,i] = zero_prob
            else:
                probs[rep,i] = pdf_sign(x[i], v_samples[rep], a, z_samples[rep], ter_samples[rep], err=err, logp=logp)

    return np.mean(probs, axis=0)
