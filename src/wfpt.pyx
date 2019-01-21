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

import scipy.integrate as integrate
from copy import copy
import numpy as np

cimport numpy as np
cimport cython

from cython.parallel import *
#cimport openmp

#include "pdf.pxi"
include 'integrate.pxi'

def pdf_array(np.ndarray[double, ndim=1] x, double v, double sv, double a, double z, double sz,
              double t, double st, double err=1e-4, bint logp=0, int n_st=2, int n_sz=2, bint use_adaptive=1,
              double simps_err=1e-3, double p_outlier=0, double w_outlier=0):

    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef np.ndarray[double, ndim=1] y = np.empty(size, dtype=np.double)

    for i in prange(size, nogil=True):
        y[i] = full_pdf(x[i], v, sv, a, z, sz, t, st, err, n_st, n_sz, use_adaptive, simps_err)

    y = y * (1 - p_outlier) + (w_outlier * p_outlier)
    if logp==1:
        return np.log(y)
    else:
        return y

cdef inline bint p_outlier_in_range(double p_outlier): return (p_outlier >= 0) & (p_outlier <= 1)

def wiener_like(np.ndarray[double, ndim=1] x, double v, double sv, double a, double z, double sz, double t,
                double st, double err, int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-8,
                double p_outlier=0, double w_outlier=0):
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef double p
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier

    if not p_outlier_in_range(p_outlier):
        return -np.inf

    for i in range(size):
        #print("rt = %.2f v = %.2f a = %.2f t = %.2f z = %.2f sv = %.2f st = %.2f err = %.2f n_st = %.2f n_sz = %.2f use_adaptive = %.2f simps_err = %.2f p_outlier = %.2f w_outlier = %.2f" % (x[i],v,a,t,z,sv,st, err, n_st, n_sz, use_adaptive, simps_err,p_outlier,w_outlier))
        p = full_pdf(x[i], v, sv, a, z, sz, t, st, err, n_st, n_sz, use_adaptive, simps_err)
        # If one probability = 0, the log sum will be -Inf
        #print('p: ',p)
        p = p * (1 - p_outlier) + wp_outlier
        #print('after p: ',p)
        if p == 0:
            return -np.inf

        sum_logp += log(p)

    return sum_logp

def wiener_like_rlddm(np.ndarray[double, ndim=1] x, 
                      np.ndarray[double, ndim=1] response,
                      np.ndarray[double, ndim=1] feedback,
                      np.ndarray[double, ndim=1] q,
                      np.ndarray[long, ndim=1] split_by,
                      long unique,
                      double alpha, double dual_alpha, double v, double sv, double a, double z, double sz, double t,
                      double st, double err, int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-8,
                      double p_outlier=0, double w_outlier=0):
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef int s
    cdef int s_size
    #cdef double sd
    #cdef int n_up = 0
    #cdef int n_low = 0
    #cdef double sd_up
    #cdef double sd_low
    cdef double p
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier
    cdef double alfa = 0
    cdef double neg_alpha = np.exp(alpha)/(1+np.exp(alpha))
    cdef double pos_alpha = np.exp(alpha + dual_alpha)/(1+np.exp(alpha + dual_alpha))
    #cdef double exp_ups
    #cdef double exp_lows
    #cdef np.ndarray rew_ups
    #cdef np.ndarray rew_lows
    cdef np.ndarray feedbacks
    cdef np.ndarray responses
    cdef np.ndarray xs
    cdef np.ndarray qs
    
    if not p_outlier_in_range(p_outlier):
        return -np.inf
    
    # unique represent # of conditions
    for s in range(unique):
        #select trials for current condition, identified by the split_by-array
        qs = q
        feedbacks = feedback[split_by==s]
        responses = response[split_by==s]
        xs = x[split_by==s]
        s_size = xs.shape[0]
        
        # don't calculate pdf for first trial but still update q
        if feedbacks[0] > qs[responses[0]]:
            alfa = pos_alpha
        else:
            alfa = neg_alpha
            
        #qs[1] is upper bound, qs[0] is lower bound. feedbacks is reward received on current trial.
        qs[responses[0]] = qs[responses[0]]+alfa*(feedbacks[0]-qs[responses[0]])
        
        #loop through all trials in current condition
        for i in range(1,s_size):
            
            #calculate uncertainty:
            #sd_up = np.sqrt((exp_ups[i]*(1-exp_ups[i]))/(n_up+1))
            #sd_low = np.sqrt((exp_lows[i]*(1-exp_lows[i]))/(n_low+1))
            #sd = sd_up + sd_low + 1
            #exp_ups[i]-exp_lows[i])*v)/sd
            #print("n_up = %.2f n_low = %.2f sd_up = %.2f sd_low = %.2f sd = %.2f exp_up = %.2f exp_low = %.2f" % (n_up,n_low,sd_up,sd_low,sd,exp_ups[i],exp_lows[i]))
            print("rt = %.2f drift = %.2f v = %.2f alpha = %.2f dual_alpha = %.2f a = %.2f qup = %.2f qlow = %.2f feedback = %.2f responses = %.2f split = %.2f t = %.2f z = %.2f sv = %.2f st = %.2f err = %.2f n_st = %.2f n_sz = %.2f use_adaptive = %.2f simps_err = %.2f p_outlier = %.2f w_outlier = %.2f" % (xs[i],(qs[1]-qs[0])*v,v,alpha,dual_alpha,a,qs[1],qs[0],feedbacks,responses[i],s,t,z,sv,st, err, n_st, n_sz, use_adaptive, simps_err,p_outlier,w_outlier))
            p = full_pdf(xs[i], (qs[1]-qs[0])*v, sv, a, z, sz, t, st, err, n_st, n_sz, use_adaptive, simps_err)
            # If one probability = 0, the log sum will be -Inf
            #print('p: ',p)
            p = p * (1 - p_outlier) + wp_outlier
            #print('p after: ',p) 
            if p == 0:
                return -np.inf
            sum_logp += log(p)
            
            # calculate learning rate for current trial. if dual_alpha is not in include it will be same as alpha so can still use this calculation:
            if feedbacks[i] > qs[responses[i]]:
                alfa = pos_alpha
            else:
                alfa = neg_alpha
            
            #qs[1] is upper bound, qs[0] is lower bound. feedbacks is reward received on current trial.
            qs[responses[i]] = qs[responses[i]]+alfa*(feedbacks[i]-qs[responses[i]])

    return sum_logp
  
def wiener_like_rl(np.ndarray[double, ndim=1] response,
                      np.ndarray[double, ndim=1] rew_up, 
                      np.ndarray[double, ndim=1] rew_low, 
                      np.ndarray[double, ndim=1] exp_up,
                      np.ndarray[double, ndim=1] exp_low, 
                      np.ndarray[long, ndim=1] split_by,
                      long unique,
                      double alpha, double dual_alpha, double v, double sv, double z, double sz, double t,
                      double st, double err, int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-8,
                      double p_outlier=0, double w_outlier=0):
    cdef Py_ssize_t size = response.shape[0]
    cdef Py_ssize_t i
    cdef int s
    cdef int s_size
    cdef double p
    cdef double drift
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier
    cdef double alfa = 0
    cdef double neg_alpha = np.exp(alpha)/(1+np.exp(alpha))
    cdef double pos_alpha = np.exp(alpha + dual_alpha)/(1+np.exp(alpha + dual_alpha))
    cdef np.ndarray exp_ups
    cdef np.ndarray exp_lows
    cdef np.ndarray rew_ups
    cdef np.ndarray rew_lows
    cdef np.ndarray responses
    #cdef np.ndarray xs
    
    if not p_outlier_in_range(p_outlier):
        return -np.inf
    
    # unique represent # of conditions
    for s in range(unique):
        #select trials for current condition, identified by the split_by-array
        exp_ups = exp_up[split_by==s]
        exp_lows = exp_low[split_by==s]
        rew_ups = rew_up[split_by==s]
        rew_lows = rew_low[split_by==s]
        responses = response[split_by==s]
        #xs = x[split_by==s]
        s_size = responses.shape[0]
        
        #loop through all trials in current condition
        for i in range(1,s_size):
            
            # calculate learning rate for current trial. if dual_alpha is not in include it will be 0 so can still use this calculation:
            if responses[i-1] == 0:
                if rew_lows[i-1] > exp_lows[i-1]:
                    alfa = pos_alpha
                else:
                    alfa = neg_alpha
            else:
                if rew_ups[i-1] > exp_ups[i-1]:
                    alfa = pos_alpha
                else:
                    alfa = neg_alpha
            
            #exp[1,x] is upper bound, exp[0,x] is lower bound. same for rew.
            exp_ups[i] = (exp_ups[i-1]*(1-responses[i-1])) + ((responses[i-1])*(exp_ups[i-1]+(alfa*(rew_ups[i-1]-exp_ups[i-1]))))
            exp_lows[i] = (exp_lows[i-1]*(responses[i-1])) + ((1-responses[i-1])*(exp_lows[i-1]+(alfa*(rew_lows[i-1]-exp_lows[i-1]))))
            
            drift = (exp_ups[i]-exp_lows[i])*v
            if drift == 0:
              p = 0.5
            else:
              if responses[i] == 1:
                p = (np.exp(-2*z*drift)-1)/(np.exp(-2*drift)-1)
              else:
                p = 1-(np.exp(-2*z*drift)-1)/(np.exp(-2*drift)-1)
            # If one probability = 0, the log sum will be -Inf
            #print("p = %.2f drift = %.2f v = %.2f alpha = %.2f dual_alpha = %.2f a = %.2f exp_up = %.2f exp_low = %.2f rew_up = %.2f rew_low = %.2f responses = %.2f split = %.2f t = %.2f z = %.2f sv = %.2f st = %.2f err = %.2f n_st = %.2f n_sz = %.2f use_adaptive = %.2f simps_err = %.2f p_outlier = %.2f w_outlier = %.2f" % (p,(exp_ups[i]-exp_lows[i])*v,v,alpha,dual_alpha,a,exp_ups[i],exp_lows[i],rew_ups[i],rew_lows[i],responses[i],s,t,z,sv,st, err, n_st, n_sz, use_adaptive, simps_err,p_outlier,w_outlier))
            #print('p: ',p)
            p = p * (1 - p_outlier) + wp_outlier
            #print('p after: ',p) 
            if p == 0:
                return -np.inf

            sum_logp += log(p)

    return sum_logp

def wiener_like_multi(np.ndarray[double, ndim=1] x, v, sv, a, z, sz, t, st, double err, multi=None,
                      int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-3,
                      double p_outlier=0, double w_outlier=0):
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef double p = 0
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier

    if multi is None:
        return full_pdf(x, v, sv, a, z, sz, t, st, err)
    else:
        params = {'v':v, 'z':z, 't':t, 'a':a, 'sv':sv, 'sz':sz, 'st':st}
        params_iter = copy(params)
        for i in range(size):
            for param in multi:
                params_iter[param] = params[param][i]

            p = full_pdf(x[i], params_iter['v'],
                         params_iter['sv'], params_iter['a'], params_iter['z'],
                         params_iter['sz'], params_iter['t'], params_iter['st'],
                         err, n_st, n_sz, use_adaptive, simps_err)
            p = p * (1 - p_outlier) + wp_outlier
            sum_logp += log(p)

        return sum_logp

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

def gen_cdf_using_pdf(double v, double sv, double a, double z, double sz, double t, double st, double err,
            int N=500, double time=5., int n_st=2, int n_sz=2, bint use_adaptive=1, double simps_err=1e-3,
            double p_outlier=0, double w_outlier=0):
    """
    generate cdf vector using the pdf
    """
    if (sv < 0) or (a <=0 ) or (z < 0) or (z > 1) or (sz < 0) or (sz > 1) or (z+sz/2.>1) or \
    (z-sz/2.<0) or (t-st/2.<0) or (t<0) or (st < 0) or not p_outlier_in_range(p_outlier):
        raise ValueError("at least one of the parameters is out of the support")

    cdef np.ndarray[double, ndim=1] x = np.linspace(-time, time, 2*N+1)
    cdef np.ndarray[double, ndim=1] cdf_array = np.empty(x.shape[0], dtype=np.double)
    cdef int idx

    #compute pdf on the real line
    cdf_array = pdf_array(x, v, sv, a, z, sz, t, st, err, 0, n_st, n_sz, use_adaptive, simps_err, p_outlier, w_outlier)

    #integrate
    cdf_array[1:] = integrate.cumtrapz(cdf_array)

    #normalize
    cdf_array /= cdf_array[x.shape[0]-1]

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
