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

cdef extern from "math.h":
    double sin(double)
    double cos(double)
    double log(double)
    double exp(double)
    double sqrt(double)
    double fmax(double, double)
    double pow(double, double)
    double ceil(double)
    double floor(double)
    double fabs(double)

cdef double infinity = np.inf

# Define data type
DTYPE = np.double
ctypedef double DTYPE_t

int_DTYPE = np.int
ctypedef int int_DTYPE_t

cdef double PI = 3.1415926535897
cdef double PIs = 9.869604401089358 # PI^2
    
cpdef double ftt_01w(double tt, double w, double err):
    """Compute f(t|0,1,w) for the likelihood of the drift diffusion model using the method
    and implementation of Navarro & Fuss, 2009.
    """
    cdef double kl, ks, p
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
            p+=(w+2*k)*exp(-(pow((w+2*k),2))/2/tt) # increment sum
        p/=sqrt(2*PI*pow(tt,3)) # add constant term
  
    else: # if large t is better...
        K=<int>(ceil(kl)) # round to smallest integer meeting error
        for k from 1 <= k <= K:
            p+=k*exp(-(pow(k,2))*(PIs)*tt/2)*sin(k*PI*w) # increment sum
        p*=PI # add constant term

    return p

cpdef double pdf(double x, double v, double a, double w, double err):
    """Compute the likelihood of the drift diffusion model f(t|v,a,z) using the method
    and implementation of Navarro & Fuss, 2009.
    """
    if x <= 0:
        return 0

    cdef double tt = x/(pow(a,2)) # use normalized time
    cdef double p = ftt_01w(tt, w, err) #get f(t|0,1,w)
  
    # convert to f(t|v,a,w)
    return p*exp(-v*a*w -(pow(v,2))*x/2.)/(pow(a,2))

cpdef double pdf_sign(double x, double v, double a, double z, double t, double err):
    """Wiener likelihood function for two response types. Lower bound
    responses have negative t, upper boundary response have positive t"""
    if z<0 or z>1 or a<0:
        return 0

    if x<0:
        # Lower boundary
        return pdf(fabs(x)-t, v, a, z, err)
    else:
        # Upper boundary, flip v and z
        return pdf(x-t, -v, a, 1.-z, err)


cpdef double pdf_V(double x, double v, double V, double a, double z, double err):
    """Compute the likelihood of the drift diffusion model f(t|v,a,z,V) using the method    
    and implementation of Navarro & Fuss, 2009.
    V is the std of the drift rate
    """
    if x <= 0:
        return 0
    
    if V==0:
        return pdf(x, v, a, z, err) 
        
    cdef double tt = x/(pow(a,2)) # use normalized time
    cdef double p  = ftt_01w(tt, z, err) #get f(t|0,1,w)
  
    # convert to f(t|v,a,w)
    return p*exp(((a*z*V)**2 - 2*a*v*z - (v**2)*x)/(2*(V**2)*x+2))/sqrt((V**2)*x+1)/(a**2)

cpdef double pdf_V_sign(double x, double v, double V, double a, double z, double t, double err):
    """Wiener likelihood function for two response types. Lower bound
    responses have negative t, upper boundary response have positive t."""
    if z<0 or z>1 or a<0:
        return 0

    if x<0:
        # Lower boundary
        return pdf_V(fabs(x)-t, v, V, a, z, err)
    else:
        # Upper boundary, flip v and z
        return pdf_V(x-t, -v, V, a, 1.-z, err)

@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def pdf_array(np.ndarray[DTYPE_t, ndim=1] x, double v, double a, double z, double t, double err, bint logp=0):
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef np.ndarray[DTYPE_t, ndim=1] y = np.empty(size, dtype=DTYPE)

    for i from 0 <= i < size:
        y[i] = pdf_sign(x[i], v, a, z, t, err)

    if logp==1:
        return np.log(y)
    else:
        return y

@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def wiener_like_simple(np.ndarray[DTYPE_t, ndim=1] x, double v, double a, double z, double t, double err):
    cdef Py_ssize_t i
    cdef double p
    cdef double sum_logp = 0
    for i from 0 <= i < x.shape[0]:
        p = pdf_sign(x[i], v, a, z, t, err)
        # If one probability = 0, the log sum will be -Inf
        if p == 0:
            return -infinity
        sum_logp += log(p)
        
    return sum_logp

cdef inline double prob_ub(double v, double a, double z):
    """Probability of hitting upper boundary."""
    return (exp(-2*a*z*v) - 1) / (exp(-2*a*v) - 1)

@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def wiener_like_simple_contaminant(np.ndarray[DTYPE_t, ndim=1] value, np.ndarray[int_DTYPE_t, ndim=1] cont_x, double gamma, double v, double a, double z, double t, double t_min, double t_max, double err):
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
                return -infinity
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
def wiener_like_simple_multi(np.ndarray[DTYPE_t, ndim=1] x, v, a, z, t, double err, multi=None):
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

