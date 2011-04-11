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

# Define data type
DTYPE = np.double
ctypedef double DTYPE_t


cpdef double ftt_01w(double tt, double w, double err):
    """Compute f(t|0,1,w) for the likelihood of the drift diffusion model using the method
    and implementation of Navarro & Fuss, 2009.
    """
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

    return p

cpdef double pdf(double x, double v, double a, double z, double err, unsigned int logp=0):
    """Compute the likelihood of the drift diffusion model f(t|v,a,z) using the method
    and implementation of Navarro & Fuss, 2009.
    """
    if x <= 0:
        if logp == 0:
            return 0
        else:
            return -np.Inf
        
    cdef double tt = x/(pow(a,2)) # use normalized time
    cdef w = z
    cdef double p  = ftt_01w(tt, w, err) #get f(t|0,1,w)
  
    # convert to f(t|v,a,w)
    if logp == 0:
        return p*exp(-v*a*w -(pow(v,2))*x/2.)/(pow(a,2))
    else:
        return log(p) + (-v*a*w -(pow(v,2))*x/2.) - 2*log(a)

cpdef double pdf_V(double x, double v, double V, double a, double z, double err, unsigned int logp=0):
    """Compute the likelihood of the drift diffusion model f(t|v,a,z,V) using the method
    and implementation of Navarro & Fuss, 2009.
    """
    if x <= 0:
        if logp == 0:
            return 0
        else:
            return -np.Inf
        
    cdef double tt = x/(pow(a,2)) # use normalized time
    cdef double p  = ftt_01w(tt, z, err) #get f(t|0,1,w)
  
    # convert to f(t|v,a,w)
    if logp == 0:
        return p*exp(((a*z*V)**2 - 2*a*v*z - (v**2)*x)/(2*(V**2)*x+2))/sqrt((V**2)*x+1)
    else:
        return log(p) - 2*log(a) + ((a*z*V)**2 - 2*a*v*z - (v**2)*x)/(2*(V**2)*x+2) - log(sqrt((V**2)*x+1))

cpdef double pdf_diff(double x, double v, double a, double z, double err, char diff, unsigned int logp=0):
    """Compute the likelihood of the drift diffusion model using the method
    and implementation of Navarro & Fuss, 2009.
    """
    if x <= 0:
        if logp == 0:
            return 0
        else:
            return -np.Inf
    cdef double t = x
    cdef double tt = x/(pow(a,2)) # use normalized time
    # CHANGE: Relative starting point is expected now.
    cdef double w = z
    #cdef double w = z/a # convert to relative start point

    cdef double kl, ks, p, p2, p3, p_out, p2_out, p3_out
    cdef double PI = 3.1415926535897
    cdef double pi = 3.1415926535897
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
    p2=0
    p3=0
    p_out=0
    p_out2=0
    p_out3=0
    if ks<kl: # if small t is better (i.e., lambda<0)
        K=<int>(ceil(ks)) # round to smallest integer meeting error
        lower = <int>(-floor((K-1)/2.))
        upper = <int>(ceil((K-1)/2.))
        for k from lower <= k <= upper: # loop over k
            # Calculate fixed factor
            p = p + (w + 2*k)*exp(t*(w + 2*k)**2/(2*a**2))    
            if diff == 'z':
                p2 = p2 + t*(w + 2*k)*(2*w + 4*k)*exp(t*(w + 2*k)**2/(2*a**2))/(2*a**2) + exp(t*(w + 2*k)**2/(2*a**2))
            if diff == 'a':
                p2 = p2 + -t*(w + 2*k)**3*exp(t*(w + 2*k)**2/(2*a**2))/a**3
                p3 = p3 + (w + 2*k)*exp(t*(w + 2*k)**2/(2*a**2))

        if diff == 'z':
            p_out = -v*2**(1/2)*exp(-a*v*w - t*v**2/2) * p / (2*pi**(1/2)*a*(t**3/a**6)**(1/2))
            p2_out = 2**(1/2)*exp(-a*v*w - t*v**2/2)* p2/ (2*pi**(1/2)*a**2*(t**3/a**6)**(1/2))
            p3_out = p/(2*pi**(1/2)*a**2*(t**3/a**6)**(1/2))
        elif diff == 'a':
            p_out = 2**(1/2)*exp(-a*v*w - t*v**2/2) * p /(2*pi**(1/2)*a**3*(t**3/a**6)**(1/2)) + 2**(1/2)*exp(-a*v*w - t*v**2/2)* p2 /(2*pi**(1/2)*a**2*(t**3/a**6)**(1/2))
            p3_out = - v*w*2**(1/2)*exp(-a*v*w - t*v**2/2)*p3/(2*pi**(1/2)*a**2*(t**3/a**6)**(1/2))
        elif diff == 'v':
            p_out = 2**(1/2)*(-a*w - t*v)*exp(-a*v*w - t*v**2/2) * p /(2*pi**(1/2)*a**2*(t**3/a**6)**(1/2))

    else: # if large t is better...
        K=<int>(ceil(kl)) # round to smallest integer meeting error
        for k from 1 <= k <= K:
            # Diff sum terms
            if diff == 'z':
                p = p + pi*k**2*cos(pi*k*w)*exp(-t*pi**2*k**2/(2*a**2))
                p2 = p2 + k*exp(-t*pi**2*k**2/(2*a**2))*sin(pi*k*w)
            elif diff == 'a':
                p = p + t*pi**2*k**3*exp(-t*pi**2*k**2/(2*a**2))*sin(pi*k*w)/a**3
                p2 = p2 + k*exp(-t*pi**2*k**2/(2*a**2))*sin(pi*k*w)
            elif diff == 'v':
                p = p + k*exp(-t*pi**2*k**2/(2*a**2))*sin(pi*k*w)
        
        if diff == 'z':
            p_out = pi*exp(-a*v*w - t*v**2/2) * p /a**2
            p2_out = - pi*v*exp(-a*v*w - t*v**2/2)* p2 /a
        elif diff == 'a':
            p_out = pi*exp(-a*v*w - t*v**2/2)* p /a**2
            p2_out = - 2*pi*exp(-a*v*w - t*v**2/2)* p2 /a**3
            p3_out = - pi*v*w*exp(-a*v*w - t*v**2/2)* p2 / a**2
        elif diff == 'v':
            p_out = pi*(-a*w - t*v)*exp(-a*v*w - t*v**2/2)* p /a**2
            
    return p_out + p2_out + p3_out


cpdef double pdf_sign(double x, double v, double a, double z, double t, double err, int logp=0):
    """Wiener likelihood function for two response types. Lower bound
    responses have negative t, upper boundary response have positive t"""
    if z<0 or z>1 or a<0:
        if logp==1:
            return -np.Inf
        else:
            return 0

    if x<0:
        # Lower boundary
        return pdf(fabs(x)-t, v, a, z, err, logp)
    else:
        # Upper boundary, flip v and z
        return pdf(x-t, -v, a, 1.-z, err, logp)

cpdef double pdf_V_sign(double x, double v, double V, double a, double z, double t, double err, int logp=0):
    """Wiener likelihood function for two response types. Lower bound
    responses have negative t, upper boundary response have positive t"""
    if z<0 or z>1 or a<0:
        if logp==1:
            return -np.Inf
        else:
            return 0

    if x<0:
        # Lower boundary
        return pdf_V(fabs(x)-t, v, V, a, z, err, logp)
    else:
        # Upper boundary, flip v and z
        return pdf_V(x-t, -v, V, a, 1.-z, err, logp)

#cpdef double simpson(func f, double a, double b, int n):
#    """f=name of function, a=initial value, b=end value, n=number of double intervals of size 2h"""
#    assert(n&1==0, "n has to be an even number")
# 
#    cdef double h = (b - a) / n
#    cdef double S = f(a)
#    cdef double x
#    cdef int i
#    
#    for i  from 1 <= i < n:        
#        x = a + h * i
#        if i&1: #check if i is odd
#            S += (4 * f(x))
#        else:
#            S += (2 * f(x))
# 
#    S = S + f(b)
#    return (h * S / 3)
#    
#    
#
#cpdef double full_pdf(double x, double v, double V, double a, double z, double Z, 
#                     double t, double T, double err, int nT= 10, int nZ=10, int logp=0):
#
#    
#
#    if (V==0):
#        if (Z==0):
#            if (T==0):
#                return pdf_sign(x, v, a, z, t, err, logp) #V=0,Z=0,T=0
#            else: #T=0
#                return simpson(pdf_sign(x, v, a, z, t, err, logp), -T/2.,T/2., nT) #V=0,Z=0,T=1
#        else: #Z=1
#            if (T==0):
#                return simpson(pdf_sign(x, v, a, z, t, err, logp), -Z/2.,Z/2., nT) #V=0,Z=1,T=0
#            else: #T=1
#                pass
#                #dbl_simpson #V=0,Z=1,T=1
#    else:
#        if (Z==0):
#            if (T==0):
#                return pdf_V_sign(x, v, V, a, z, Z, t, T, err, logp) #V=1,Z=0,T=0
#            else: #T=0
#                return simpson(T, -T/2.,T/2., nT) #V=1,Z=0,T=1
#        else: #Z=1
#            if (T==0):
#                return simpson(T, -Z/2.,Z/2., nT) #V=1,Z=1,T=0
#            else: #T=1
#                pass
#                #dbl_simpson #V=1,Z=1,T=1    
#    
    
@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def pdf_array(np.ndarray[DTYPE_t, ndim=1] x, double v, double a, double z, double t, double err, int logp=0):
    cdef unsigned int size = x.shape[0]
    cdef unsigned int i
    cdef np.ndarray[DTYPE_t, ndim=1] y = np.empty(size, dtype=DTYPE)
    for i from 0 <= i < size:
        y[i] = pdf_sign(x[i], v, a, z, t, err, logp)
    return y

@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def pdf_array_multi(np.ndarray[DTYPE_t, ndim=1] x, v, a, z, t, double err, int logp=0, multi=None):
    cdef unsigned int size = x.shape[0]
    cdef unsigned int i
    cdef np.ndarray[DTYPE_t, ndim=1] y = np.empty(size, dtype=DTYPE)

    if multi is None:
        return pdf_array(x, v=v, a=a, z=z, t=t, err=err, logp=logp)
    else:
        params = {'v':v, 'z':z, 't':t, 'a':a}
        params_iter = copy(params)
        for i from 0 <= i < size:
            for param in multi:
                params_iter[param] = params[param][i]
                
            y[i] = pdf_sign(x[i], params_iter['v'], params_iter['a'], params_iter['z'], params_iter['t'], err, logp)

        return y


@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def wiener_like_full_mc_multi_thresh(np.ndarray[DTYPE_t, ndim=1] x, double v, double V, double z, double Z, t, double T, np.ndarray[DTYPE_t, ndim=1] a, double err=.0001, int logp=0, unsigned int reps=10):
    cdef unsigned int num_resps = x.shape[0]
    cdef unsigned int rep, i

    if logp == 1:
        zero_prob = -np.Inf
    else:
        zero_prob = 0
        
    # Create samples
    cdef np.ndarray[DTYPE_t, ndim=1] t_samples = np.random.uniform(size=reps, low=t-T/2., high=t+T/2.)
    cdef np.ndarray[DTYPE_t, ndim=1] z_samples = np.random.uniform(size=reps, low=z-Z/2., high=z+Z/2.)
    cdef np.ndarray[DTYPE_t, ndim=1] v_samples
    
    if V == 0.:
        v_samples = np.repeat(v, reps)
    else:
        v_samples = np.random.normal(size=reps, loc=v, scale=V)
        
    cdef np.ndarray[DTYPE_t, ndim=2] probs = np.empty((reps,num_resps), dtype=DTYPE)

    for rep from 0 <= rep < reps:
        for i from 0 <= i < num_resps:
            if (fabs(x[i])-t_samples[rep]) < 0:
                probs[rep,i] = zero_prob
            elif a[i] <= z_samples[rep]:
                probs[rep,i] = zero_prob
            else:
                probs[rep,i] = pdf_sign(x[i], v_samples[rep], a[i], z_samples[rep], t_samples[rep], err, logp)

    return np.mean(probs, axis=0)


@cython.boundscheck(False) # turn of bounds-checking for entire function
def wiener_like_full_mc(np.ndarray[DTYPE_t, ndim=1] x, double v, double V, double z, double Z, double t, double T, double a, double err=.0001, int logp=0, unsigned int reps=10):
    cdef unsigned int num_resps = x.shape[0]
    cdef unsigned int rep, i
    
    cdef unsigned int zero_prob = 0
        
    # Create samples
    cdef np.ndarray[DTYPE_t, ndim=1] t_samples = np.random.uniform(size=reps, low=t-T/2., high=t+T/2.)
    cdef np.ndarray[DTYPE_t, ndim=1] z_samples = np.random.uniform(size=reps, low=z-Z/2., high=z+Z/2.)
    # np.random.normal does not work for scale=0, create special case.
    cdef np.ndarray[DTYPE_t, ndim=1] v_samples
    if V == 0.:
        v_samples = np.repeat(v, reps)
    else:
        v_samples = np.random.normal(size=reps, loc=v, scale=V)

    cdef np.ndarray[DTYPE_t, ndim=1] probs = np.zeros(num_resps, dtype=DTYPE)

    # Loop through RTs and reps and add up the resulting probabilities
    for i from 0 <= i < num_resps:
        for rep from 0 <= rep < reps:           
            if (fabs(x[i])-t_samples[rep]) < 0:
                probs[i] += zero_prob
            elif a <= z_samples[rep]:
                probs[i] += zero_prob
            else:
                probs[i] += pdf_sign(x[i], v_samples[rep], a, z_samples[rep], t_samples[rep], err, 0)

    if logp==0:
        return (probs/reps)
    else:
        return np.log(probs/reps)