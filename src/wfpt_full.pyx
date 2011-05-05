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

include "wfpt.pyx"

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
def wiener_like_full_intrp(np.ndarray[DTYPE_t, ndim=1] x, double v, double V, double a, double z, double Z, double t, 
                           double T, double err, int nT= 10, int nZ=10, bint use_adaptive=1, double simps_err=1e-8):
    cdef Py_ssize_t i
    cdef double p
    cdef sum_logp = 0
    
    for i from 0 <= i < x.shape[0]:
        p = full_pdf(x[i], v, V, a, z, Z, t, T, err, nT, nZ, use_adaptive, simps_err)
        # If one probability = 0, the log sum will be -Inf
        if p == 0:
            return -infinity
        sum_logp += log(p)
        
    return sum_logp


cpdef double simpson_1D(double x, double v, double V, double a, double z, double t, double err, 
                        double lb_z, double ub_z, int nZ, double lb_t, double ub_t, int nT):
    assert ((nZ&1)==0 and (nT&1)==0), "nT and nZ have to be even"
    assert ((ub_t-lb_t)*(ub_z-lb_z)==0 and (nZ*nT)==0), "the function is defined for 1D-integration only"
    
    cdef double ht, hz
    cdef int n = max(nT,nZ)
    if nT==0: #integration over z
        hz = (ub_z-lb_z)/n
        ht = 0
        lb_t = t
        ub_t = t
    else: #integration over t
        hz = 0
        ht = (ub_t-lb_t)/n
        lb_z = z
        ub_z = z

    cdef double S = pdf_V(x - lb_t, v, V, a, lb_z, err)
    cdef double z_tag, t_tag, y
    cdef int i
    
    for i from 1 <= i <= n:
        z_tag = lb_z + hz * i
        t_tag = lb_t + ht * i
        y = pdf_V(x - t_tag, v, V, a, z_tag, err)
        if i&1: #check if i is odd
            S += (4 * y)
        else:
            S += (2 * y)
    S = S - y #the last term should be f(b) and not 2*f(b) so we subtract y
    S = S / ((ub_t-lb_t)+(ub_z-lb_z)) #the right function if pdf_V()/Z or pdf_V()/T

    return ((ht+hz) * S / 3)

cpdef double simpson_2D(double x, double v, double V, double a, double z, double t, double err, double lb_z, double ub_z, int nZ, double lb_t, double ub_t, int nT):
    assert ((nZ&1)==0 and (nT&1)==0), "nT and nZ have to be even"
    assert ((ub_t-lb_t)*(ub_z-lb_z)>0 and (nZ*nT)>0), "the function is defined for 2D-integration only, lb_t: %f, ub_t %f, lb_z %f, ub_z %f, nZ: %d, nT %d" % (lb_t, ub_t, lb_z, ub_z, nZ, nT)

    cdef double ht
    cdef double S
    cdef double t_tag, y
    cdef int i_t

    ht = (ub_t-lb_t)/nT

    S = simpson_1D(x, v, V, a, z, lb_t, err, lb_z, ub_z, nZ, 0, 0, 0)

    for i_t  from 1 <= i_t <= nT:
        t_tag = lb_t + ht * i_t
        y = simpson_1D(x, v, V, a, z, t_tag, err, lb_z, ub_z, nZ, 0, 0, 0)
        if i_t&1: #check if i is odd
            S += (4 * y)
        else:
            S += (2 * y)
    S = S - y #the last term should be f(b) and not 2*f(b) so we subtract y
    S = S/ (ub_t-lb_t)

    return (ht * S / 3)

cpdef double adaptiveSimpsonsAux(double x, double v, double V, double a, double z, double t, double pdf_err,
                                 double lb_z, double ub_z, double lb_t, double ub_t, double ZT, double simps_err,
                                 double S, double f_beg, double f_end, double f_mid, int bottom):
    
    cdef double z_c, z_d, z_e, t_c, t_d, t_e, h
    cdef double fd, fe
    cdef double Sleft, Sright, S2
    #print "in AdaptiveSimpsAux: lb_z: %f, ub_z: %f, lb_t %f, ub_t %f, f_beg: %f, f_end: %f, bottom: %d" % (lb_z, ub_z, lb_t, ub_t, f_beg, f_end, bottom)
    
    if (ub_t-lb_t) == 0: #integration over Z
        h = ub_z - lb_z
        z_c = (ub_z + lb_z)/2.
        z_d = (lb_z + z_c)/2.
        z_e = (z_c  + ub_z)/2.
        t_c = t
        t_d = t
        t_e = t
    
    else: #integration over t
        h = ub_t - lb_t
        t_c = (ub_t + lb_t)/2.
        t_d = (lb_t + t_c)/2.
        t_e = (t_c  + ub_t)/2.
        z_c = z
        z_d = z
        z_e = z
    
    fd = pdf_V(x - t_d, v, V, a, z_d, pdf_err)/ZT
    fe = pdf_V(x - t_e, v, V, a, z_e, pdf_err)/ZT
    
    Sleft = (h/12)*(f_beg + 4*fd + f_mid)
    Sright = (h/12)*(f_mid + 4*fe + f_end)
    S2 = Sleft + Sright                                          
    if (bottom <= 0 or fabs(S2 - S) <= 15*simps_err):
        return S2 + (S2 - S)/15
    return adaptiveSimpsonsAux(x, v, V, a, z, t, pdf_err,
                                 lb_z, z_c, lb_t, t_c, ZT, simps_err/2,
                                 Sleft, f_beg, f_mid, fd, bottom-1) + \
            adaptiveSimpsonsAux(x, v, V, a, z, t, pdf_err,
                                 z_c, ub_z, t_c, ub_t, ZT, simps_err/2,
                                 Sright, f_mid, f_end, fe, bottom-1)
 
cpdef double adaptiveSimpsons_1D(double x, double v, double V, double a, double z, double t, 
                              double pdf_err, double lb_z, double ub_z, double lb_t, double ub_t, 
                              double simps_err, int maxRecursionDepth):

    cdef double h
    
    if (ub_t - lb_t) == 0: #integration over z
        lb_t = t
        ub_t = t
        h = ub_z - lb_z
    else: #integration over t
        h = (ub_t-lb_t)
        lb_z = z
        ub_z = z
    
    cdef double ZT = h
    cdef double c_t = (lb_t + ub_t)/2.
    cdef double c_z = (lb_z + ub_z)/2.

    cdef double f_beg, f_end, f_mid, S
    f_beg = pdf_V(x - lb_t, v, V, a, lb_z, pdf_err)/ZT
    f_end = pdf_V(x - ub_t, v, V, a, ub_z, pdf_err)/ZT
    f_mid = pdf_V(x - c_t, v, V, a, c_z, pdf_err)/ZT
    S = (h/6)*(f_beg + 4*f_mid + f_end)                                 
    cdef double res =  adaptiveSimpsonsAux(x, v, V, a, z, t, pdf_err,
                                 lb_z, ub_z, lb_t, ub_t, ZT, simps_err,           
                                 S, f_beg, f_end, f_mid, maxRecursionDepth)
    return res

cdef double adaptiveSimpsonsAux_2D(double x, double v, double V, double a, double z, double t, double pdf_err, double err_1d,
                                 double lb_z, double ub_z, double lb_t, double ub_t, double T, double err_2d,
                                 double S, double f_beg, double f_end, double f_mid, int maxRecursionDepth_Z, int bottom):

    cdef double fd, fe
    cdef double Sleft, Sright, S2
    #print "in AdaptiveSimpsAux_2D: lb_z: %f, ub_z: %f, lb_t %f, ub_t %f, f_beg: %f, f_end: %f, bottom: %d" % (lb_z, ub_z, lb_t, ub_t, f_beg, f_end, bottom)
    
    cdef double t_c = (ub_t + lb_t)/2.
    cdef double t_d = (lb_t + t_c)/2.
    cdef double t_e = (t_c  + ub_t)/2.
    cdef double h = ub_t - lb_t
    
    fd = adaptiveSimpsons_1D(x, v, V, a, z, t_d, pdf_err, lb_z, ub_z,
                              0, 0, err_1d, maxRecursionDepth_Z)/T
    fe = adaptiveSimpsons_1D(x, v, V, a, z, t_e, pdf_err, lb_z, ub_z,
                              0, 0, err_1d, maxRecursionDepth_Z)/T
    
    Sleft = (h/12)*(f_beg + 4*fd + f_mid)
    Sright = (h/12)*(f_mid + 4*fe + f_end)
    S2 = Sleft + Sright

    if (bottom <= 0 or fabs(S2 - S) <= 15*err_2d):                                     
        return S2 + (S2 - S)/15;
        
    return adaptiveSimpsonsAux_2D(x, v, V, a, z, t, pdf_err, err_1d,
                                 lb_z, ub_z, lb_t, t_c, T, err_2d/2,
                                 Sleft, f_beg, f_mid, fd, maxRecursionDepth_Z, bottom-1) + \
            adaptiveSimpsonsAux_2D(x, v, V, a, z, t, pdf_err, err_1d,
                                 lb_z, ub_z, t_c, ub_t, T, err_2d/2,
                                 Sright, f_mid, f_end, fe, maxRecursionDepth_Z, bottom-1)
                             
                                 
        
cpdef double adaptiveSimpsons_2D(double x, double v, double V, double a, double z, double t,  
                                 double pdf_err, double lb_z, double ub_z, double lb_t, double ub_t, 
                                 double simps_err, int maxRecursionDepth_Z, maxRecursionDepth_T):

    cdef double h = (ub_t-lb_t)
    
    cdef double T = (ub_t - lb_t)
    cdef double c_t = (lb_t + ub_t)/2.
    cdef double c_z = (lb_z + ub_z)/2.
 
    cdef double f_beg, f_end, f_mid, S
    cdef double err_1d = simps_err
    cdef double err_2d = simps_err
    
    f_beg = adaptiveSimpsons_1D(x, v, V, a, z, lb_t, pdf_err, lb_z, ub_z,
                              0, 0, err_1d, maxRecursionDepth_Z)/T

    f_end = adaptiveSimpsons_1D(x, v, V, a, z, ub_t, pdf_err, lb_z, ub_z,
                              0, 0, err_1d, maxRecursionDepth_Z)/T
    f_mid = adaptiveSimpsons_1D(x, v, V, a, z, (lb_t+ub_t)/2, pdf_err, lb_z, ub_z, 
                              0, 0, err_1d, maxRecursionDepth_Z)/T
    S = (h/6)*(f_beg + 4*f_mid + f_end)    
    cdef double res =  adaptiveSimpsonsAux_2D(x, v, V, a, z, t, pdf_err, err_1d,
                                 lb_z, ub_z, lb_t, ub_t, T, err_2d,
                                 S, f_beg, f_end, f_mid, maxRecursionDepth_Z, maxRecursionDepth_T)
    return res

cpdef double full_pdf(double x, double v, double V, double a, double z, double Z, 
                     double t, double T, double err, int nT=4, int nZ=4, bint use_adaptive = 1, double simps_err = 1e-5):
    """pull pdf"""

    # Check if parpameters are valid
    if z<0 or z>1 or a<0 or ((fabs(x)-(t-T/2.))<0) or (z+Z/2.>1) or (z-Z/2.<0) or (t-T/2.<0) or (t<0):
        return 0

    # transform x,v,z if x is upper bound response
    if x > 0:
        v = -v
        z = 1.-z
    
    x = fabs(x)
    
    if T<1e-3:
        T = 0
    if Z <1e-3:
        Z = 0  

    if (Z==0):
        if (T==0): #V=0,Z=0,T=0
            return pdf_V(x - t, v, V, a, z, err)
        else:      #V=0,Z=0,T=$
            if use_adaptive>0:
                return adaptiveSimpsons_1D(x,  v, V, a, z, t, err, z, z, t-T/2., t+T/2., simps_err, nT)
            else:
                return simpson_1D(x, v, V, a, z, t, err, z, z, 0, t-T/2., t+T/2., nT)
            
    else: #Z=$
        if (T==0): #V=0,Z=$,T=0
            if use_adaptive:
                return adaptiveSimpsons_1D(x, v, V, a, z, t, err, z-Z/2., z+Z/2., t, t, simps_err, nZ)
            else:
                return simpson_1D(x, v, V, a, z, t, err, z-Z/2., z+Z/2., nZ, t, t , 0)
        else:      #V=0,Z=$,T=$
            if use_adaptive:
                return adaptiveSimpsons_2D(x, v, V, a, z, t, err, z-Z/2., z+Z/2., t-T/2., t+T/2., simps_err, nZ, nT)
            else:
                return simpson_2D(x, v, V, a, z, t, err, z-Z/2., z+Z/2., nZ, t-T/2., t+T/2., nT)

@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def wiener_like_full(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] v, np.ndarray[DTYPE_t, ndim=1] a, np.ndarray[DTYPE_t, ndim=1] z, np.ndarray[DTYPE_t, ndim=1] t, err):
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef double p
    cdef double sum_logp = 0

    for i from 0 <= i < size:
        p = pdf_sign(x[i], v[i], a[i], z[i], t[i], err)
        # If one probability = 0, the log sum will be -Inf
        if p == 0:
            return -infinity
        sum_logp += log(p)

    return sum_logp

@cython.boundscheck(False) # turn of bounds-checking for entire function
def wiener_like_full_mc(np.ndarray[DTYPE_t, ndim=1] x, double v, double V, double z, double Z, double t, double T, double a, double err=.0001, bint logp=0, unsigned int reps=10):
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
                probs[i] += pdf_sign(x[i], v_samples[rep], a, z_samples[rep], t_samples[rep], err)

    if logp==0:
        return (probs/reps)
    else:
        return np.log(probs/reps)
