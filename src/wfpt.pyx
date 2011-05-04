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
    
    f = pdf_sign(rt, v_switch, a, x, t+t_switch, 1e-4) * drift_dens(x, t_switch, v, a, z*a
    #f = pdf_sign(rt, v, a, x, t, 1e-4) * (gsl_ran_gaussian_pdf(x, sqrt(t_switch)) + (t_switch * v + (z*a)))
    return f



cdef inline double drift_dens_term(double x, double t, double v, double a, double z, int n):
    # Ratcliff 1980 Equation 12
    return 2/a * sin(n*PI*z/a) * sin(n*PI*x/a) * exp(-.5*(v**2 + (n**2*PIs)/a**2)*t)

cpdef inline double drift_dens(double x, double t, double v, double a, double z):
    cdef int N=40
    cdef int i
    cdef double terms[40]
    cdef double sum_accel, err
    cdef double summed = 0
    cdef gsl_sum_levin_u_workspace * w = gsl_sum_levin_u_alloc(N)
    
    for i from 1 <= i <= N:
        terms[i-1] = drift_dens_term(x, t, v, a, z, i)
        summed += terms[i-1]

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
    params[0] = rt
    params[1] = v
    params[2] = v_switch
    params[3] = a
    params[4] = z
    params[5] = t
    params[6] = t_switch

    F.function = &wfpt_gsl
    F.params = params

    gsl_integration_qag(&F, 0, a, 0, 1e-7, 1000, GSL_INTEG_GAUSS15, W, &result, &error)
    gsl_integration_workspace_free(W)

    return result

cpdef switch_pdf(DTYPE_t rt, int instruct, double v, double v_switch, double a, double z, double t, double t_switch, double err):
    cdef double p

    if instruct == 0: 
        # Prosaccade trial
        p = pdf_sign(rt, v, a, z, t, err)
    else:
        # Antisaccade trial
        if fabs(rt) =< t+t_switch:
            # Pre switch is not yet online
            p = pdf_sign(rt, v, a, z, t, err)
        else:
            # Post switch
            p = pdf_Z_norm_sign(rt, v, v_switch, a, z, t, t_switch, err)
            
    return p

@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def wiener_like_switch(np.ndarray[DTYPE_t, ndim=1] rt, np.ndarray[int, ndim=1] instruct, double v, double v_switch, double a, double z, double t, double t_switch, double err):
    cdef Py_ssize_t size = rt.shape[0]
    cdef Py_ssize_t i
    cdef double p
    cdef double sum_logp = 0

    for i from 0 <= i < size:
        p = switch_pdf(rt[i], instruct[i], v, v_switch, a, z, t, t_switch, err)
        if p == 0:
            return -infinity
        sum_logp += log(p)

    return sum_logp


@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def wiener_like_simple(np.ndarray[DTYPE_t, ndim=1] x, double v, double a, double z, double t, double err):
    cdef Py_ssize_t i
    cdef double p
    cdef sum_logp = 0
    for i from 0 <= i < x.shape[0]:
        p = pdf_sign(x[i], v, a, z, t, err)
        # If one probability = 0, the log sum will be -Inf
        if p == 0:
            return -infinity
        sum_logp += log(p)
        
    return sum_logp

cdef inline double prob_boundary(double x, double v, double a, double z, double t, double err):
    """Probability of hitting upper boundary."""
    p = (exp(-2*a*z*v) - 1) / (exp(-2*a*v) - 1)
    if x > 0:
        return p
    else:
        return 1-p

@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def wiener_like_simple_contaminant(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[bint, ndim=1] cont_x, np.ndarray[bint, ndim=1] cont_y, double v, double a, double z, double t, double t_min, double t_max, double err):
    """Wiener likelihood function where RTs could come from a
    separate, uniform contaminant distribution.

    Reference: Lee, Vandekerckhove, Navarro, & Tuernlinckx (2007)
    """
    cdef Py_ssize_t i
    cdef double p
    cdef sum_logp = 0
    for i from 0 <= i < x.shape[0]:
        if cont_x[i] == 1:
            p = pdf_sign(x[i], v, a, z, t, err)
        elif cont_y[i] == 0:
            p = prob_boundary(x[i], v, a, z, t, err) * 1./(t_max-t_min)
        else:
            p = .5 * 1./(t_max-t_min)
        #print p, x[i], v, a, z, t, err, t_max, t_min, cont_x[i], cont_y[i]
        # If one probability = 0, the log sum will be -Inf
        if p == 0:
            return -infinity

        sum_logp += log(p)
        
    return sum_logp

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


@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def pdf_array_multi(np.ndarray[DTYPE_t, ndim=1] x, v, a, z, t, double err, bint logp=0, multi=None):
    cdef unsigned int size = x.shape[0]
    cdef unsigned int i
    cdef np.ndarray[DTYPE_t, ndim=1] y = np.empty(size, dtype=DTYPE)
    cdef double prob

    if multi is None:
        return pdf_array(x, v=v, a=a, z=z, t=t, err=err, logp=logp)
    else:
        params = {'v':v, 'z':z, 't':t, 'a':a}
        params_iter = copy(params)
        for i from 0 <= i < size:
            for param in multi:
                params_iter[param] = params[param][i]
                
            prob = pdf_sign(x[i], params_iter['v'], params_iter['a'], params_iter['z'], params_iter['t'], err)
            if logp==1:
                y[i] = log(prob)
            else:
                y[i] = prob
                
        return y

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
