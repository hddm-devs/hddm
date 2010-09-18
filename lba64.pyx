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
cimport numpy as cnp

import scipy as sp
import scipy.stats
from scipy.stats.distributions import norm

cimport cython

# Define data type
DTYPE = np.double
ctypedef cnp.double_t DTYPE_t
cdef double PI = 3.1415926535897

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
    double erf(double)
    

@cython.boundscheck(False) # turn of bounds-checking for entire function
def fptcdf(cnp.ndarray[DTYPE_t, ndim=1] t, double z, double a, double driftrate, double sddrift):
    cdef cnp.ndarray[DTYPE_t, ndim=1] zs=t*sddrift
    cdef cnp.ndarray[DTYPE_t, ndim=1] zu=t*driftrate
    cdef cnp.ndarray[DTYPE_t, ndim=1] aminuszu=a-zu
    cdef cnp.ndarray[DTYPE_t, ndim=1] xx=aminuszu-z
    cdef cnp.ndarray[DTYPE_t, ndim=1] azu=aminuszu/zs
    cdef cnp.ndarray[DTYPE_t, ndim=1] azumax=xx/zs
    cdef cnp.ndarray[DTYPE_t, ndim=1] tmp1=zs*(norm.pdf(azumax)-norm.pdf(azu))
    cdef cnp.ndarray[DTYPE_t, ndim=1] tmp2=xx*norm.cdf(azumax)-aminuszu*norm.cdf(azu)
    return 1+(tmp1+tmp2)/z

@cython.boundscheck(False) # turn of bounds-checking for entire function
def fptpdf(cnp.ndarray[DTYPE_t, ndim=1] t, double z, double a, double driftrate, double sddrift):
    cdef cnp.ndarray[DTYPE_t, ndim=1] zs=t*sddrift
    cdef cnp.ndarray[DTYPE_t, ndim=1] zu=t*driftrate
    cdef cnp.ndarray[DTYPE_t, ndim=1] aminuszu=a-zu
    cdef cnp.ndarray[DTYPE_t, ndim=1] azu=aminuszu/zs
    cdef cnp.ndarray[DTYPE_t, ndim=1] azumax=(aminuszu-z)/zs
    
    return (driftrate*(norm.cdf(azu)-norm.cdf(azumax)) + sddrift*(norm.pdf(azumax)-norm.pdf(azu)))/z

def fptpdf2(cnp.ndarray[DTYPE_t, ndim=1] t, double z, double a, double driftrate, double sddrift):
    return driftrate*(norm.cdf((a-(t*driftrate))/(t*sddrift))-norm.cdf(a-(t*driftrate)-z/(t*sddrift))) + \
           sddrift*(norm.pdf(a-(t*driftrate)-z/(t*sddrift))-norm.pdf(a-((t*driftrate)/(t*sddrift))))/z

@cython.boundscheck(False) # turn of bounds-checking for entire function
def lba(cnp.ndarray[DTYPE_t, ndim=1] t, double z, double a, cnp.ndarray[DTYPE_t, ndim=1] drift, double sv):
    # Generates defective PDF for responses on node #1.
    cdef int i=0
    cdef int N=drift.shape[0] # Number of responses.
    #cdef np.ndarray[DTYPE_t, ndim=1] G = np.empty_like(t)
    #cdef np.ndarray[DTYPE_t, ndim=2] tmp = np.empty((t.shape[0], N-1), dtype=DTYPE)
    cdef cnp.ndarray[DTYPE_t, ndim=1] G=1-fptcdf(t=t,z=z,a=a,driftrate=drift[1],sddrift=sv)
    #if (N>2):
        #for i from 1 <= i <= N:
            #tmp[:,i]=fptcdf(t=t,z=z,a=a,driftrate=drift[i],sddrift=sv)
        #G=apply(1-tmp,1,prod)
        #np.prod(1-tmp, axis=0, out=G)
    #    print ''
    #else:
    #    cdef np.ndarray[DTYPE_t, ndim=1] G=1-fptcdf(t=t,z=z,a=a,driftrate=drift[1],sddrift=sv)
        
    return G*fptpdf(t=t,z=z,a=a,driftrate=drift[0],sddrift=sv)

# Second approach
cdef inline double norm_pdf(double x): return 0.3989422804014327 * exp(-(x**2/2))
cdef inline double norm_cdf(double x): return .5 * (1 + erf((x)/1.4142135623730951))

cdef DTYPE_t fptcdf_single(DTYPE_t t, double z, double a, double driftrate, double sddrift):
    cdef double zs=t*sddrift
    cdef double zu=t*driftrate
    cdef double aminuszu=a-zu
    cdef double xx=aminuszu-z
    cdef double azu=aminuszu/zs
    cdef double azumax=xx/zs
    cdef double tmp1 = zs * (norm_pdf(azumax) - norm_pdf(azu))
    cdef double tmp2 = (xx * norm_cdf(azumax)) - (aminuszu * norm_cdf(azu))
    return 1+(tmp1+tmp2)/z

cdef DTYPE_t fptpdf_single(DTYPE_t t, double z, double a, double driftrate, double sddrift):
    cdef double zs=t*sddrift
    cdef double zu=t*driftrate
    cdef double aminuszu=a-zu
    cdef double azu=aminuszu/zs
    cdef double azumax=(aminuszu-z)/zs
    
    return (driftrate*(norm_cdf(azu)-norm_cdf(azumax)) + sddrift*(norm_pdf(azumax)-norm_pdf(azu)))/z

@cython.boundscheck(False) # turn of bounds-checking for entire function
def lba_single(cnp.ndarray[DTYPE_t, ndim=1] t, double z, double a, cnp.ndarray[DTYPE_t, ndim=1] drift, double sv, cnp.ndarray[DTYPE_t, ndim=1] out=None):
    # Generates defective PDF for responses on node #1.
    cdef unsigned int i=0
    cdef unsigned int max = <unsigned int> t.shape[0]
    if out is None:
        out = np.empty(max, dtype=DTYPE)
    elif out.shape[0] != t.shape[0]:
        raise ValueError('out array must have the same shape as input array t')
        

    for i from 0 <= i < max:
         out[i] = (1-fptcdf_single(t[i], z, a, drift[<unsigned int> 1], sv)) * \
                fptpdf_single(t[i], z, a, drift[<unsigned int> 0], sv)

    return out

@cython.boundscheck(False) # turn of bounds-checking for entire function
def lba_like(cnp.ndarray[DTYPE_t, ndim=1] value, cnp.ndarray[DTYPE_t, ndim=1] resps, double z, double a, double ter, cnp.ndarray[DTYPE_t, ndim=1] drift, double sv, unsigned int logp=0):
    # Rescale parameters so as to fit in the same range as the DDM
    cdef cnp.ndarray[DTYPE_t, ndim=1] rt = (np.abs(value) - ter)
    cdef unsigned int nresp = value.shape[0]
    cdef unsigned int i
    cdef cnp.ndarray[DTYPE_t, ndim=1] probs = np.empty(nresp, dtype=DTYPE)
    
    if a <= z:
        #print "Starting point larger than threshold!"
        return -np.Inf

    #print "z: %f, a: %f, ter: %i, v: %f, sv: %f" % (z, a, ter, v[0], sv)
    for i from 0 <= i < nresp:
        if resps[i] == 1:
            probs[i] = (1-fptcdf_single(rt[i], z, a, drift[1], sv)) * \
                       fptpdf_single(rt[i], z, a, drift[0], sv)
        else:
            probs[i] = (1-fptcdf_single(rt[i], z, a, drift[0], sv)) * \
                       fptpdf_single(rt[i], z, a, drift[1], sv)

    if logp == 1:
        return np.sum(np.log(probs))
    else:
        return probs
