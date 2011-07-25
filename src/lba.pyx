#!/usr/bin/python 
#
# Cython version of the Navarro & Fuss, 2009 DDM PDF. Based directly
# on the following code by Navarro & Fuss:
# http://www.psychocmath.logy.adelaide.edu.au/personalpages/staff/danielnavarro/resources/wfpt.m
#
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

@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def lba_like(cnp.ndarray[DTYPE_t, ndim=1] value, double z, double a, double ter, double sv, double v0, double v1, unsigned int logp=0, unsigned int normalize_v=0):
    cdef cnp.ndarray[DTYPE_t, ndim=1] rt = (np.abs(value) - ter)
    cdef unsigned int nresp = <unsigned int> value.shape[0]
    cdef unsigned int i = 0
    cdef cnp.ndarray[DTYPE_t, ndim=1] probs = np.empty_like(value)

    assert sv >= 0, "sv must be larger than 0"
    
    if normalize_v == 1:
        v1 = 1 - v0
        
    if a <= z:
        #print "Starting point larger than threshold!"
        return -np.Inf

    #print "z: %f, a: %f, ter: %i, v: %f, sv: %f" % (z, a, ter, v[0], sv)
    for i from 0 <= i < nresp:
        if value[i] > 0:
            probs[i] = (1-fptcdf_single(rt[i], z, a, v1, sv)) * \
                       fptpdf_single(rt[i], z, a, v0, sv)
        else:
            probs[i] = (1-fptcdf_single(rt[i], z, a, v0, sv)) * \
                       fptpdf_single(rt[i], z, a, v1, sv)

    if logp == 1:
        return np.sum(np.log(probs))
    else:
        return probs


# Backed up if in the future multiple responses are to be supported
@cython.boundscheck(False) # turn of bounds-checking for entire function
def lba_like_old(cnp.ndarray[DTYPE_t, ndim=1] value, cnp.ndarray[DTYPE_t, ndim=1] resps, double z, double a, double ter, cnp.ndarray[DTYPE_t, ndim=1] drift, double sv, unsigned int logp=0):
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
