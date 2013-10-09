#cython: embedsignature=True
#cython: cdivision=True
#cython: wraparound=False
#cython: boundscheck=False
#
# Copyleft Thomas Wiecki (thomas_wiecki[at]brown.edu), 2010
# GPLv3

import numpy as np
cimport numpy as np

# Define data type
DTYPE = np.double
ctypedef np.double_t DTYPE_t

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

cdef inline bint p_outlier_in_range(double p_outlier):
    if (p_outlier >= 0) and (p_outlier <= 1):
        return 1
    else:
        return 0

def lba_like(np.ndarray[DTYPE_t, ndim=1] value, double z, double a, double ter, double sv, double v0, double v1, bint normalize_v=False, double p_outlier=0, double w_outlier=0):
    cdef Py_ssize_t size = value.shape[0]
    cdef Py_ssize_t i = 0
    cdef double sum_logp = 0
    cdef double p = 0
    cdef double wp_outlier = w_outlier * p_outlier
    cdef double rt

    assert sv >= 0, "sv must be larger than 0"

    if not p_outlier_in_range(p_outlier):
        return -np.inf

    if normalize_v == 1:
        v1 = 1 - v0

    if a <= z:
        #print "Starting point larger than threshold!"
        return -np.Inf

    #print "z: %f, a: %f, ter: %i, v: %f, sv: %f" % (z, a, ter, v[0], sv)
    for i in range(size):
        rt = fabs(value[i]) - ter
        if rt < 0:
            p = 0
        elif value[i] > 0:
            p = (1-fptcdf_single(rt, z, a, v1, sv)) * \
                fptpdf_single(rt, z, a, v0, sv)
        elif value[i] < 0:
            p = (1-fptcdf_single(rt, z, a, v0, sv)) * \
                fptpdf_single(rt, z, a, v1, sv)

        p = p * (1 - p_outlier) + wp_outlier
        if p == 0:
            return -np.inf

        sum_logp += log(p)

    return sum_logp
