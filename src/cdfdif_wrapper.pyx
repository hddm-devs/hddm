
cimport numpy as np
import numpy as np

cdef extern from "cdfdif.h":
    double cdfdif(double t, int x, double *par, double *prob)

cdef extern from "math.h":
    double fabs(double)

cdef inline double add_outlier_cdf(double y, double x, double p_outlier, double w_outlier):
    return y * (1 - p_outlier) + (x + (1. / (2 * w_outlier))) * w_outlier * p_outlier

cdef inline bint p_outlier_in_range(double p_outlier): return (p_outlier >= 0) & (p_outlier <= 1)

def dmat_cdf_array(np.ndarray[double, ndim=1] x, double v, double sv,
                 double a, double z, double sz, double t, double st, double p_outlier, double w_outlier):

    #check arguments
    if p_outlier > 0:
        assert np.max(np.abs(x)) < (1./(2*w_outlier)), ValueError('1. / (2*w_outlier) must be smaller than RT')

    if (sv < 0) or (a <=0 ) or (z < 0) or (z > 1) or (sz < 0) or (sz > 1) or (z+sz/2.>1) or \
    (z-sz/2.<0) or (t-st/2.<0) or (t<0) or (st < 0) or not p_outlier_in_range(p_outlier):
        raise ValueError("at least one of the parameters is out of the support")


    cdef Py_ssize_t size = x.shape[0]
    cdef np.ndarray[double, ndim=1] y = np.empty(size, dtype=np.double)
    cdef int boundary
    cdef double p_boundary
    cdef double params[7]
    cdef double epsi = 1e-10

    #transform parameters
    params[0] = a/10.
    params[1] = t
    params[2] = sv/10. + epsi
    params[3] = z*(a/10.)
    params[4] = sz*(a/10.) + epsi
    params[5] = st + epsi
    params[6] = v/10.


    for i in range(size):
        boundary = (int) (x[i] > 0)
        y[i] = cdfdif(fabs(x[i]), boundary, params, &p_boundary)
        y[i] = (1 - p_boundary) + np.sign(x[i])*y[i]

        #add p_outlier probability
        y[i] = add_outlier_cdf(y[i], x[i], p_outlier, w_outlier)

    return y
