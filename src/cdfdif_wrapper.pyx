
cimport numpy as np
import numpy as np

cdef extern double cdfdif(double t, int x, double *par, double *prob)

cdef extern from "math.h":
    double fabs(double)

def cdfdif_array(np.ndarray[double, ndim=1] x, double v, double sv,
                 double a, double z, double sz, double t, double st):

    cdef Py_ssize_t size = x.shape[0]
    cdef np.ndarray[double, ndim=1] y = np.empty(size, dtype=np.double)
    cdef int boundary
    cdef double p_boundary

    cdef double params[7]

    params[0] = a/10.
    params[1] = t
    params[2] = sv/10.
    params[3] = z*(a/10.)
    params[4] = sz*(a/10.)
    params[5] = st
    params[6] = v/10.

    for i in range(size):
        boundary = (int) (x[i] > 0)
        y[i] = cdfdif(fabs(x[i]), boundary, params, &p_boundary)

    #y = y * (1 - p_outlier) + (w_outlier * p_outlier)
    return y, p_boundary

    #double a = par[0], Ter = par[1], eta = par[2], z = par[3], sZ = par[4],
    #st = par[5], nu = par[6]