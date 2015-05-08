#cython: embedsignature=True
#cython: cdivision=True
#cython: wraparound=False
#cython: boundscheck=False

cimport cython

#include "integrate.pxi"

#from libc.math cimport tan, sin, cos, log, exp, sqrt, fmax, pow, ceil, floor, fabs, M_PI

cdef extern from "math.h" nogil:
    double sin(double)
    double cos(double)
    double log(double)
    double exp(double)
    double sqrt(double)
#    double fmax(double, double)
    double pow(double, double)
    double ceil(double)
    double floor(double)
    double fabs(double)
    double M_PI

cdef extern from "<algorithm>" namespace "std" nogil:
    T max[T](T a, T b)

cdef double ftt_01w(double tt, double w, double err) nogil:
    """Compute f(t|0,1,w) for the likelihood of the drift diffusion model using the method
    and implementation of Navarro & Fuss, 2009.
    """
    cdef double kl, ks, p
    cdef int k, K, lower, upper

    # calculate number of terms needed for large t
    if M_PI*tt*err<1: # if error threshold is set low enough
        kl=sqrt(-2*log(M_PI*tt*err)/(M_PI**2*tt)) # bound
        kl=max(kl,1./(M_PI*sqrt(tt))) # ensure boundary conditions met
    else: # if error threshold set too high
        kl=1./(M_PI*sqrt(tt)) # set to boundary condition

    # calculate number of terms needed for small t
    if 2*sqrt(2*M_PI*tt)*err<1: # if error threshold is set low enough
        ks=2+sqrt(-2*tt*log(2*sqrt(2*M_PI*tt)*err)) # bound
        ks=max(ks,sqrt(tt)+1) # ensure boundary conditions are met
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
        p/=sqrt(2*M_PI*pow(tt,3)) # add con_stant term

    else: # if large t is better...
        K=<int>(ceil(kl)) # round to smallest integer meeting error
        for k from 1 <= k <= K:
            p+=k*exp(-(pow(k,2))*(M_PI**2)*tt/2)*sin(k*M_PI*w) # increment sum
        p*=M_PI # add con_stant term

    return p

cdef inline double prob_ub(double v, double a, double z) nogil:
    """Probability of hitting upper boundary."""
    return (exp(-2*a*z*v) - 1) / (exp(-2*a*v) - 1)

cdef double pdf(double x, double v, double a, double w, double err) nogil:
    """Compute the likelihood of the drift diffusion model f(t|v,a,z) using the method
    and implementation of Navarro & Fuss, 2009.
    """
    if x <= 0:
        return 0

    cdef double tt = x/a**2 # use normalized time
    cdef double p = ftt_01w(tt, w, err) #get f(t|0,1,w)

    # convert to f(t|v,a,w)
    return p*exp(-v*a*w -(pow(v,2))*x/2.)/(pow(a,2))

cdef double pdf_sv(double x, double v, double sv, double a, double z, double err) nogil:
    """Compute the likelihood of the drift diffusion model f(t|v,a,z,sv) using the method
    and implementation of Navarro & Fuss, 2009.
    sv is the std of the drift rate
    """
    if x <= 0:
        return 0

    if sv==0:
        return pdf(x, v, a, z, err)

    cdef double tt = x/(pow(a,2)) # use normalized time
    cdef double p  = ftt_01w(tt, z, err) #get f(t|0,1,w)

    # convert to f(t|v,a,w)
    return exp(log(p) + ((a*z*sv)**2 - 2*a*v*z - (v**2)*x)/(2*(sv**2)*x+2))/sqrt((sv**2)*x+1)/(a**2)

cpdef double full_pdf(double x, double v, double sv, double a, double
                      z, double sz, double t, double st, double err, int
                      n_st=2, int n_sz=2, bint use_adaptive=1, double
                      simps_err=1e-3) nogil:
    """full pdf"""

    # Check if parpameters are valid
    if (z<0) or (z>1) or (a<0) or (t<0) or (st<0) or (sv<0) or (sz<0) or (sz>1) or \
       ((fabs(x)-(t-st/2.))<0) or (z+sz/2.>1) or (z-sz/2.<0) or (t-st/2.<0):
        return 0

    # transform x,v,z if x is upper bound response
    if x > 0:
        v = -v
        z = 1.-z

    x = fabs(x)

    if st<1e-3:
        st = 0
    if sz <1e-3:
        sz = 0

    if (sz==0):
        if (st==0): #sv=0,sz=0,st=0
            return pdf_sv(x - t, v, sv, a, z, err)
        else:      #sv=0,sz=0,st=$
            if use_adaptive>0:
                return adaptiveSimpsons_1D(x,  v, sv, a, z, t, err, z, z, t-st/2., t+st/2., simps_err, n_st)
            else:
                return simpson_1D(x, v, sv, a, z, t, err, z, z, 0, t-st/2., t+st/2., n_st)

    else: #sz=$
        if (st==0): #sv=0,sz=$,st=0
            if use_adaptive:
                return adaptiveSimpsons_1D(x, v, sv, a, z, t, err, z-sz/2., z+sz/2., t, t, simps_err, n_sz)
            else:
                return simpson_1D(x, v, sv, a, z, t, err, z-sz/2., z+sz/2., n_sz, t, t , 0)
        else:      #sv=0,sz=$,st=$
            if use_adaptive:
                return adaptiveSimpsons_2D(x, v, sv, a, z, t, err, z-sz/2., z+sz/2., t-st/2., t+st/2., simps_err, n_sz, n_st)
            else:
                return simpson_2D(x, v, sv, a, z, t, err, z-sz/2., z+sz/2., n_sz, t-st/2., t+st/2., n_st)
