include "integrate.pxi"

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
    double M_PI


cpdef double ftt_01w(double tt, double w, double err):
    """Compute f(t|0,1,w) for the likelihood of the drift diffusion model using the method
    and implementation of Navarro & Fuss, 2009.
    """
    cdef double kl, ks, p
    cdef int k, K, lower, upper

    # calculate number of terms needed for large t
    if M_PI*tt*err<1: # if error threshold is set low enough
        kl=sqrt(-2*log(M_PI*tt*err)/(M_PI**2*tt)) # bound
        kl=fmax(kl,1./(M_PI*sqrt(tt))) # ensure boundary conditions met
    else: # if error threshold set too high
        kl=1./(M_PI*sqrt(tt)) # set to boundary condition

    # calculate number of terms needed for small t
    if 2*sqrt(2*M_PI*tt)*err<1: # if error threshold is set low enough
        ks=2+sqrt(-2*tt*log(2*sqrt(2*M_PI*tt)*err)) # bound
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
        p/=sqrt(2*M_PI*pow(tt,3)) # add constant term
  
    else: # if large t is better...
        K=<int>(ceil(kl)) # round to smallest integer meeting error
        for k from 1 <= k <= K:
            p+=k*exp(-(pow(k,2))*(M_PI**2)*tt/2)*sin(k*M_PI*w) # increment sum
        p*=M_PI # add constant term

    return p

cdef inline double prob_ub(double v, double a, double z):
    """Probability of hitting upper boundary."""
    return (exp(-2*a*z*v) - 1) / (exp(-2*a*v) - 1)


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
    if z<0 or z>1 or a<=0:
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


cpdef double full_pdf(double x, double v, double V, double a, double z, double Z, 
                     double t, double T, double err, int nT=2, int nZ=2, bint use_adaptive = 1, double simps_err = 1e-3):
    """full pdf"""

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
