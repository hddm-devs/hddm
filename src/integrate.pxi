import numpy as np
cimport numpy as np

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
