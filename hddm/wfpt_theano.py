import theano
import theano.tensor as T
import numpy as np

def wfpt():
    x = T.dvector('x')
    p, k = T.dvectors(['p', 'k'])
    a, v, z, err = T.dscalars(['a','v','z','err'])

    tt = x/a**2

    kl = T.switch(np.pi*tt*err<1, # if
                  T.max(T.stack(T.sqrt(-2*T.log(np.pi*tt*err)/(np.pi**2*tt)), 1/(np.pi*T.sqrt(tt))), axis=0), # then
                  1/(np.pi*T.sqrt(tt))) # else

    kl_f = theano.function([x, a, v, z, err], kl)

    rnd = np.random.rand(50)

    print kl_f(rnd, 2, 1, .5, .0001)

    ks = T.switch(2*T.sqrt(2*np.pi*tt)*err<1, # if error threshold is set low enough
                  T.max(T.stack(2+T.sqrt(-2*tt*T.log(2*T.sqrt(2*np.pi*tt)*err)), # bound
                                T.sqrt(tt)+1), axis=0), # ensure boundary conditions are met
                  2) # if error threshold was set too high -> minimal kappa for that case

    ks_f = theano.function([x, a, v, z, err], ks)
    
    Ks = T.ceil(ks)
    Kl = T.ceil(kl)

    print ks_f(rnd, 2, 1, .5, .0001)
    
    ps = theano.function([k, z, tt], (z+2*k)*T.exp(-((z+2*k)**2)/2/tt))
    pl = theano.function([k, z, tt], k*T.exp(-(k**2)*(np.pi**2)*tt/2)*T.sin(k*np.pi*z))
    
    print (-T.floor((Ks-1)/2)).ndim
    ps_range = T.arange(-T.floor((Ks-1)/2), T.ceil((Ks-1)/2.), 1)
    pl_range = T.arange(1, Kl, 1)

    ps_terms = theano.map(ps, ps_range, non_sequences=(z, tt))
    pl_terms = theano.map(pl, pl_range, non_sequences=(z, tt))

    ps_out = ps_terms * np.pi
    pl_out = pl_terms / T.sqrt(2*PI*T.pow(tt,3))

    p_out = T.switch(ks < kl, ps_out, pl_out)

    like = p_out*T.exp(-v*a*w -(T.pow(v,2))*x/2.)/(T.pow(a,2))

    like_func = theano.function([x, a, v, z, err], like)

    print like_func(rnd, 2, 1, .5, .0001)

    return