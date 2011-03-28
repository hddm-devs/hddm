import theano
import theano.tensor as T
import numpy as np

def wfpt(x, a, v, z, err):
    #p, k = T.dvectors(['p', 'k'])
    p, k = T.dscalars(['k', 'p'])

    tt = x/a**2

    kl = T.switch(np.pi*tt*err<1, # if
                  T.max(T.stack(T.sqrt(-2*T.log(np.pi*tt*err)/(np.pi**2*tt)), 1/(np.pi*T.sqrt(tt))), axis=0), # then
                  1/(np.pi*T.sqrt(tt))) # else

    #kl_f = theano.function([x, a, v, z, err], kl)

    #rnd = np.random.rand(50)

    #print kl_f(1, 2, 1, .5, .0001)

    ks = T.switch(2*T.sqrt(2*np.pi*tt)*err<1, # if error threshold is set low enough
                  T.max(T.stack(2+T.sqrt(-2*tt*T.log(2*T.sqrt(2*np.pi*tt)*err)), # bound
                                T.sqrt(tt)+1), axis=0), # ensure boundary conditions are met
                  2) # if error threshold was set too high -> minimal kappa for that case

    #ks_f = theano.function([x, a, v, z, err], ks)
    
    Ks = T.ceil(ks)
    Kl = T.ceil(kl)

    #print ks_f(1, 2, 1, .5, .0001)

    ps = lambda k, z, tt: T.sum( (z+2*k)*T.exp(-((z+2*k)**2)/2/tt))
    pl = lambda k, z, tt: T.sum(k*T.exp(-(k**2)*(np.pi**2)*tt/2)*T.sin(k*np.pi*z))
    
    ps_range = T.arange(-T.floor((Ks-1)/2), T.ceil((Ks-1)/2.), 1)
    pl_range = T.arange(1, Kl, 1)

    ps_terms, tmp = theano.map(ps, ps_range, non_sequences=(z, tt))
    pl_terms, tmp = theano.map(pl, pl_range, non_sequences=(z, tt))

    ps_out = ps_terms.sum() * np.pi
    pl_out = pl_terms.sum() / T.sqrt(2*np.pi*T.pow(tt,3))

    p_out = T.switch(ks < kl, ps_out, pl_out)

    out = p_out*T.exp(-v*a*z -(T.pow(v,2))*x/2.)/(T.pow(a,2))
    
    #logp_func = theano.function(inputs=[x_input, a_input, v_input, z_input, err_input], outputs=logp)

    #like_func = theano.function([x, a, v, z, err], like)
    
    print theano.pprint(out)
    return out

x_input = T.dvector('x_input')
x, a_input, v_input, z_input, err_input = T.dscalars(['x', 'a', 'v', 'z', 'err'])

p, tmp = theano.map(fn=wfpt, sequences=x_input, non_sequences=(a_input, v_input, z_input, err_input))
logp_sum = T.log(p).sum()
p_func = theano.function(inputs=[x_input, a_input, v_input, z_input, err_input], outputs=p)
logp_sum_func = theano.function(inputs=[x_input, a_input, v_input, z_input, err_input], outputs=logp_sum)

print theano.pp(logp)
logp_grad = T.grad(logp_sum, [x_input, a_input, v_input, z_input, err_input])
logp_grad_func = theano.function(inputs=[x_input, a_input, v_input, z_input, err_input], outputs=logp_grad_func)