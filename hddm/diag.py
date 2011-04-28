from __future__ import division
from copy import copy
import platform
import pymc as pm
import numpy as np
np.seterr(divide='ignore')
from numpy.random import rand

import hddm
from scipy.stats import scoreatpercentile


def check_model(model, params_true, assert_=False, conf_interval = 95):
    """calculate the posterior estimate error if hidden parameters are known (e.g. when simulating data)."""

    # Check for proper chain convergence
    #check_geweke(model, assert_=True)
    print model.summary()
    # Test for correct parameter estimation
    fail = False
    for param in model.param_names:
        est = model.params_est[param]
        est_std = model.params_est_std[param]
        truth = params_true[param]
        trace = model.group_params[param].trace()[:]
        lb = (50 - conf_interval/2.)
        lb_score = scoreatpercentile(trace, lb)
        ub = (50 + conf_interval/2.)
        ub_score = scoreatpercentile(trace, ub)
        fell = np.sum(truth > trace)*1./len(trace) * 100
        if lb_score==ub_score:
            fell = 50


        print "%s: Truth: %f, Estimated: %f, lb: %f, ub: %f,  fell in: %f" % \
        (param, truth, est, lb_score, ub_score, fell)
        if (fell < lb) or (fell > ub):
            fail = True
            print "the true value of %s is outsize of the confidence interval !*!*!*!*!*!*!" % param
        
    if assert_:
        assert (fail==False)
    
    ok = not fail
    return ok

def check_rejection(model, assert_ = True):    
    """ check if the rejection ratio is not too high"""
    
    for param in model.group_params:
        trace = model.group_params[param].trace()[:]
        rej =  np.sum(np.diff(trace)==0)
        rej_ratio = rej*1.0/len(trace)
        print "rejection ratio for %s: %f" %(param, rej_ratio)
        if (rej_ratio < 0.5) or (rej_ratio > 0.8):
            msg = "%s still need to be tuned" % param
            if assert_:
                assert 1==0, msg
            else:
                print msg


def rand_simple_params():
    params = {}
    params['V'] = 0    
    params['Z'] = 0
    params['T'] = 0
    params['v'] = (rand()-.5)*4
    params['t'] = 0.2+rand()*0.3+(params['T']/2)
    params['a'] = 1.5+rand()
    params['z'] = .4+rand()*0.2
    return params


def test_simple(nTimes=20):
    thin = 1
    samples = 10000
    burn = 10000
    n_iter = burn + samples*thin
    n_data = 300
    for i_time in range(nTimes):
        params = rand_simple_params()
        data,temp = hddm.generate.gen_rand_data(n_data, params)
        model = hddm.model.HDDM(data, no_bias=False)
        model.mcmc(sample=False);
        model.mcmc_model.sample(n_iter, burn=burn, thin=thin)
        [model.mcmc_model.use_step_method(pm.Metropolis, x,proposal_sd=0.001) for x in model.group_params.values()]
        model._gen_stats()
        check_model(model, params, assert_=False)
        check_rejection(model, assert_ = False)
