from __future__ import division

import nose

import unittest

import numpy as np
import numpy.testing
import numpy.lib.recfunctions as rec
import matplotlib.pyplot as plt

import platform
from copy import copy
import subprocess

import pymc as pm

import hddm

from hddm.likelihoods import *
from hddm.generate import *
from scipy.stats import scoreatpercentile
from numpy.random import rand

from nose import SkipTest

def analyze_regress(posterior_trace, x_range=(-.5, .5), bins=100, plot=True):
    import scipy.interpolate
    
    x = np.linspace(x_range[0], x_range[1], bins)
    #prior_dist = pm.Normal('prior', mu=0, tau=100)
    prior_dist = pm.Uniform('prior', lower=-.5, upper=.5)

    prior_trace = np.array([prior_dist.rand() for i in range(10000)])

    sav_dick, prior, posterior, prior0, posterior0 = savage_dickey(prior_trace, posterior_trace, range=x_range, bins=bins)

    print "Savage Dickey: %f." % (sav_dick)

    prior_inter = scipy.interpolate.spline(xk=x, yk=prior, xnew=x)
    posterior_inter = scipy.interpolate.spline(xk=x, yk=posterior, xnew=x)

    plt.plot(x, prior_inter, label='prior')
    plt.plot(x, posterior_inter, label='posterior')

    return sav_dick

def create_multi_data(params1, params2, samples=1000, subj=False, num_subjs=10.):
    import numpy.lib.recfunctions as rec

    if not subj:
        data1 = hddm.generate.gen_rts(params1, structured=True, samples=samples)
        data2 = hddm.generate.gen_rts(params2, structured=True, samples=samples)
    else:
        data1 = gen_rand_subj_data(num_subjs=num_subjs, params=params1, samples=samples/num_subjs, )[0]
        data2 = gen_rand_subj_data(num_subjs=num_subjs, params=params2, samples=samples/num_subjs, )[0]

    # Add stimulus field
    data1 = rec.append_fields(data1, names='stim', data=np.zeros(data1.shape[0]), dtypes='i8', usemask=False)
    data2 = rec.append_fields(data2, names='stim', data=np.ones(data2.shape[0]), dtypes='i8', usemask=False)

    data = rec.stack_arrays((data1, data2), usemask=False)

    return data

def diff_model(param, subj=True, num_subjs=10, change=.5, samples=10000):
    params1 = {'v':.5, 'a':2., 'z':.5, 't': .3, 'T':0., 'V':0., 'Z':0.}
    params2 = copy(params1)
    params2[param] = params1[param]+change

    data = create_multi_data(params1, params2, subj=subj, num_subjs=num_subjs, samples=samples)

    model = hddm.model.HDDM(data, depends_on={param:['stim']}, is_subj_model=subj)
    model.mcmc(retry=False)

    print model.summary()
    return model

def check_model(model, params_true, assert_=True, conf_interval = 95):
    """calculate the posterior estimate error if hidden parameters are known (e.g. when simulating data)."""
    err=[]

    # Check for proper chain convergence
    #check_geweke(model, assert_=True)
    print model.summary()
    # Test for correct parameter estimation
    fail = False
    for param in model.param_names:
        est = model.params_est[param]
        est_std = model.params_est_std[param]
        truth = params_true[param]
        err.append((est - truth)**2)
        trace = model.group_params[param].trace()[:]
        lb = (50 - conf_interval/2.)
        lb_score = scoreatpercentile(trace, lb)
        ub = (50 + conf_interval/2.)
        ub_score = scoreatpercentile(trace, ub)
        fell = np.sum(truth > trace)*1./len(trace) * 100


        print "%s: Truth: %f, Estimated: %f, lb: %f, ub: %f,  fell in: %f" % \
        (param, truth, est, lb_score, ub_score, fell)
        if (fell <= lb) or (fell >= ub):
            fail = True
            print "the true value of %s is outsize of the confidence interval !*!*!*!*!*!*!" % param
        
    if assert_:
        assert (fail==False) 

   

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
            

class TestAcc(unittest.TestCase):
    
    
    def __init__(self, *args, **kwargs):
        super(TestAcc, self).__init__(*args, **kwargs)
        self.thin = 1
        self.samples = 10000
        self.burn = 10000
        self.iter = self.burn + self.samples*self.thin


    def rand_simple_params(self):
        params = {}
        params['V'] = 0    
        params['Z'] = 0
        params['T'] = 0
        params['v'] = (rand()-.5)*4
        params['t'] = 0.2+rand()*0.3+(params['T']/2)
        params['a'] = 1.5+rand()
        params['z'] = .4+rand()*0.2
        return params

    
    def test_basic(self, nTimes=20):
        for i_time in range(nTimes):
            params = self.rand_simple_params()
            data,temp = hddm.generate.gen_rand_data(300, params)
            model = hddm.model.HDDM(data, no_bias=False)
            model.mcmc(sample=False);
            model.mcmc_model.sample(self.iter, burn=self.burn, thin=self.thin)
            [model.mcmc_model.use_step_method(pm.Metropolis, x,proposal_sd=0.001) for x in model.group_params.values()]
            model._gen_stats()
            check_model(model, params, assert_=False)
            check_rejection(model, assert_ = False)
        
class TestMulti(unittest.TestCase):
    def runTest(self):
        pass

    def test_diff_v(self, samples=1000):
        m = diff_model('v', subj=False, change=.5, samples=samples)
        return m
    
    def test_diff_a_subj(self, samples=1000):
        m = diff_model('a', subj=True, change=-.5, samples=samples)
        return m
        
    
class TestSingle(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSingle, self).__init__(*args, **kwargs)
        self.data, self.params_true = gen_rand_data()
        self.data_subj, self.params_true_subj = gen_rand_subj_data(samples=300)
        self.data_basic, self.params_true_basic = gen_rand_data(samples=5000, no_var=True)
        
        self.assert_ = True
        
        self.params_true_no_s = copy(self.params_true)
        del self.params_true_no_s['V']
        del self.params_true_no_s['Z']
        del self.params_true_no_s['T']

        self.params_true_subj_no_s = copy(self.params_true_subj)
        del self.params_true_subj_no_s['V']
        del self.params_true_subj_no_s['Z']
        del self.params_true_subj_no_s['T']

        self.params_true_lba = copy(self.params_true)
        del self.params_true_lba['Z']
        del self.params_true_lba['T']

        self.params_true_basic_no_s = copy(self.params_true_basic)
        del self.params_true_basic_no_s['V']
        del self.params_true_basic_no_s['Z']
        del self.params_true_basic_no_s['T']

        self.samples = 10000
        self.burn = 5000

    def runTest(self):
        return
    
    def test_basic(self):
        model = hddm.model.HDDM(self.data_basic, no_bias=True)
        model.mcmc(samples=self.samples, burn=self.burn)
        check_model(model, self.params_true_basic_no_s)
        
    def test_full_mc(self, assert_=True):
        return
        model = hddm.model.HDDM(self.data, model_type='full_mc', no_bias=False)
        model.mcmc(samples=self.samples, burn=self.burn)
        check_model(model, self.params_true, assert_=assert_)

        return model

    def test_full_interp_subjs_V_T_Z(self, assert_=True):
        return
        data_subj, params_true = gen_rand_subj_data(params={'a':2,'v':.5,'z':.5,'t':.3,'V':1., 'Z':.75, 'T':.6},
                                                    samples=300)
        model = hddm.model.HDDM(data_subj, model_type='full_intrp', no_bias=True)
        model.mcmc(samples=self.samples, burn=self.burn)
        check_model(model, params_true, assert_=assert_)

        return model

    def test_full_interp_subjs_T(self, assert_=True):
        return
        data_subj, params_true = gen_rand_subj_data(params={'a':2,'v':.5,'z':.5,'t':.3,'V':0,'Z':0.,'T':.6},
                                                    samples=300)
        model = hddm.model.HDDM(data_subj, model_type='full_intrp', no_bias=True, exclude_inter_var_params=['Z','V'])
        model.mcmc(samples=self.samples, burn=self.burn)
        check_model(model, params_true, assert_=assert_)

        return model

    def test_full_interp_subjs_V(self, assert_=True):
        return
        data_subj, params_true = gen_rand_subj_data(params={'a':2,'v':.5,'z':.5,'t':.3,'V':1., 'Z':0, 'T':0},
                                                    samples=300)
        model = hddm.model.HDDM(data_subj, model_type='full_intrp', no_bias=True, exclude_inter_var_params=['Z','T'])
        model.mcmc(samples=self.samples, burn=self.burn)
        check_model(model, params_true, assert_=assert_)

        return model

    def test_full_interp_subjs_Z(self, assert_=True):
        return
        data_subj, params_true = gen_rand_subj_data(params={'a':2,'v':.5,'z':.5,'t':.3,'V':1., 'Z':.7, 'T':0},
                                                    samples=300)
        model = hddm.model.HDDM(data_subj, model_type='full_intrp', no_bias=True, exclude_inter_var_params=['V','T'])
        model.mcmc(samples=self.samples, burn=self.burn)
        check_model(model, params_true, assert_=assert_)

        return model

    def test_lba(self):
        model = hddm.model.HDDM(self.data, is_subj_model=False, normalize_v=True, model_type='lba')
        model.mcmc(samples=self.samples, burn=self.burn)
        #self.check_model(model, self.params_true_lba)
        print model.params_est
        return model

    def test_lba_subj(self):
        model = hddm.model.HDDM(self.data_subj, is_subj_model=True, normalize_v=True, model_type='lba')
        model.mcmc(samples=self.samples, burn=self.burn)
        #self.check_model(model, self.params_true_lba)
        print model.params_est
        return model

    def test_full(self):
        return
        model = hddm.model.HDDM(self.data, model_type='full', no_bias=False)
        model.mcmc(samples=self.samples, burn=self.burn)
        check_model(model, self.params_true)

    def test_full_subj(self):
        return
        model = hddm.model.HDDM(self.data_subj, model_type='full', is_subj_model=True, no_bias=False)
        model.mcmc(samples=self.samples, burn=self.burn)
        check_model(model, self.params_true_subj)

    def test_subjs_no_bias(self):
        model = hddm.model.HDDM(self.data_subj, model_type='simple', is_subj_model=True, no_bias=True)
        model.mcmc(samples=self.samples, burn=self.burn)
        check_model(model, self.params_true)

    def test_subjs_bias(self):
        model = hddm.model.HDDM(self.data_subj, model_type='simple', is_subj_model=True, no_bias=False)
        model.mcmc(samples=self.samples, burn=self.burn)
        check_model(model, self.params_true)

    def test_simple(self):
        model = hddm.model.HDDM(self.data_basic, model_type='simple', no_bias=True)
        model.mcmc(samples=self.samples, burn=self.burn)
        check_model(model, self.params_true_no_s)
        return model
    
    def test_simple_subjs(self):
        model = hddm.model.HDDM(self.data_subj, model_type='simple', is_subj_model=True, no_bias=True)
        model.mcmc(samples=self.samples, burn=self.burn)
        check_model(model, self.params_true_subj_no_s)
        return model
    
    def test_chains(self):
        return
        models = run_parallel_chains(hddm.model.HDDM, [data, data], ['test1', 'test2'])
        return models

    
if __name__=='__main__':
    pass
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestLikelihoodFuncsGPU)
    #unittest.TextTestRunner(verbosity=2).run(suite)

    #suite = unittest.TestLoader().loadTestsFromTestCase(TestLikelihoodFuncsLBA)
    #unittest.TextTestRunner(verbosity=2).run(suite)

    #suite2 = unittest.TestLoader().loadTestsFromTestCase(TestHDDM)
    #unittest.TextTestRunner(verbosity=2).run(suite2)
