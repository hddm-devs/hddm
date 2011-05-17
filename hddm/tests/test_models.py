from __future__ import division

import nose

import unittest
from hddm.diag import *

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

    model = hddm.model.HDDM(data, depends_on={param:['stim']}, is_group_model=subj)
    model.mcmc(retry=False)

    print model.summary()
    return model

            
        
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
        nodes = model.create()
        mc = pm.MCMC(nodes)
        mc.sample(self.samples, burn=self.burn)
        check_model(mc, self.params_true_basic_no_s)
        
    def test_full_mc(self, assert_=True):
        raise SkipTest("Disabled.")
        model = hddm.model.HDDM(self.data, model_type='full_mc', no_bias=False)
        nodes = model.create()
        mc = pm.MCMC(nodes)
        mc.sample(self.samples, burn=self.burn)
        check_model(mc, self.params_true, assert_=assert_)

        return model

    def test_full_interp_subjs_V_T_Z(self, assert_=True):
        raise SkipTest("Disabled.")
        data_subj, params_true = gen_rand_subj_data(params={'a':2,'v':.5,'z':.5,'t':.3,'V':1., 'Z':.75, 'T':.6},
                                                    samples=300)
        model = hddm.model.HDDM(data_subj, model_type='full_intrp', no_bias=True)
        nodes = model.create()
        mc = pm.MCMC(nodes)
        mc.sample(self.samples, burn=self.burn)
        check_model(mc, params_true, assert_=assert_)

        return model

    def test_full_interp_subjs_T(self, assert_=True):
        raise SkipTest("Disabled.")
        data_subj, params_true = gen_rand_subj_data(params={'a':2,'v':.5,'z':.5,'t':.3,'V':0,'Z':0.,'T':.6},
                                                    samples=300)
        model = hddm.model.HDDM(data_subj, model_type='full_intrp', no_bias=True, exclude_inter_var_params=['Z','V'])
        nodes = model.create()
        mc = pm.MCMC(nodes)
        mc.sample(self.samples, burn=self.burn)
        check_model(mc, params_true, assert_=assert_)

        return model

    def test_full_interp_subjs_V(self, assert_=True):
        raise SkipTest("Disabled.")
        data_subj, params_true = gen_rand_subj_data(params={'a':2,'v':.5,'z':.5,'t':.3,'V':1., 'Z':0, 'T':0},
                                                    samples=300)
        model = hddm.model.HDDM(data_subj, model_type='full_intrp', no_bias=True, exclude_inter_var_params=['Z','T'])
        nodes = model.create()
        mc = pm.MCMC(nodes)
        mc.sample(self.samples, burn=self.burn)
        check_model(mc, params_true, assert_=assert_)

        return model

    def test_full_interp_subjs_Z(self, assert_=True):
        raise SkipTest("Disabled.")
        data_subj, params_true = gen_rand_subj_data(params={'a':2,'v':.5,'z':.5,'t':.3,'V':1., 'Z':.7, 'T':0},
                                                    samples=300)
        model = hddm.model.HDDM(data_subj, model_type='full_intrp', no_bias=True, exclude_inter_var_params=['V','T'])
        nodes = model.create()
        mc = pm.MCMC(nodes)
        mc.sample(self.samples, burn=self.burn)
        check_model(mc, params_true, assert_=assert_)

        return model

    def test_lba(self):
        model = hddm.model.HDDM(self.data, is_group_model=False, normalize_v=True, model_type='lba')
        nodes = model.create()
        mc = pm.MCMC(nodes)
        mc.sample(self.samples, burn=self.burn)
        #self.check_model(mc, self.params_true_lba)
        print model.params_est
        return model

    def test_lba_subj(self):
        model = hddm.model.HDDM(self.data_subj, is_group_model=True, normalize_v=True, model_type='lba')
        nodes = model.create()
        mc = pm.MCMC(nodes)
        mc.sample(self.samples, burn=self.burn)
        #self.check_model(mc, self.params_true_lba)
        print model.params_est
        return model

    def test_full(self):
        raise SkipTest("Disabled.")
        model = hddm.model.HDDM(self.data, model_type='full', no_bias=False)
        nodes = model.create()
        mc = pm.MCMC(nodes)
        mc.sample(self.samples, burn=self.burn)
        check_model(mc, self.params_true)

    def test_full_subj(self):
        raise SkipTest("Disabled.")
        model = hddm.model.HDDM(self.data_subj, model_type='full', is_group_model=True, no_bias=False)
        nodes = model.create()
        mc = pm.MCMC(nodes)
        mc.sample(self.samples, burn=self.burn)
        check_model(mc, self.params_true_subj)

    def test_subjs_no_bias(self):
        model = hddm.model.HDDM(self.data_subj, model_type='simple', is_group_model=True, no_bias=True)
        nodes = model.create()
        mc = pm.MCMC(nodes)
        mc.sample(self.samples, burn=self.burn)
        check_model(mc, self.params_true)

    def test_subjs_bias(self):
        model = hddm.model.HDDM(self.data_subj, model_type='simple', is_group_model=True, no_bias=False)
        nodes = model.create()
        mc = pm.MCMC(nodes)
        mc.sample(self.samples, burn=self.burn)
        check_model(mc, self.params_true)

    def test_simple(self):
        model = hddm.model.HDDM(self.data_basic, model_type='simple', no_bias=True)
        nodes = model.create()
        mc = pm.MCMC(nodes)
        mc.sample(self.samples, burn=self.burn)
        check_model(mc, self.params_true_no_s)
        return model
    
    def test_simple_subjs(self):
        model = hddm.model.HDDM(self.data_subj, model_type='simple', is_group_model=True, no_bias=True)
        nodes = model.create()
        mc = pm.MCMC(nodes)
        mc.sample(self.samples, burn=self.burn)
        check_model(mc, self.params_true_subj_no_s)
        return model
    
    def test_chains(self):
        raise SkipTest("Disabled.")
        models = run_parallel_chains(hddm.model.HDDM, [data, data], ['test1', 'test2'])
        return models

class TestAntisaccade(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestAntisaccade, self).__init__(*args, **kwargs)
        self.params = {'v':-2.,
                       'v_switch': 2.,
                       'a': 2.5,
                       't': .3,
                       't_switch': .3,
                       'z':.5,
                       'T': 0, 'Z':0, 'V':0}

        self.data = hddm.generate.gen_antisaccade_rts(params=self.params, samples_pro=100, samples_anti=500)[0]
        self.assert_ = True

        self.samples = 2000
        self.burn = 1000

    def runTest(self):
        return
    
    def test_pure_switch(self):
        model = hddm.model.HDDMAntisaccade(self.data, no_bias=True, is_group_model=False)
        nodes = model.create()
        mc = pm.MCMC(nodes)
        mc.sample(self.samples, burn=self.burn)
        check_model(mc, self.params)

    
if __name__=='__main__':
    pass
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestLikelihoodFuncsGPU)
    #unittest.TextTestRunner(verbosity=2).run(suite)

    #suite = unittest.TestLoader().loadTestsFromTestCase(TestLikelihoodFuncsLBA)
    #unittest.TextTestRunner(verbosity=2).run(suite)

    #suite2 = unittest.TestLoader().loadTestsFromTestCase(TestHDDM)
    #unittest.TextTestRunner(verbosity=2).run(suite2)
