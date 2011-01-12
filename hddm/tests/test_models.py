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


def scale(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

    
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

class TestMulti(unittest.TestCase):
    def runTest(self):
        pass

    def create_multi_data(self, params1, params2, samples=1000, subj=False, num_subjs=10.):
        import numpy.lib.recfunctions as rec

        if not subj:
            data1 = hddm.generate.gen_rts(params1, structured=True, samples=samples)
            data2 = hddm.generate.gen_rts(params2, structured=True, samples=samples)
        else:
            data1 = gen_rand_subj_data(num_subjs=num_subjs, params=params1, samples=samples/num_subjs, gen_data=True, add_noise=True, tag=None)[0]
            data2 = gen_rand_subj_data(num_subjs=num_subjs, params=params2, samples=samples/num_subjs, gen_data=True, add_noise=True, tag=None)[0]
           
        # Add stimulus field
        data1 = rec.append_fields(data1, names='stim', data=np.zeros(data1.shape[0]), dtypes='i8', usemask=False)
        data2 = rec.append_fields(data2, names='stim', data=np.ones(data2.shape[0]), dtypes='i8', usemask=False)

        data = rec.stack_arrays((data1, data2), usemask=False)

        return data

    def diff_model(self, param, subj=True, num_subjs=10, change=.5, samples=10000):
        params1 = {'v':.5, 'a':2., 'z':1., 't': .3, 'T':0., 'V':0., 'Z':0.}
        params2 = copy(params1)
        params2[param] = params1[param]+change

        data = self.create_multi_data(params1, params2, subj=subj, num_subjs=num_subjs, samples=samples)

        model = hddm.models.Multi(data, depends_on={param:['stim']}, is_subj_model=subj)
        model.mcmc()

        print model.summary()
        return model

    def test_diff_v(self, samples=1000):
        m = self.diff_model('v', subj=False, change=.5, samples=samples)
        return m
    
    def test_diff_a_subj(self, samples=10000):
        m = self.diff_model('a', subj=True, change=-.5, samples=samples)
        return m
        
    
class TestSingle(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSingle, self).__init__(*args, **kwargs)
        gen_data = False
        self.data, self.params_true = gen_rand_data(gen_data=gen_data, tag='global_test')
        self.data_subj, self.params_true_subj = gen_rand_subj_data(gen_data=gen_data, samples=300, add_noise=False, tag='subj_test')
        self.data_basic, self.params_true_basic = gen_rand_data(gen_data=gen_data, samples=2000, no_var=True, tag='global_basic_test')
        
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
        
    def test_basic(self):
        model = hddm.models.Multi(self.data_basic, no_bias=True)
        model.mcmc(samples=self.samples, burn=self.burn)
        self.check_model(model, self.params_true_basic_no_s)
        
    def test_full_mc(self):
        model = hddm.models.Multi(self.data, model_type='full_mc', no_bias=False)
        model.mcmc(samples=self.samples, burn=self.burn)
        self.check_model(model, self.params_true)

    def test_lba(self):
        model = hddm.models.Multi(self.data, is_subj_model=False, normalize_v=True, model_type='lba')
        model.mcmc(samples=self.samples, burn=self.burn)
        #self.check_model(model, self.params_true_lba)
        print model.params_est
        return model

    def test_lba_subj(self):
        model = hddm.models.Multi(self.data_subj, is_subj_model=True, normalize_v=True, model_type='lba')
        model.mcmc(samples=self.samples, burn=self.burn)
        #self.check_model(model, self.params_true_lba)
        print model.params_est
        return model

    def test_full(self):
        return
        model = hddm.models.Multi(self.data, model_type='full', no_bias=False)
        model.mcmc(samples=self.samples, burn=self.burn)
        self.check_model(model, self.params_true)

    def test_full_subj(self):
        return
        model = hddm.models.Multi(self.data_subj, model_type='full', is_subj_model=True, no_bias=False)
        model.mcmc(samples=self.samples, burn=self.burn)
        self.check_model(model, self.params_true_subj)

    def test_subjs_fixed_z(self):
        model = hddm.models.Multi(self.data_subj, model_type='simple', is_subj_model=True, no_bias=False)
        model.mcmc(samples=self.samples, burn=self.burn)
        self.check_model(model, self.params_true)

    def test_simple(self):
        model = hddm.models.Multi(self.data, model_type='simple', no_bias=True)
        model.mcmc(samples=self.samples, burn=self.burn)
        self.check_model(model, self.params_true_no_s)
        return model
    
    def test_simple_subjs(self):
        model = hddm.models.Multi(self.data_subj, model_type='simple', is_subj_model=True, no_bias=True)
        model.mcmc(samples=self.samples, burn=self.burn)
        self.check_model(model, self.params_true_subj_no_s)
        return model
    
    def test_simple_diff(self):
        return
        # Fit first model
        model = hddm.models.HDDM_simple(data, no_bias=True)
        model.mcmc()

        # Fit second model
        params = {'v': .1, 'V': 0.3, 'z': 0.9, 'Z': 0.25, 't': .3, 'T': 0.1, 'a': 1.8}
        data2, params_true2 = gen_rand_data(gen_data=True, params=params, num_samples=200)
        del params_true2['V']
        del params_true2['Z']
        del params_true2['T']
        model2 = hddm.models.HDDM_simple(data2, no_bias=True)
        model2.mcmc()

        self.check_model(model, params_true, assert_=True)
        self.check_model(model2, params_true2, assert_=True)
        model.plot_global(params_true=params_true)
        model2.plot_global(params_true=params_true2)

        return model, model2

    def test_chains(self):
        return
        models = run_parallel_chains(hddm.models.Multi, [data, data], ['test1', 'test2'])
        return models

    def check_model(self, model, params_true):
        """calculate the posterior estimate error if hidden parameters are known (e.g. when simulating data)."""
        err=[]

        # Check for proper chain convergence
        #check_geweke(model, assert_=True)

        # Test for correct parameter estimation
        for param in params_true.iterkeys():
            # no-bias models do not contain z
            if model.no_bias and param == 'z':
                continue
            
            est = model.params_est[param]
            est_std = model.params_est_std[param]
            truth = params_true[param]
            err.append((est - truth)**2)
            if self.assert_:
                self.assertAlmostEqual(est, truth, 1)
            
            print "%s: Truth: %f, Estimated: %f, Error: %f, Std: %f" % (param, truth, est, err[-1], est_std)
                
            print "Summed error: %f" % np.sum(err)

        return np.sum(err)

    
if __name__=='__main__':
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestLikelihoodFuncsGPU)
    #unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestLikelihoodFuncsLBA)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestHDDM)
    unittest.TextTestRunner(verbosity=2).run(suite2)
