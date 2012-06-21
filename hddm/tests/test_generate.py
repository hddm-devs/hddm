
import unittest

import pymc as pm

import hddm
from hddm.likelihoods import *
from hddm.generate import *
from scipy.integrate import *
from scipy.stats import kstest

from nose import SkipTest

class TestGenerate(unittest.TestCase):
    def test_compare_sampled_data_to_analytic(self):
        sampler = hddm.likelihoods.wfpt_like
        sampler.sample_method = 'cdf'
        sampler.dt = 1e-6
        self._test_compare_samples_to_analytic(sampler.rv)

    def test_compare_drift_simulated_data_to_analytic(self):
        sampler = hddm.likelihoods.wfpt_like
        sampler.sample_method = 'drifts'
        self._test_compare_samples_to_analytic(sampler.rv)

    def _test_compare_samples_to_analytic(self, sampler):
        includes = [[],['z', 'sv'],['z', 'st'],['z', 'sz'], ['z', 'sz','st'], ['z', 'sz','st','sv']]
        for include in includes:
            params = hddm.generate.gen_rand_params(include=include)
            [D, p_value] = kstest(sampler.rvs, sampler.cdf,
                                  args=(params['v'], params['sv'], params['a'], params['z'], params['sz'], params['t'], params['st']), N=1000)
            print 'p_value: %f' % p_value
            self.assertTrue(p_value > 0.05)
