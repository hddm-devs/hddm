
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
        sampler = hddm.likelihoods.wfpt
        sampler.sample_method = 'cdf_py'
        self._test_compare_samples_to_analytic(sampler)
        
    def test_compare_drift_simulated_data_to_analytic(self):
        sampler = hddm.likelihoods.wfpt
        sampler.sample_method = 'drifts'
        self._test_compare_samples_to_analytic(sampler)

    def _test_compare_samples_to_analytic(self, sampler):
        excludes = [['Z','T','V'],['Z','T'],['V'],['T'],['Z']]
        for i_exclude in excludes:
            params = hddm.diag.rand_params(model_type='full_intrp', exclude=i_exclude)
            [D, p_value] = kstest(sampler.rvs, sampler.cdf,
                                  args=(params['v'], params['V'], params['a'], params['z'], params['Z'], params['t'], params['T']),N=100)
            print 'p_value: %f' % p_value
            self.assertTrue(p_value > 0.05)
