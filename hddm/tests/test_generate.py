import unittest

import hddm
from scipy.stats import ks_2samp, kstest
import numpy as np

from nose import SkipTest

class TestGenerate(unittest.TestCase):
    def test_compare_drift_simulated_data_to_analytic(self):
        raise SkipTest("The function used KStest in the past, but since Wfpt is not scipy_distribution anymore it does not work")
        includes = [[],['z', 'sv'],['z', 'st'],['z', 'sz'], ['z', 'sz','st'], ['z', 'sz','st','sv']]
        Stochastic = hddm.likelihoods.generate_wfpt_stochastic_class(sampling_method='drift')
        for include in includes:
            params = hddm.generate.gen_rand_params(include=include)
            stoch = Stochastic('temp', size=1000, **params)
            [D, p_value] = kstest(stoch.rvs, stoch.cdf, N=1000)
            print('p_value: %f' % p_value)
            self.assertTrue(p_value > 0.05)

    def test_cdf_samples_to_drift_samples(self):
        np.random.seed(100)
        includes = [[],['z', 'sv'],['z', 'st'],['z', 'sz'], ['z', 'sz','st'], ['z', 'sz','st','sv']]
        for include in includes:
            params = hddm.generate.gen_rand_params(include=include)
            [D, p_value] = ks_2samp(hddm.generate.gen_rts(method='cdf', **params).rt.values,
                                    hddm.generate.gen_rts(method='drift', dt=1e-4, **params).rt.values)
            print('p_value: %f' % p_value)
            self.assertTrue(p_value > 0.05)

    def test_generate_breakdown(self):
        hddm.generate.gen_rand_data(subjs=10)
        hddm.generate.gen_rand_data(subjs=1)
        hddm.generate.gen_rand_data(n_fast_outliers=5, n_slow_outliers=5)
        hddm.generate.gen_rand_data(size=100)
