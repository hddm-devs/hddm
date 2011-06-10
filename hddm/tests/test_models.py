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

        self.samples = 10000
        self.burn = 5000

    def runTest(self):
        return
    
    def test_HDDM(self, assert_=True):
        #raise SkipTest("Disabled.")
        includes = [['z'],['z', 'V'],['z', 'T'],['z', 'Z'], ['z', 'Z','T'], ['z', 'Z','T','V']]
        for include in includes:
            data, params_true = hddm.generate.gen_rand_data(samples=500, include=include)
            model = hddm.model.HDDM(data, include=include, no_bias=False, is_group_model=False)
            nodes = model.create()
            mc = pm.MCMC(nodes)
            mc.sample(self.samples, burn=self.burn)
            check_model(mc, params_true, assert_=assert_)

        return mc

    def test_HDDM_group(self, assert_=True):
        includes = [['z'],['z', 'V'],['z', 'T'],['z', 'Z'], ['z', 'Z','T'], ['z', 'Z','T','V']]
        for include in includes:
            data, params_true = hddm.generate.gen_rand_subj_data(samples=500, num_subjs=5)
            model = hddm.model.HDDM(data, include=include, no_bias=False, is_group_model=True)
            nodes = model.create()
            mc = pm.MCMC(nodes)
            mc.sample(self.samples, burn=self.burn)
            check_model(mc, params_true, assert_=assert_)

        return mc

    def test_HDDM_full_extended(self, assert_=True):
        data, params_true = hddm.generate.gen_rand_data(samples=500, include='all')

        model = hddm.model.HDDMFullExtended(data, no_bias=False, is_group_model=False)
        nodes = model.create()
        #pm.MAP(nodes).fit()
        mc = pm.MCMC(nodes)
        mc.sample(self.samples, burn=self.burn)
        check_model(mc, params_true, assert_=assert_)

        return mc

    def test_HDDM_full_extended_subj(self):
        raise SkipTest("Disabled.")
        data, params_true = hddm.generate.gen_rand_subj_data(samples=100)

        model = hddm.model.HDDMFullExtended(data, no_bias=False, is_group_model=True)
        nodes = model.create()
        mc = pm.MCMC(nodes)
        mc.sample(20000, burn=15000)
        check_model(mc, params_true, assert_=assert_)

        return mc
        
    def test_lba(self):
        raise SkipTest("Disabled.")
        model = hddm.model.HLBA(self.data, is_group_model=False, normalize_v=True)
        nodes = model.create()
        mc = pm.MCMC(nodes)
        mc.sample(self.samples, burn=self.burn)
        #self.check_model(mc, self.params_true_lba)
        print model.params_est
        return model

    def test_lba_subj(self):
        raise SkipTest("Disabled.")
        model = hddm.model.HLBA(self.data_subj, is_group_model=True, normalize_v=True)
        nodes = model.create()
        mc = pm.MCMC(nodes)
        mc.sample(self.samples, burn=self.burn)
        #self.check_model(mc, self.params_true_lba)
        print model.params_est
        return model

if __name__=='__main__':
    pass
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestLikelihoodFuncsGPU)
    #unittest.TextTestRunner(verbosity=2).run(suite)

    #suite = unittest.TestLoader().loadTestsFromTestCase(TestLikelihoodFuncsLBA)
    #unittest.TextTestRunner(verbosity=2).run(suite)

    #suite2 = unittest.TestLoader().loadTestsFromTestCase(TestHDDM)
    #unittest.TextTestRunner(verbosity=2).run(suite2)
