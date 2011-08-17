from __future__ import division

import nose
import sys
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


def diff_model(param, subj=True, num_subjs=10, change=.5, samples=500):
    params1 = {'v':.5, 'a':2., 'z':.5, 't': .3, 'T':0., 'V':0., 'Z':0.}
    params2 = copy(params1)
    params2[param] = params1[param]+change

    data, tmp = hddm.generate.gen_rand_cond_data((params1, params2), samples_per_cond=samples)

    model = hddm.model.HDDM(data, depends_on={param:['cond']}, is_group_model=subj)

    return model
        
class TestMulti(unittest.TestCase):
    def runTest(self):
        pass

    def test_diff_v(self, samples=1000):
        m = diff_model('v', subj=False, change=.5, samples=samples)
        return m
    
    def test_diff_a_subj(self, samples=1000):
        raise SkipTest("Disabled.")
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
        includes = [[], ['z'],['z', 'V'],['z', 'T'],['z', 'Z'], ['z', 'Z','T'], ['z', 'Z','T','V']]
        for include in includes:
            data, params_true = hddm.generate.gen_rand_data(samples=500, include=include, method='cdf')
            model = hddm.model.HDDM(data, include=include, bias='z' in include, is_group_model=False)
            mc = model.mcmc()
            mc.sample(self.samples, burn=self.burn)
            check_model(mc, params_true, assert_=assert_)

        return mc

    def test_HDDM_group(self, assert_=True):
        raise SkipTest("Disabled.")
        includes = [[], ['z'],['z', 'V'],['z', 'T'],['z', 'Z'], ['z', 'Z','T'], ['z', 'Z','T','V']]
        for include in includes:
            data, params_true = hddm.generate.gen_rand_subj_data(samples=500, num_subjs=5)
            model = hddm.model.HDDM(data, include=include, bias='z' in include, is_group_model=True)
            mc = model.mcmc()
            mc.sample(self.samples, burn=self.burn)
            check_model(mc, params_true, assert_=assert_)

        return mc

    def test_HDDMGPU(self, assert_=True):
        try:
            import pycuda
        except ImportError:
            raise SkipTest("Disabled.")
        include = ['V']
        data, params_true = hddm.generate.gen_rand_data(samples=500, include=include)
        model = hddm.model.HDDMGPU(data, include=include, no_bias=False, is_group_model=False)
        nodes = model.create()
        mc = pm.MCMC(nodes)
        mc.sample(self.samples, burn=self.burn)
        check_model(mc, params_true, assert_=assert_)

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
    
    def test_cont(self, assert_=False):
        params_true = gen_rand_params(include = ())
        data, temp = hddm.generate.gen_rand_data(samples=300, params=params_true)
        data[0]['rt'] = min(abs(data['rt']))/2.
        data[1]['rt'] = max(abs(data['rt'])) + 0.8           
        hm = hddm.HDDMContaminant(data, bias=True, is_group_model=False)
        hm.sample(self.samples, burn=self.burn)
        check_model(hm.mc, params_true, assert_=assert_)
        cont_res = hm.cont_report(plot=False)
        cont_idx = cont_res['cont_idx']
        self.assertTrue((0 in cont_idx) and (1 in cont_idx), "did not find the right outliers")
        self.assertTrue(len(cont_idx)<15, "found too many outliers (%d)" % len(cont_idx))

        return hm

    def test_cont_subj(self, assert_=False):
        data_samples = 200
        num_subjs = 2
        data, params_true = hddm.generate.gen_rand_subj_data(num_subjs=num_subjs, params=None, 
                                                        samples=data_samples, noise=0.0001,include=())
        for i in range(num_subjs):
            data[data_samples*i]['rt'] = min(abs(data['rt']))/2.
            data[data_samples*i + 1]['rt'] = max(abs(data['rt'])) + 0.8           
        hm = hddm.model.HDDMContaminant(data, bias=True, is_group_model=True)
        hm.sample(self.samples, burn=self.burn)
        check_model(hm.mc, params_true, assert_=assert_)
        cont_res = hm.cont_report(plot=False)
        for i in range(num_subjs):
            cont_idx = cont_res[i]['cont_idx']
            self.assertTrue((0 in cont_idx) and (1 in cont_idx), "did not found the right outliers")
            self.assertTrue(len(cont_idx)<15, "found too many outliers (%d)" % len(cont_idx))

        return hm


if __name__=='__main__':
    print "Run nosetest.py"
