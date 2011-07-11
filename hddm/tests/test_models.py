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

    data = hddm.generate.gen_rand_cond_data((params1, params2), samples_per_cond=samples)

    model = hddm.model.HDDM(data, depends_on={param:['cond']}, is_group_model=subj)

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
            model = hddm.model.HDDM(data, include=include, bias=True, is_group_model=False)
            mc = model.mcmc()
            mc.sample(self.samples, burn=self.burn)
            check_model(mc, params_true, assert_=assert_)

        return mc

    def test_HDDM_group(self, assert_=True):
        raise SkipTest("Disabled.")
        includes = [['z'],['z', 'V'],['z', 'T'],['z', 'Z'], ['z', 'Z','T'], ['z', 'Z','T','V']]
        for include in includes:
            data, params_true = hddm.generate.gen_rand_subj_data(samples=500, num_subjs=5)
            model = hddm.model.HDDM(data, include=include, bias=True, is_group_model=True)
            mc = model.mcmc()
            mc.sample(self.samples, burn=self.burn)
            check_model(mc, params_true, assert_=assert_)

        return mc


if __name__=='__main__':
    print "Run nosetest.py"
