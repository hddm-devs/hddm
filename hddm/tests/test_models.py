from __future__ import division
from copy import copy

import unittest
import numpy as np
import pandas as pd
pd.set_printoptions(precision=4)
from nose import SkipTest

import hddm
from hddm.diag import check_model

def diff_model(param, subj=True, num_subjs=10, change=.5, size=500):
    params_cond_a = {'v':.5, 'a':2., 'z':.5, 't': .3, 'st':0., 'sv':0., 'sz':0.}
    params_cond_b = copy(params_cond_a)
    params_cond_b[param] += change

    params = {'A': params_cond_a, 'B': params_cond_b}

    data, subj_params = hddm.generate.gen_rand_data(params, subjs=num_subjs, size=size)

    model = hddm.model.HDDM(data, depends_on={param:['condition']}, is_group_model=subj)

    return model

class TestMulti(unittest.TestCase):
    def runTest(self):
        pass

    def test_diff_v(self, size=100):
        m = diff_model('v', subj=False, change=.5, size=size)
        return m

    def test_diff_a(self, size=100):
        m = diff_model('a', subj=False, change=-.5, size=size)
        return m

    def test_diff_a_subj(self, size=100):
        raise SkipTest("Disabled.")
        m = diff_model('a', subj=True, change=-.5, size=size)
        return m

class TestSingleBreakdown(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSingleBreakdown, self).__init__(*args, **kwargs)

        self.size = 50
        self.burn = 10

    def runTest(self):
        return

    def test_HDDM(self, assert_=False):
        includes = [[], ['z'],['z', 'sv'],['z', 'st'],['z', 'sz'], ['z', 'sz','st'], ['z', 'sz','st','sv']]
        for include in includes:
            params = hddm.generate.gen_rand_params(include=include)
            data, params_true = hddm.generate.gen_rand_data(params, size=500, subjs=1)
            model = hddm.model.HDDM(data, include=include, bias='z' in include, is_group_model=False)
            model.map()
            model.sample(self.size, burn=self.burn)
            check_model(model.mc, params_true, assert_=assert_)

        return model.mc

    def test_HDDM_group(self, assert_=False):
        includes = [[], ['z'],['z', 'sv'],['z', 'st'],['z', 'sz'], ['z', 'sz','st'], ['z', 'sz','st','sv']]
        for include in includes:
            params = hddm.generate.gen_rand_params(include=include)
            data, params_true = hddm.generate.gen_rand_data(params, size=500, subjs=5)
            model = hddm.model.HDDM(data, include=include, bias='z' in include, is_group_model=True)
            model.approximate_map()
            model.sample(self.size, burn=self.burn)
            check_model(model.mc, params_true, assert_=assert_)

        return model.mc

    def test_HDDMTransform(self, assert_=False):
        includes = [[], ['z'],['z', 'sv'],['z', 'st'],['z', 'sz'], ['z', 'sz','st'], ['z', 'sz','st','sv']]
        for include in includes:
            params = hddm.generate.gen_rand_params(include=include)
            data, params_true = hddm.generate.gen_rand_data(params, size=500, subjs=1)
            model = hddm.model.HDDMTransform(data, include=include, bias='z' in include, is_group_model=False)
            model.map()
            model.sample(self.size, burn=self.burn)
            check_model(model.mc, params_true, assert_=assert_)

        return model.mc

    def test_HDDMTransform_group(self, assert_=False):
        includes = [[], ['z'],['z', 'sv'],['z', 'st'],['z', 'sz'], ['z', 'sz','st'], ['z', 'sz','st','sv']]
        for include in includes:
            params = hddm.generate.gen_rand_params(include=include)
            data, params_true = hddm.generate.gen_rand_data(params, size=500, subjs=5)
            model = hddm.model.HDDMTransform(data, include=include, bias='z' in include, is_group_model=True)
            model.approximate_map()
            model.sample(self.size, burn=self.burn)
            check_model(model.mc, params_true, assert_=assert_)

        return model.mc

    def test_cont(self, assert_=False):
        raise SkipTest("Disabled.")
        params_true = hddm.generate.gen_rand_params(include=())
        data, temp = hddm.generate.gen_rand_data(size=300, params=params_true)
        data[0]['rt'] = min(abs(data['rt']))/2.
        data[1]['rt'] = max(abs(data['rt'])) + 0.8
        hm = hddm.HDDMContUnif(data, bias=True, is_group_model=False)
        hm.sample(self.size, burn=self.burn)
        check_model(hm.mc, params_true, assert_=assert_)
        cont_res = hm.cont_report(plot=False)
        cont_idx = cont_res['cont_idx']
        self.assertTrue((0 in cont_idx) and (1 in cont_idx), "did not find the right outliers")
        self.assertTrue(len(cont_idx)<15, "found too many outliers (%d)" % len(cont_idx))

        return hm

    def test_cont_subj(self, assert_=False):
        raise SkipTest("Disabled.")
        data_samples = 200
        num_subjs = 2
        data, params_true = hddm.generate.gen_rand_subj_data(num_subjs=num_subjs, params=None,
                                                        size=data_samples, noise=0.0001,include=())
        for i in range(num_subjs):
            data[data_samples*i]['rt'] = min(abs(data['rt']))/2.
            data[data_samples*i + 1]['rt'] = max(abs(data['rt'])) + 0.8
        hm = hddm.model.HDDMContUnif(data, bias=True, is_group_model=True)
        hm.sample(self.size, burn=self.burn)
        check_model(hm.mc, params_true, assert_=assert_)
        cont_res = hm.cont_report(plot=False)
        for i in range(num_subjs):
            cont_idx = cont_res[i]['cont_idx']
            self.assertTrue((0 in cont_idx) and (1 in cont_idx), "did not found the right outliers")
            self.assertTrue(len(cont_idx)<15, "found too many outliers (%d)" % len(cont_idx))

        return hm


def test_chisquare_recovery_single_subject(repeats=10):

    #init
    initial_value = {'a': 1,
                     'v': 0,
                     'z': 0.5,
                     't': 0.01,
                     'st': 0,
                     'sv': 0,
                     'sz': 0}

    all_params = set(['a','v','t','z','st','sz','sv'])
    include_sets = [set(['a','v','t']),
                  set(['a','v','t','st']),
                  set(['a','v','t','sz']),
                  set(['a','v','t','sv'])]

    wfpt = hddm.likelihoods.generate_wfpt_stochastic_class()
    v = [0, 0.5, 0.75, 1.]
    n_conds = len(v)

    np.random.seed(1)
    for include in include_sets:
        for i in range(repeats):
            #generate params for experiment with n_conds conditions
            org_params = hddm.generate.gen_rand_params(include)
            merged_params = org_params.copy()
            del merged_params['v']
            cond_params = {};
            for i in range(n_conds):
                #create a set of parameters for condition i
                #put them in i_params, and in cond_params[c#i]
                i_params = org_params.copy()
                del i_params['v']
                i_params['v'] = v[i]
                cond_params['c%d' %i] = i_params

                #create also a set of all the parameters in on dictionary so we can compare them
                #to our estimation at the end
                merged_params['v(c%d)' % i] = v[i]

            print merged_params

            #generate samples
            samples, _ = hddm.generate.gen_rand_data(cond_params, size=5000)

            #optimize
            h = hddm.model.HDDM(samples, include=include, depends_on={'v':'condition'})
            print "optimizing"
            recovered_params = h.quantiles_chi2square_optimization(verbose=0)

            #compare results to true values
            index = ['observed', 'estimated']
            df = pd.DataFrame([merged_params, recovered_params], index=index).dropna(1)
            pd.set_printoptions(precision=4)
            print df
            np.testing.assert_array_almost_equal(df.values[0], df.values[1], 1)


if __name__=='__main__':
    print "Run nosetest.py"
