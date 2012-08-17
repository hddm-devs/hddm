from __future__ import division
from copy import copy
import itertools

import unittest
import pymc as pm
import numpy as np
import pandas as pd
pd.set_printoptions(precision=4)
from nose import SkipTest
import kabuki

import hddm
from hddm.diag import check_model

def diff_model(param, subj=True, num_subjs=10, change=.5, size=500):
    params_cond_a = {'v':.5, 'a':2., 'z':.5, 't': .3, 'st':0., 'sv':0., 'sz':0.}
    params_cond_b = copy(params_cond_a)
    params_cond_b[param] += change

    params = {'A': params_cond_a, 'B': params_cond_b}

    data, subj_params = hddm.generate.gen_rand_data(params, subjs=num_subjs, size=size)

    model = hddm.model.HDDMTruncated(data, depends_on={param:['condition']}, is_group_model=subj)

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

        self.iter = 50
        self.burn = 10

    def runTest(self):
        return

    def test_HDDM(self, assert_=False):
        includes = [[], ['z'],['z', 'sv'],['z', 'st'],['z', 'sz'], ['z', 'sz','st'], ['z', 'sz','st','sv']]
        model_classes = [hddm.model.HDDMTruncated, hddm.model.HDDM]
        for include, model_class in itertools.product(includes, model_classes):
            params = hddm.generate.gen_rand_params(include=include)
            data, params_true = hddm.generate.gen_rand_data(params, size=500, subjs=1)
            model = model_class(data, include=include, bias='z' in include, is_group_model=False)
            model.map()
            model.sample(self.iter, burn=self.burn)
            check_model(model.mc, params_true, assert_=assert_)

        return model.mc

    def test_HDDM_group(self, assert_=False):
        includes = [[], ['z'],['z', 'sv'],['z', 'st'],['z', 'sz'], ['z', 'sz','st'], ['z', 'sz','st','sv']]
        model_classes = [hddm.model.HDDMTruncated, hddm.model.HDDM]
        for include, model_class in itertools.product(includes, model_classes):
            params = hddm.generate.gen_rand_params(include=include)
            data, params_true = hddm.generate.gen_rand_data(params, size=500, subjs=5)
            model = model_class(data, include=include, bias='z' in include, is_group_model=True)
            model.approximate_map()
            model.sample(self.iter, burn=self.burn)
            check_model(model.mc, params_true, assert_=assert_)

        return model.mc

    def test_HDDM_group_only_group_nodes(self, assert_=False):
        group_only_nodes = [[], ['z'], ['z', 'st'], ['v', 'a']]
        model_classes = [hddm.model.HDDMTruncated, hddm.model.HDDM]

        for nodes, model_class in itertools.product(group_only_nodes, model_classes):
            params = hddm.generate.gen_rand_params(include=nodes)
            data, params_true = hddm.generate.gen_rand_data(params, size=500, subjs=5)
            model = model_class(data, include=nodes, group_only_nodes=nodes, is_group_model=True)
            for node in nodes:
                self.assertNotIn(node+'_subj', model.nodes_db.index)
                self.assertIn(node, model.nodes_db.index)


    def test_cont(self, assert_=False):
        raise SkipTest("Disabled.")
        params_true = hddm.generate.gen_rand_params(include=())
        data, temp = hddm.generate.gen_rand_data(size=300, params=params_true)
        data[0]['rt'] = min(abs(data['rt']))/2.
        data[1]['rt'] = max(abs(data['rt'])) + 0.8
        hm = hddm.HDDMContUnif(data, bias=True, is_group_model=False)
        hm.sample(self.iter, burn=self.burn)
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
        hm.sample(self.iter, burn=self.burn)
        check_model(hm.mc, params_true, assert_=assert_)
        cont_res = hm.cont_report(plot=False)
        for i in range(num_subjs):
            cont_idx = cont_res[i]['cont_idx']
            self.assertTrue((0 in cont_idx) and (1 in cont_idx), "did not found the right outliers")
            self.assertTrue(len(cont_idx)<15, "found too many outliers (%d)" % len(cont_idx))

        return hm


def optimization_recovery_single_subject(repeats=10, seed=1, true_starting_point=True,
                                         optimization_method='ML', max_retries=10):
    """
    recover parameters for single subjects model using ML
    """

    #init
    include_sets = [set(['a','v','t']),
                  set(['a','v','t','z']),
                  set(['a','v','t','st']),
                  set(['a','v','t','sz']),
                  set(['a','v','t','sv'])]

    #for each include set create a set of parametersm generate random data
    #and test the optimization function max_retries times.
    v = [0, 0.5, 0.75, 1.]
    np.random.seed(seed)
    for include in include_sets:
        for i in range(repeats):

            #generate params for experiment with n_conds conditions
            cond_params, merged_params = hddm.generate.gen_rand_params(include=include, cond_dict={'v':v})
            print "*** the true parameters ***"
            print merged_params

            #generate samples
            samples, _ = hddm.generate.gen_rand_data(cond_params, size=10000)

            #init model
            h = hddm.model.HDDM(samples, include=include, depends_on={'v':'condition'})

            #run optimization max_tries times
            recovery_ok = False
            for i_tries in range(max_retries):
                print "recovery attempt %d" % (i_tries + 1)

                #set the starting point of the optimization to the true value
                #of the parameters
                if true_starting_point:
                    set_hddm_nodes_values(h, merged_params)

                #optimize
                if optimization_method == 'ML':
                    recovered_params = h.ML_optimization()
                elif optimization_method == 'chisquare':
                    recovered_params = h.quantiles_chisquare_optimization(quantiles=np.linspace(0.05,0.95,10))
                else:
                    raise ValueError('unknown optimization method')

                #compare results to true values
                index = ['true', 'estimated']
                df = pd.DataFrame([merged_params, recovered_params], index=index, dtype=np.float).dropna(1)
                print df
                try:
                    #assert
                    np.testing.assert_allclose(df.values[0], df.values[1], atol=0.1)
                    recovery_ok = True
                    break
                except AssertionError:
                    #if assertion fails try to advance the model using mcmc to get
                    #out of local maximum
                    if i_tries < (max_retries - 1):
                        h.sample(500)
                        print

            assert recovery_ok, 'could not recover the true parameters'


def set_hddm_nodes_values(model, params_dict):
    """
    set hddm nodes values according to the params_dict
    """
    for (param_name, row) in model.iter_stochastics():
        if param_name in ('sv_trans', 'st_trans','t_trans','a_trans'):
            transform = np.log
            org_name = '%s' %  list(row['node'].children)[0].__name__
        elif param_name in ('sz_trans', 'z_trans'):
            transform = pm.logit
            if param_name == 'z_trans':
                org_name = 'z'
            else:
                org_name = 'sz'
        else:
            org_name = param_name
            transform = lambda x:x

        model.nodes_db.ix[param_name]['node'].value = transform(params_dict[org_name])


def test_ML_recovery_single_subject_from_random_starting_point():
    raise SkipTest("""Disabled. sometimes changes in sz and sv have little effect on logp,
     which makes their recovery impossible""")
    optimization_recovery_single_subject(repeats=5, seed=1, true_starting_point=False, optimization_method='ML')

def test_ML_recovery_single_subject_from_true_starting_point():
    optimization_recovery_single_subject(repeats=5, seed=1, true_starting_point=True, optimization_method='ML')

def test_chisquare_recovery_single_subject_from_true_starting_point():
    optimization_recovery_single_subject(repeats=5, seed=1, true_starting_point=True, optimization_method='chisquare')


if __name__=='__main__':
    print "Run nosetest.py"
