from __future__ import division
from copy import copy
import itertools
import glob
import os

import unittest
import pymc as pm
import numpy as np
import pandas as pd
import nose
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

    def test_HDDMTruncated_distributions(self):
        params = hddm.generate.gen_rand_params()
        data, params_subj = hddm.generate.gen_rand_data(subjs=4, params=params)
        m = hddm.HDDMTruncated(data)
        m.sample(self.iter, burn=self.burn)
        assert isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['v'], pm.Normal)
        assert isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['mu'], pm.Normal)
        assert isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['tau'], pm.Deterministic)
        assert isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['tau'].parents['x'], pm.Uniform)
        assert isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['a'], pm.TruncatedNormal)
        assert isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['a'].parents['mu'], pm.Uniform)
        assert isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['a'].parents['tau'], pm.Deterministic)
        assert isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['a'].parents['tau'].parents['x'], pm.Uniform)
        assert isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['t'], pm.TruncatedNormal)
        assert isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['t'].parents['tau'], pm.Deterministic)
        assert isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['t'].parents['tau'].parents['x'], pm.Uniform)

    def test_HDDM_distributions(self):
        params = hddm.generate.gen_rand_params()
        data, params_subj = hddm.generate.gen_rand_data(subjs=4, params=params)
        m = hddm.HDDM(data)
        m.sample(self.iter, burn=self.burn)
        assert isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['v'], pm.Normal)
        assert isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['mu'], pm.Normal)
        assert isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['tau'], pm.Deterministic)
        assert isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['tau'].parents['x'], pm.Uniform)
        assert isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['a'], pm.Deterministic)
        assert isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['a'].parents['x'], pm.Normal)
        assert isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['a'].parents['x'].parents['mu'], pm.Normal)
        assert isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['a'].parents['x'].parents['tau'], pm.Deterministic)
        assert isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['a'].parents['x'].parents['tau'].parents['x'], pm.Uniform)
        assert isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['t'], pm.Deterministic)
        assert isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['t'].parents['x'], pm.Normal)
        assert isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['t'].parents['x'].parents['mu'], pm.Normal)
        assert isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['t'].parents['x'].parents['tau'], pm.Deterministic)
        assert isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['t'].parents['x'].parents['tau'].parents['x'], pm.Uniform)


    def test_HDDMStimCoding(self):
        params_full, params = hddm.generate.gen_rand_params(cond_dict={'v': [-1, 1], 'z': [.8, .4]})
        data, params_subj = hddm.generate.gen_rand_data(params=params_full)
        m = hddm.HDDMStimCoding(data, stim_col='condition', split_param='v')
        m.sample(self.iter, burn=self.burn)
        assert isinstance(m.nodes_db.ix['wfpt(c0)']['node'].parents['v'], pm.PyMCObjects.Deterministic)
        assert isinstance(m.nodes_db.ix['wfpt(c0)']['node'].parents['v'].parents['self'], pm.Normal)
        assert isinstance(m.nodes_db.ix['wfpt(c1)']['node'].parents['v'], pm.Normal)

        m = hddm.HDDMStimCoding(data, stim_col='condition', split_param='z')
        m.sample(self.iter, burn=self.burn)
        assert isinstance(m.nodes_db.ix['wfpt(c0)']['node'].parents['z'], pm.PyMCObjects.Deterministic)
        assert isinstance(m.nodes_db.ix['wfpt(c0)']['node'].parents['z'].parents['a'], int)
        assert isinstance(m.nodes_db.ix['wfpt(c0)']['node'].parents['z'].parents['b'], pm.CommonDeterministics.InvLogit)
        assert isinstance(m.nodes_db.ix['wfpt(c1)']['node'].parents['z'], pm.CommonDeterministics.InvLogit)

    def test_HDDMRegressor(self):
        reg_func = lambda args, cols: args[0] + args[1]*cols[:,0]

        reg = {'func': reg_func, 'args':['v_slope','v_inter'], 'covariates': 'cov', 'outcome':'v'}

        params = hddm.generate.gen_rand_params()
        data, params_true = hddm.generate.gen_rand_data(params, size=500, subjs=5)
        data = pd.DataFrame(data)
        data['cov'] = 1.
        m = hddm.HDDMRegressor(data, regressor=reg)
        m.sample(self.iter, burn=self.burn)

        self.assertTrue(all(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['cols'][:,0] == 1))
        self.assertTrue(isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][0], pm.Normal))
        self.assertEqual(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][0].__name__, 'v_slope_subj.0')
        self.assertTrue(isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][1], pm.Normal))
        self.assertEqual(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][1].__name__, 'v_inter_subj.0')
        self.assertEqual(len(np.unique(m.nodes_db.ix['wfpt.0']['node'].parents['v'].value)), 1)

    def test_HDDMRegressor_two_covariates(self):
        reg_func = lambda args, cols: args[0] + args[1]*cols[:,0] + cols[:,1]

        reg = {'func': reg_func, 'args':['v_slope','v_inter'], 'covariates': ['cov1', 'cov2'], 'outcome':'v'}

        params = hddm.generate.gen_rand_params()
        data, params_true = hddm.generate.gen_rand_data(params, size=500, subjs=5)
        data = pd.DataFrame(data)
        data['cov1'] = 1.
        data['cov2'] = -1
        m = hddm.HDDMRegressor(data, regressor=reg)
        m.sample(self.iter, burn=self.burn)

        self.assertTrue(all(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['cols'][:,0] == 1))
        self.assertTrue(all(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['cols'][:,1] == -1))
        self.assertTrue(isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][0], pm.Normal))
        self.assertEqual(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][0].__name__, 'v_slope_subj.0')
        self.assertTrue(isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][1], pm.Normal))
        self.assertEqual(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][1].__name__, 'v_inter_subj.0')
        self.assertEqual(len(np.unique(m.nodes_db.ix['wfpt.0']['node'].parents['v'].value)), 1)

    def test_HDDMRegressorGroupOnly(self):
        reg_func = lambda args, cols: args[0] + args[1]*cols[:,0]

        reg = {'func': reg_func, 'args':['v_slope','v_inter'], 'covariates': 'cov', 'outcome':'v'}

        params = hddm.generate.gen_rand_params()
        data, params_true = hddm.generate.gen_rand_data(params, size=500, subjs=5)
        data = pd.DataFrame(data)
        data['cov'] = 1.
        m = hddm.HDDMRegressor(data, regressor=reg, group_only_nodes=['v_slope', 'v_inter'])
        m.sample(self.iter, burn=self.burn)

        self.assertTrue(all(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['cols'][:,0] == 1))
        self.assertTrue(isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][0], pm.Normal))
        self.assertEqual(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][0].__name__, 'v_slope')
        self.assertTrue(isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][1], pm.Normal))
        self.assertEqual(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][1].__name__, 'v_inter')
        self.assertEqual(len(np.unique(m.nodes_db.ix['wfpt.0']['node'].parents['v'].value)), 1)

def test_posterior_plots_breakdown():
    params = hddm.generate.gen_rand_params()
    data, params_subj = hddm.generate.gen_rand_data(params=params, subjs=5)
    m = hddm.HDDM(data)
    m.sample(200, burn=10)
    m.plot_posterior_predictive()
    m.plot_posterior_quantiles()
    m.plot_posteriors()
    # clean up
    for fname in ['a.png', 'a_var.png', 't.png', 't_var.png', 'v.png', 'v_var.png']:
        os.remove(fname)

def add_outliers(data, p_outlier):
    """add outliers to data. half of the outliers will be fast, and the rest will be slow
    Input:
        data - data
        p_outliers - probability of outliers
    """
    data = pd.DataFrame(data)

    #generating outliers
    n_outliers = int(len(data) * p_outlier)
    outliers = data[:n_outliers].copy()

    #fast outliers
    outliers.rt[:n_outliers//2] = np.random.rand(n_outliers//2) * (min(abs(data['rt'])) - 0.11)  + 0.11

    #slow outliers
    outliers.rt[n_outliers//2:] = np.random.rand(n_outliers - n_outliers//2) * 2 + max(abs(data['rt']))
    outliers.response = np.random.randint(0,2,n_outliers)

    #combine data with outliers
    data = pd.concat((data, outliers), ignore_index=True)
    return data

def optimization_recovery_single_subject(repeats=10, seed=1, true_starting_point=True,
                                         optimization_method='ML', max_retries=10):
    """
    recover parameters for single subjects model.
    The test does include recover of inter-variance variables since many times they have only small effect
    on logpable, which makes their recovery impossible.
    """

    #init
    include_sets = [set(['a','v','t']),
                    set(['a','v','t','z'])]

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
                    recovered_params = h.optimize(method='ML')
                elif optimization_method == 'chisquare':
                    recovered_params = h.optimize(method='chisquare', quantiles=np.linspace(0.05,0.95,10))
                elif optimization_method == 'gsquare':
                    recovered_params = h.optimize(method='gsquare', quantiles=np.linspace(0.05,0.95,10))
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
        try:
            model.nodes_db.ix[param_name]['node'].value = transform(params_dict[org_name])
        except KeyError:
            pass


def test_ML_recovery_single_subject_from_random_starting_point():
    optimization_recovery_single_subject(repeats=5, seed=1, true_starting_point=False, optimization_method='ML')

def test_ML_recovery_single_subject_from_true_starting_point():
    raise SkipTest()
    optimization_recovery_single_subject(repeats=5, seed=1, true_starting_point=True, optimization_method='ML')

def test_chisquare_recovery_single_subject_from_true_starting_point():
    raise SkipTest()
    optimization_recovery_single_subject(repeats=5, seed=1, true_starting_point=True, optimization_method='chisquare')

def test_gsquare_recovery_single_subject_from_true_starting_point():
    raise SkipTest()
    optimization_recovery_single_subject(repeats=5, seed=1, true_starting_point=True, optimization_method='gsquare')

@nose.tools.raises(AssertionError)
def test_recovery_with_outliers():
    optimization_recovery_single_subject(repeats=1, seed=1, optimization_method='ML', true_starting_point=False,
                                         call_add_outliers=True, max_retries=0)

def recovery_with_outliers(repeats=10, seed=1, random_p_outlier=True):
    """
    recover parameters with outliers for single subjects model.
    The test does include recover of inter-variance variables since many times they have only small effect
    on logpable, which makes their recovery impossible.
    """
    #init
    include_sets = [set(['a','v','t']),
                  set(['a','v','t','z'])]
    p_outlier = 0.05

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
            samples, _ = hddm.generate.gen_rand_data(cond_params, size=200)

            #get best recovered_params
            h = hddm.model.HDDM(samples, include=include, p_outlier=p_outlier, depends_on={'v':'condition'})
            best_params = h.optimize(method='ML')

            #add outliers
            samples = add_outliers(samples, p_outlier=p_outlier)

            #init model
            if random_p_outlier:
                h = hddm.model.HDDM(samples, include=include.union(['p_outlier']), depends_on={'v':'condition'})
            else:
                h = hddm.model.HDDM(samples, include=include, p_outlier=p_outlier, depends_on={'v':'condition'})

            #optimize
            recovered_params = h.optimize(method='ML')

            #compare results to true values
            index = ['best_estimate', 'current_estimate']
            df = pd.DataFrame([best_params, recovered_params], index=index, dtype=np.float).dropna(1)
            print df

            #assert
            np.testing.assert_allclose(df.values[0], df.values[1], atol=0.1)

def test_recovery_with_random_p_outlier():
    raise SkipTest()
    recovery_with_outliers(repeats=5, seed=1, random_p_outlier=True)

def test_recovery_with_fixed_p_outlier():
    raise SkipTest()
    recovery_with_outliers(repeats=5, seed=1, random_p_outlier=False)


if __name__=='__main__':
    print "Run nosetest.py"
