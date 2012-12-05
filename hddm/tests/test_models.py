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

    model = hddm.models.HDDMTruncated(data, depends_on={param:['condition']}, is_group_model=subj)

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
        model_classes = [hddm.models.HDDMTruncated, hddm.models.HDDM]
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
        model_classes = [hddm.models.HDDMTruncated, hddm.models.HDDM]
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
        model_classes = [hddm.models.HDDMTruncated, hddm.models.HDDM]

        for nodes, model_class in itertools.product(group_only_nodes, model_classes):
            params = hddm.generate.gen_rand_params(include=nodes)
            data, params_true = hddm.generate.gen_rand_data(params, size=500, subjs=5)
            model = model_class(data, include=nodes, group_only_nodes=nodes, is_group_model=True)
            for node in nodes:
                self.assertNotIn(node+'_subj', model.nodes_db.index)
                self.assertIn(node, model.nodes_db.index)

    def test_HDDM_load_save(self, assert_=False):
        dbs = ['pickle', 'sqlite']
        model_classes = [hddm.models.HDDMTruncated, hddm.models.HDDM]
        for db, model_class in itertools.product(dbs, model_classes):
            include = ['z', 'sz','st','sv']
            params = hddm.generate.gen_rand_params(include=include)
            data, params_true = hddm.generate.gen_rand_data(params, size=10, subjs=2)
            model = model_class(data, include=include, is_group_model=True)
            model.sample(20, dbname='test.db', db=db)
            model.save('test.model')
            m_load = hddm.load('test.model')
            os.remove('test.db')
            os.remove('test.model')

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
        assert isinstance(m.nodes_db.ix['wfpt(c0)']['node'].parents['v'], pm.Normal)
        assert isinstance(m.nodes_db.ix['wfpt(c1)']['node'].parents['v'], pm.PyMCObjects.Deterministic)
        assert isinstance(m.nodes_db.ix['wfpt(c1)']['node'].parents['v'].parents['self'], pm.Normal)

        m = hddm.HDDMStimCoding(data, stim_col='condition', split_param='z')
        m.sample(self.iter, burn=self.burn)
        assert isinstance(m.nodes_db.ix['wfpt(c0)']['node'].parents['z'], pm.CommonDeterministics.InvLogit)
        assert isinstance(m.nodes_db.ix['wfpt(c1)']['node'].parents['z'], pm.PyMCObjects.Deterministic)
        assert isinstance(m.nodes_db.ix['wfpt(c1)']['node'].parents['z'].parents['a'], int)
        assert isinstance(m.nodes_db.ix['wfpt(c1)']['node'].parents['z'].parents['b'], pm.CommonDeterministics.InvLogit)

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

    def test_HDDMRegressor_no_group(self):
        reg_func = lambda args, cols: args[0] + args[1]*cols[:,0]

        reg = {'func': reg_func, 'args':['v_slope','v_inter'], 'covariates': 'cov', 'outcome':'v'}

        params = hddm.generate.gen_rand_params()
        data, params_true = hddm.generate.gen_rand_data(params, size=500, subjs=1)
        data = pd.DataFrame(data)
        data['cov'] = 1.
        del data['subj_idx']
        m = hddm.HDDMRegressor(data, regressor=reg, is_group_model=False, depends_on={})
        m.sample(self.iter, burn=self.burn)
        print m.nodes_db.index

        self.assertTrue(all(m.nodes_db.ix['wfpt']['node'].parents['v'].parents['cols'][:,0] == 1))
        self.assertTrue(isinstance(m.nodes_db.ix['wfpt']['node'].parents['v'].parents['args'][0], pm.Normal))
        self.assertEqual(m.nodes_db.ix['wfpt']['node'].parents['v'].parents['args'][0].__name__, 'v_slope')
        self.assertTrue(isinstance(m.nodes_db.ix['wfpt']['node'].parents['v'].parents['args'][1], pm.Normal))
        self.assertEqual(m.nodes_db.ix['wfpt']['node'].parents['v'].parents['args'][1].__name__, 'v_inter')
        self.assertEqual(len(np.unique(m.nodes_db.ix['wfpt']['node'].parents['v'].value)), 1)

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

    def test_HDDMRegressor_two_regressors(self):
        reg_func1 = lambda args, cols: args[0] + args[1]*cols[:,0]
        reg1 = {'func': reg_func1, 'args':['v_slope','v_inter'], 'covariates': 'cov1', 'outcome':'v'}

        reg_func2 = lambda args, cols: args[0] + args[1]*cols[:,0]
        reg2 = {'func': reg_func2, 'args':['a_slope','a_inter'], 'covariates': 'cov2', 'outcome':'a'}

        params = hddm.generate.gen_rand_params()
        data, params_true = hddm.generate.gen_rand_data(params, size=500, subjs=5)
        data = pd.DataFrame(data)
        data['cov1'] = 1.
        data['cov2'] = -1
        m = hddm.HDDMRegressor(data, regressor=[reg1, reg2])
        m.sample(self.iter, burn=self.burn)

        self.assertTrue(all(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['cols'][:,0] == 1))
        self.assertTrue(all(m.nodes_db.ix['wfpt.0']['node'].parents['a'].parents['cols'][:,0] == -1))
        self.assertTrue(isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][0], pm.Normal))
        self.assertTrue(isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['a'].parents['args'][0], pm.Normal))
        self.assertEqual(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][0].__name__, 'v_slope_subj.0')
        self.assertEqual(m.nodes_db.ix['wfpt.0']['node'].parents['a'].parents['args'][0].__name__, 'a_slope_subj.0')
        self.assertTrue(isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][1], pm.Normal))
        self.assertTrue(isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['a'].parents['args'][1], pm.Normal))
        self.assertEqual(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][1].__name__, 'v_inter_subj.0')
        self.assertEqual(m.nodes_db.ix['wfpt.0']['node'].parents['a'].parents['args'][1].__name__, 'a_inter_subj.0')

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

if __name__=='__main__':
    print "Run nosetest.py"
