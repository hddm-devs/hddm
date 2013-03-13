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

        self.iter = 40
        self.burn = 10

    def runTest(self):
        return

    def test_HDDM(self, assert_=False):
        includes = [[], ['z'],['z', 'sv'],['z', 'st'],['z', 'sz'], ['z', 'sz','st'], ['z', 'sz','st','sv']]
        model_classes = [hddm.models.HDDMTruncated, hddm.models.HDDM]
        for include, model_class in itertools.product(includes, model_classes):
            params = hddm.generate.gen_rand_params(include=include)
            data, params_true = hddm.generate.gen_rand_data(params, size=10, subjs=1)
            model = model_class(data, include=include, bias='z' in include, is_group_model=False)
            model.map(runs=1)
            model.sample(self.iter, burn=self.burn)

        return model.mc

    def test_HDDM_group(self, assert_=False):
        includes = [[], ['z'],['z', 'sv'],['z', 'st'],['z', 'sz'], ['z', 'sz','st'], ['z', 'sz','st','sv']]
        model_classes = [hddm.models.HDDMTruncated, hddm.models.HDDM]
        for include, model_class in itertools.product(includes, model_classes):
            params = hddm.generate.gen_rand_params(include=include)
            data, params_true = hddm.generate.gen_rand_data(params, size=10, subjs=4)
            model = model_class(data, include=include, bias='z' in include, is_group_model=True)
            model.approximate_map()
            model.sample(self.iter, burn=self.burn)

        return model.mc

    def test_HDDM_group_only_group_nodes(self, assert_=False):
        group_only_nodes = [[], ['z'], ['z', 'st'], ['v', 'a']]
        model_classes = [hddm.models.HDDMTruncated, hddm.models.HDDM]

        for nodes, model_class in itertools.product(group_only_nodes, model_classes):
            params = hddm.generate.gen_rand_params(include=nodes)
            data, params_true = hddm.generate.gen_rand_data(params, size=10, subjs=4)
            model = model_class(data, include=nodes, group_only_nodes=nodes, is_group_model=True)
            for node in nodes:
                self.assertNotIn(node+'_subj', model.nodes_db.index)
                self.assertIn(node, model.nodes_db.index)

    def test_HDDM_load_save(self, assert_=False):
        include = ['z', 'sz','st','sv']
        dbs = ['pickle', 'sqlite']
        model_classes = [hddm.models.HDDMTruncated, hddm.models.HDDM, hddm.models.HDDMRegressor]
        reg_func = lambda args, cols: args[0] + args[1]*cols['cov']
        reg = {'func': reg_func, 'args':['v_slope','v_inter'], 'covariates': 'cov', 'outcome':'v'}
        params = hddm.generate.gen_rand_params(include=include)
        data, params_true = hddm.generate.gen_rand_data(params, size=10, subjs=2)
        data = pd.DataFrame(data)
        data['cov'] = 1.

        for db, model_class in itertools.product(dbs, model_classes):
            if model_class is hddm.models.HDDMRegressor:
                model = model_class(data, regressor=reg, include=include, is_group_model=True)
            else:
                model = model_class(data, include=include, is_group_model=True)
            model.sample(20, dbname='test.db', db=db)
            model.save('test.model')
            m_load = hddm.load('test.model')
            os.remove('test.db')
            os.remove('test.model')

    def test_HDDMTruncated_distributions(self):
        params = hddm.generate.gen_rand_params()
        data, params_subj = hddm.generate.gen_rand_data(subjs=4, params=params, size=10)
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
        data, params_subj = hddm.generate.gen_rand_data(subjs=4, params=params, size=10)
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
        data, params_subj = hddm.generate.gen_rand_data(params=params_full, size=10)
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


class TestHDDMRegressor(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestHDDMRegressor, self).__init__(*args, **kwargs)

        self.iter = 40
        self.burn = 10

    def runTest(self):
        return

    def test_simple(self):
        params = hddm.generate.gen_rand_params()
        data, params_true = hddm.generate.gen_rand_data(params, size=10, subjs=4)
        data = pd.DataFrame(data)
        data['cov'] = 1.
        m = hddm.HDDMRegressor(data, 'v ~ cov')
        m.sample(self.iter, burn=self.burn)

        self.assertTrue(isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][0], pm.Normal))
        self.assertEqual(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][0].__name__, 'v_Intercept_subj.0')
        self.assertTrue(isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][1], pm.Normal))
        self.assertEqual(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][1].__name__, 'v_cov_subj.0')
        self.assertEqual(len(np.unique(m.nodes_db.ix['wfpt.0']['node'].parents['v'].value)), 1)

    def test_include_z(self):
        params = hddm.generate.gen_rand_params()
        data, params_true = hddm.generate.gen_rand_data(params, size=10, subjs=4)
        data = pd.DataFrame(data)
        data['cov'] = 1.
        m = hddm.HDDMRegressor(data, 'z ~ cov', include='z')
        m.sample(self.iter, burn=self.burn)

        self.assertIn('z', m.include)
        self.assertIn('z', m.nodes_db.knode_name)
        self.assertTrue(isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][0], pm.Normal))
        self.assertEqual(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][0].__name__, 'v_Intercept_subj.0')
        self.assertTrue(isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][1], pm.Normal))
        self.assertEqual(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][1].__name__, 'v_cov_subj.0')
        self.assertEqual(len(np.unique(m.nodes_db.ix['wfpt.0']['node'].parents['v'].value)), 1)

    def test_no_group(self):
        params = hddm.generate.gen_rand_params()
        data, params_true = hddm.generate.gen_rand_data(params, size=10, subjs=1)
        data['cov'] = 1.
        del data['subj_idx']
        m = hddm.HDDMRegressor(data, 'v ~ cov', is_group_model=False, depends_on={})
        m.sample(self.iter, burn=self.burn)
        print m.nodes_db.index

        self.assertTrue(isinstance(m.nodes_db.ix['wfpt']['node'].parents['v'].parents['args'][0], pm.Normal))
        self.assertEqual(m.nodes_db.ix['wfpt']['node'].parents['v'].parents['args'][0].__name__, 'v_Intercept')
        self.assertTrue(isinstance(m.nodes_db.ix['wfpt']['node'].parents['v'].parents['args'][1], pm.Normal))
        self.assertEqual(m.nodes_db.ix['wfpt']['node'].parents['v'].parents['args'][1].__name__, 'v_cov')
        self.assertEqual(len(np.unique(m.nodes_db.ix['wfpt']['node'].parents['v'].value)), 1)

    def test_two_covariates(self):
        params = hddm.generate.gen_rand_params()
        data, params_true = hddm.generate.gen_rand_data(params, size=10, subjs=4)
        data = pd.DataFrame(data)
        data['cov1'] = 1.
        data['cov2'] = -1
        m = hddm.HDDMRegressor(data, 'v ~ cov1 + cov2')
        m.sample(self.iter, burn=self.burn)

        self.assertTrue(isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][0], pm.Normal))
        self.assertEqual(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][0].__name__, 'v_Intercept_subj.0')
        self.assertTrue(isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][1], pm.Normal))
        self.assertEqual(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][1].__name__, 'v_cov1_subj.0')
        self.assertEqual(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][2].__name__, 'v_cov2_subj.0')
        self.assertEqual(len(np.unique(m.nodes_db.ix['wfpt.0']['node'].parents['v'].value)), 1)

    def test_two_regressors(self):
        params = hddm.generate.gen_rand_params()
        data, params_true = hddm.generate.gen_rand_data(params, size=10, subjs=4)
        data = pd.DataFrame(data)
        data['cov1'] = 1.
        data['cov2'] = -1
        m = hddm.HDDMRegressor(data, ['v ~ cov1', 'a ~ cov2'])
        m.sample(self.iter, burn=self.burn)

        self.assertTrue(isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][0], pm.Normal))
        self.assertTrue(isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['a'].parents['args'][0], pm.Normal))
        self.assertEqual(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][0].__name__, 'v_Intercept_subj.0')
        self.assertEqual(m.nodes_db.ix['wfpt.0']['node'].parents['a'].parents['args'][0].__name__, 'a_Intercept_subj.0')
        self.assertTrue(isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][1], pm.Normal))
        self.assertTrue(isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['a'].parents['args'][1], pm.Normal))
        self.assertEqual(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][1].__name__, 'v_cov1_subj.0')
        self.assertEqual(m.nodes_db.ix['wfpt.0']['node'].parents['a'].parents['args'][1].__name__, 'a_cov2_subj.0')

    def test_group_only(self):
        params = hddm.generate.gen_rand_params()
        data, params_true = hddm.generate.gen_rand_data(params, size=10, subjs=4)
        data = pd.DataFrame(data)
        data['cov'] = 1.
        m = hddm.HDDMRegressor(data, 'v ~ cov', group_only_regressors=True)
        m.sample(self.iter, burn=self.burn)

        self.assertTrue(isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][0], pm.Normal))
        self.assertEqual(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][0].__name__, 'v_Intercept')
        self.assertTrue(isinstance(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][1], pm.Normal))
        self.assertEqual(m.nodes_db.ix['wfpt.0']['node'].parents['v'].parents['args'][1].__name__, 'v_cov')
        self.assertEqual(len(np.unique(m.nodes_db.ix['wfpt.0']['node'].parents['v'].value)), 1)

    def test_group_only_depends(self):
        params = hddm.generate.gen_rand_params(cond_dict={'v': [1, 2, 3]})
        data, params_true = hddm.generate.gen_rand_data(params[0], size=10, subjs=4)
        data = pd.DataFrame(data)
        data['cov'] = 1.
        # Create one merged column
        data['condition2'] = 'merged'
        data[data.condition == 'c1']['condition2'] = 'single'
        self.assertRaises(AssertionError, hddm.HDDMRegressor, data, 'v ~ cov', depends_on={'v_Intercept': 'condition2'}, group_only_regressors=True)

    def test_contrast_coding(self):
        params = hddm.generate.gen_rand_params(cond_dict={'v': [1, 2, 3]})
        data, params_true = hddm.generate.gen_rand_data(params[0], size=10, subjs=4)
        data = pd.DataFrame(data)
        data['cov'] = 1.
        m = hddm.HDDMRegressor(data, 'v ~ cov * C(condition)',
                               depends_on={'a': 'condition'})
        m.sample(self.iter, burn=self.burn)

        self.assertTrue(isinstance(m.nodes_db.ix['wfpt(c1).0']['node'].parents['v'].parents['args'][0], pm.Normal))
        self.assertEqual(m.nodes_db.ix['wfpt(c1).0']['node'].parents['v'].parents['args'][0].__name__, 'v_Intercept_subj.0')
        self.assertTrue(isinstance(m.nodes_db.ix['wfpt(c1).0']['node'].parents['v'].parents['args'][1], pm.Normal))
        self.assertEqual(m.nodes_db.ix['wfpt(c1).0']['node'].parents['v'].parents['args'][1].__name__, 'v_C(condition)[T.c1]_subj.0')
        self.assertEqual(len(np.unique(m.nodes_db.ix['wfpt(c1).0']['node'].parents['v'].value)), 1)


def test_posterior_plots_breakdown():
    params = hddm.generate.gen_rand_params()
    data, params_subj = hddm.generate.gen_rand_data(params=params, subjs=4)
    m = hddm.HDDM(data)
    m.sample(100, burn=10)
    m.plot_posterior_predictive()
    m.plot_posterior_quantiles()
    m.plot_posteriors()

if __name__=='__main__':
    print "Run nosetest.py"
