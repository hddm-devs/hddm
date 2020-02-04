
from copy import copy
import itertools
import kabuki
import os

import unittest
import pymc as pm
import numpy as np
import pandas as pd
import nose
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
        self.includes = [[], ['z'],['z', 'sv'],['z', 'st'],['z', 'sz'], ['z', 'sz','st'], ['z', 'sz','st','sv']]
        self.model_classes = [hddm.models.HDDMTruncated, hddm.models.HDDMTransformed,
                              hddm.models.HDDM]

        self.iter = 200
        self.burn = 10

    def runTest(self):
        return

    def test_HDDM(self):
        for include, model_class in itertools.product(self.includes, self.model_classes):
            params = hddm.generate.gen_rand_params(include=include)
            data, params_true = hddm.generate.gen_rand_data(params, size=10, subjs=1)
            model = model_class(data, include=include, bias='z' in include, is_group_model=False)
            model.map(runs=1)
            model.sample(self.iter, burn=self.burn)

        return model.mc

    def test_HDDM_split_std(self):
        data, _ = hddm.generate.gen_rand_data({'cond1': {'v':0, 'a':2, 't':.3, 'z': .5, 'sv': .1, 'st': .1, 'sz': .1},
                                               'cond2': {'v':0, 'a':2, 't':.3, 'z': .5, 'sv': .1, 'st': .1, 'sz': .1}})

        for param in ['a', 'v', 'z', 't']:
            model = hddm.HDDM(data, include='all', depends_on={param: 'condition'}, is_group_model=True, std_depends=True)
            idx = model.nodes_db.knode_name == param + '_std'
            self.assertEqual(len(model.nodes_db.node[idx]), 2)

        return model.mc

    def test_HDDM_group(self):
        for include, model_class in itertools.product(self.includes, self.model_classes):
            params = hddm.generate.gen_rand_params(include=include)
            data, params_true = hddm.generate.gen_rand_data(params, size=10, subjs=4)
            model = model_class(data, include=include, bias='z' in include, is_group_model=True)
            model.approximate_map()
            model.sample(self.iter, burn=self.burn)

        return model.mc

    def test_HDDM_group_only_group_nodes(self):
        group_only_nodes = ['v', 'a', 'z', 't']
        for nodes, model_class in itertools.product(group_only_nodes, self.model_classes):
            params = hddm.generate.gen_rand_params(include=nodes)
            data, params_true = hddm.generate.gen_rand_data(params, size=10, subjs=4)
            model = model_class(data, include=nodes, group_only_nodes=nodes, is_group_model=True)
            for node in nodes:
                self.assertNotIn(node+'_subj', model.nodes_db.index)
                self.assertIn(node, model.nodes_db.index)

    def test_HDDM_load_save(self):
        include = ['z', 'sz', 'st', 'sv']
        dbs = ['pickle', 'sqlite']
        params = hddm.generate.gen_rand_params(include=include)
        data, params_true = hddm.generate.gen_rand_data(params, size=10, subjs=2)
        data = pd.DataFrame(data)
        data['cov'] = 1.

        for db, model_class in itertools.product(dbs, self.model_classes):
            if model_class is hddm.models.HDDMRegressor:
                model = model_class(data, 'v ~ cov', include=include, is_group_model=True)
            else:
                model = model_class(data, include=include, is_group_model=True)
            model.sample(100, dbname='test.db', db=db)
            model.save('test.model')
            m_load = hddm.load('test.model')
            os.remove('test.db')
            os.remove('test.model')

    def test_HDDMTruncated_distributions(self):
        params = hddm.generate.gen_rand_params()
        data, params_subj = hddm.generate.gen_rand_data(subjs=4, params=params, size=10)
        m = hddm.HDDMTruncated(data)
        m.sample(self.iter, burn=self.burn)
        self.assertIsInstance(m.nodes_db.loc['wfpt.0']['node'].parents['v'], pm.Normal)
        self.assertIsInstance(m.nodes_db.loc['wfpt.0']['node'].parents['v'].parents['mu'], pm.Normal)
        self.assertIsInstance(m.nodes_db.loc['wfpt.0']['node'].parents['v'].parents['tau'], pm.Deterministic)
        self.assertIsInstance(m.nodes_db.loc['wfpt.0']['node'].parents['v'].parents['tau'].parents['x'], pm.Uniform)
        self.assertIsInstance(m.nodes_db.loc['wfpt.0']['node'].parents['a'], pm.TruncatedNormal)
        self.assertIsInstance(m.nodes_db.loc['wfpt.0']['node'].parents['a'].parents['mu'], pm.Uniform)
        self.assertIsInstance(m.nodes_db.loc['wfpt.0']['node'].parents['a'].parents['tau'], pm.Deterministic)
        self.assertIsInstance(m.nodes_db.loc['wfpt.0']['node'].parents['a'].parents['tau'].parents['x'], pm.Uniform)
        self.assertIsInstance(m.nodes_db.loc['wfpt.0']['node'].parents['t'], pm.TruncatedNormal)
        self.assertIsInstance(m.nodes_db.loc['wfpt.0']['node'].parents['t'].parents['tau'], pm.Deterministic)
        self.assertIsInstance(m.nodes_db.loc['wfpt.0']['node'].parents['t'].parents['tau'].parents['x'], pm.Uniform)

    def test_HDDM_distributions(self):
        params = hddm.generate.gen_rand_params()
        data, params_subj = hddm.generate.gen_rand_data(subjs=4, params=params, size=10)
        m = hddm.HDDM(data)
        m.sample(self.iter, burn=self.burn)
        self.assertIsInstance(m.nodes_db.loc['wfpt.0']['node'].parents['v'], pm.Normal)
        self.assertIsInstance(m.nodes_db.loc['wfpt.0']['node'].parents['v'].parents['mu'], pm.Normal)
        self.assertIsInstance(m.nodes_db.loc['wfpt.0']['node'].parents['v'].parents['tau'], pm.Deterministic)
        self.assertIsInstance(m.nodes_db.loc['wfpt.0']['node'].parents['v'].parents['tau'].parents['x'], pm.HalfNormal)
        self.assertIsInstance(m.nodes_db.loc['wfpt.0']['node'].parents['a'], pm.Gamma)
        self.assertIsInstance(m.nodes_db.loc['wfpt.0']['node'].parents['a'].parents['alpha'], pm.Deterministic)
        self.assertIsInstance(m.nodes_db.loc['wfpt.0']['node'].parents['a'].parents['alpha'].parents['x'], pm.Gamma)
        self.assertIsInstance(m.nodes_db.loc['wfpt.0']['node'].parents['a'].parents['beta'], pm.Deterministic)
        self.assertIsInstance(m.nodes_db.loc['wfpt.0']['node'].parents['a'].parents['beta'].parents['y'], pm.HalfNormal)
        self.assertIsInstance(m.nodes_db.loc['wfpt.0']['node'].parents['t'], pm.Gamma)
        self.assertIsInstance(m.nodes_db.loc['wfpt.0']['node'].parents['t'].parents['alpha'], pm.Deterministic)
        self.assertIsInstance(m.nodes_db.loc['wfpt.0']['node'].parents['t'].parents['alpha'].parents['x'], pm.Gamma)
        self.assertIsInstance(m.nodes_db.loc['wfpt.0']['node'].parents['t'].parents['beta'], pm.Deterministic)
        self.assertIsInstance(m.nodes_db.loc['wfpt.0']['node'].parents['t'].parents['beta'].parents['y'], pm.HalfNormal)


    def test_HDDMStimCoding(self):
        params_full, params = hddm.generate.gen_rand_params(cond_dict={'v': [-1, 1], 'z': [.8, .4]})
        data, params_subj = hddm.generate.gen_rand_data(params=params_full, size=10)
        m = hddm.HDDMStimCoding(data, stim_col='condition', split_param='v')
        m.sample(self.iter, burn=self.burn)
        self.assertIsInstance(m.nodes_db.loc['wfpt(c1)']['node'].parents['v'], pm.Normal)
        self.assertIsInstance(m.nodes_db.loc['wfpt(c0)']['node'].parents['v'], pm.PyMCObjects.Deterministic)
        self.assertIsInstance(m.nodes_db.loc['wfpt(c0)']['node'].parents['v'].parents['self'], pm.Normal)

        m = hddm.HDDMStimCoding(data, stim_col='condition', split_param='z')
        m.sample(self.iter, burn=self.burn)
        self.assertIsInstance(m.nodes_db.loc['wfpt(c1)']['node'].parents['z'], pm.CommonDeterministics.InvLogit)
        self.assertIsInstance(m.nodes_db.loc['wfpt(c0)']['node'].parents['z'], pm.PyMCObjects.Deterministic)
        self.assertIsInstance(m.nodes_db.loc['wfpt(c0)']['node'].parents['z'].parents['a'], int)
        self.assertIsInstance(m.nodes_db.loc['wfpt(c0)']['node'].parents['z'].parents['b'], pm.CommonDeterministics.InvLogit)

        m = hddm.HDDMStimCoding(data, stim_col='condition', split_param='v', drift_criterion=True)
        m.sample(self.iter, burn=self.burn)
        self.assertIsInstance(m.nodes_db.loc['wfpt(c1)']['node'].parents['v'], pm.PyMCObjects.Deterministic)
        self.assertEqual(m.nodes_db.loc['wfpt(c1)']['node'].parents['v'].parents['a'].__name__, 'v')
        self.assertEqual(m.nodes_db.loc['wfpt(c1)']['node'].parents['v'].parents['b'].__name__, 'dc')
        self.assertIsInstance(m.nodes_db.loc['wfpt(c0)']['node'].parents['v'], pm.PyMCObjects.Deterministic)
        self.assertIsInstance(m.nodes_db.loc['wfpt(c0)']['node'].parents['v'].parents['a'], pm.PyMCObjects.Deterministic)
        self.assertIsInstance(m.nodes_db.loc['wfpt(c0)']['node'].parents['v'].parents['b'], pm.Normal)
        self.assertIsInstance(m.nodes_db.loc['wfpt(c0)']['node'].parents['v'].parents['a'].parents['self'], pm.Normal)


class TestHDDMRegressor(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestHDDMRegressor, self).__init__(*args, **kwargs)

        self.iter = 200
        self.burn = 10

    def runTest(self):
        return

    def test_simple(self):
        params = hddm.generate.gen_rand_params()
        data, params_true = hddm.generate.gen_rand_data(params, size=10, subjs=4)
        data = pd.DataFrame(data)
        data['cov'] = 1.
        m = hddm.HDDMRegressor(data, 'v ~ cov', group_only_regressors=False)
        m.sample(self.iter, burn=self.burn)

        self.assertTrue(isinstance(m.nodes_db.loc['wfpt.0']['node'].parents['v'].parents['args'][0], pm.Normal))
        self.assertEqual(m.nodes_db.loc['wfpt.0']['node'].parents['v'].parents['args'][0].__name__, 'v_Intercept_subj.0')
        self.assertTrue(isinstance(m.nodes_db.loc['wfpt.0']['node'].parents['v'].parents['args'][1], pm.Normal))
        self.assertEqual(m.nodes_db.loc['wfpt.0']['node'].parents['v'].parents['args'][1].__name__, 'v_cov_subj.0')
        self.assertEqual(len(np.unique(m.nodes_db.loc['wfpt.0']['node'].parents['v'].value)), 1)

    def test_link_func_on_z(self):
        params = hddm.generate.gen_rand_params()
        data, params_true = hddm.generate.gen_rand_data(params, size=10, subjs=4)
        data = pd.DataFrame(data)
        data['cov'] = 1.
        link_func = lambda x: 1 / (1 + np.exp(-x))
        m = hddm.HDDMRegressor(data, {'model': 'z ~ cov', 'link_func':
                                      link_func}, group_only_regressors=False, include='z')
        m.sample(self.iter, burn=self.burn)

        self.assertIn('z', m.include)
        self.assertIn('z_Intercept', m.nodes_db.knode_name)
        self.assertTrue(isinstance(m.nodes_db.loc['wfpt.0']['node'].parents['z'].parents['args'][0].parents['ltheta'],
                                   pm.Normal))
        self.assertEqual(m.nodes_db.loc['wfpt.0']['node'].parents['z'].parents['args'][0].__name__, 'z_Intercept_subj.0')
        self.assertTrue(isinstance(m.nodes_db.loc['wfpt.0']['node'].parents['z'].parents['args'][1], pm.Normal))
        self.assertEqual(m.nodes_db.loc['wfpt.0']['node'].parents['z'].parents['args'][1].__name__, 'z_cov_subj.0')
        self.assertEqual(len(np.unique(m.nodes_db.loc['wfpt.0']['node'].parents['z'].value)), 1)
        self.assertEqual(m.model_descrs[0]['link_func'](2), link_func(2))

    def test_no_group(self):
        params = hddm.generate.gen_rand_params()
        data, params_true = hddm.generate.gen_rand_data(params, size=10, subjs=1)
        data['cov'] = 1.
        del data['subj_idx']
        m = hddm.HDDMRegressor(data, 'v ~ cov', group_only_regressors=False)
        m.sample(self.iter, burn=self.burn)

        self.assertTrue(isinstance(m.nodes_db.loc['wfpt']['node'].parents['v'].parents['args'][0], pm.Normal))
        self.assertEqual(m.nodes_db.loc['wfpt']['node'].parents['v'].parents['args'][0].__name__, 'v_Intercept')
        self.assertTrue(isinstance(m.nodes_db.loc['wfpt']['node'].parents['v'].parents['args'][1], pm.Normal))
        self.assertEqual(m.nodes_db.loc['wfpt']['node'].parents['v'].parents['args'][1].__name__, 'v_cov')
        self.assertEqual(len(np.unique(m.nodes_db.loc['wfpt']['node'].parents['v'].value)), 1)

    def test_two_covariates(self):
        params = hddm.generate.gen_rand_params()
        data, params_true = hddm.generate.gen_rand_data(params, size=10, subjs=4)
        data = pd.DataFrame(data)
        data['cov1'] = 1.
        data['cov2'] = -1
        m = hddm.HDDMRegressor(data, 'v ~ cov1 + cov2', group_only_regressors=False)
        m.sample(self.iter, burn=self.burn)

        self.assertTrue(isinstance(m.nodes_db.loc['wfpt.0']['node'].parents['v'].parents['args'][0], pm.Normal))
        self.assertEqual(m.nodes_db.loc['wfpt.0']['node'].parents['v'].parents['args'][0].__name__, 'v_Intercept_subj.0')
        self.assertTrue(isinstance(m.nodes_db.loc['wfpt.0']['node'].parents['v'].parents['args'][1], pm.Normal))
        self.assertEqual(m.nodes_db.loc['wfpt.0']['node'].parents['v'].parents['args'][1].__name__, 'v_cov1_subj.0')
        self.assertEqual(m.nodes_db.loc['wfpt.0']['node'].parents['v'].parents['args'][2].__name__, 'v_cov2_subj.0')
        self.assertEqual(len(np.unique(m.nodes_db.loc['wfpt.0']['node'].parents['v'].value)), 1)

    def test_two_regressors(self):
        params = hddm.generate.gen_rand_params()
        data, params_true = hddm.generate.gen_rand_data(params, size=10, subjs=4)
        data = pd.DataFrame(data)
        data['cov1'] = 1.
        data['cov2'] = -1
        m = hddm.HDDMRegressor(data, ['v ~ cov1', 'a ~ cov2'], group_only_regressors=False)
        m.sample(self.iter, burn=self.burn)

        self.assertTrue(isinstance(m.nodes_db.loc['wfpt.0']['node'].parents['v'].parents['args'][0], pm.Normal))
        self.assertTrue(isinstance(m.nodes_db.loc['wfpt.0']['node'].parents['a'].parents['args'][0], pm.Gamma))
        self.assertEqual(m.nodes_db.loc['wfpt.0']['node'].parents['v'].parents['args'][0].__name__, 'v_Intercept_subj.0')
        self.assertEqual(m.nodes_db.loc['wfpt.0']['node'].parents['a'].parents['args'][0].__name__, 'a_Intercept_subj.0')
        self.assertTrue(isinstance(m.nodes_db.loc['wfpt.0']['node'].parents['v'].parents['args'][1], pm.Normal))
        self.assertTrue(isinstance(m.nodes_db.loc['wfpt.0']['node'].parents['a'].parents['args'][1], pm.Normal))
        self.assertEqual(m.nodes_db.loc['wfpt.0']['node'].parents['v'].parents['args'][1].__name__, 'v_cov1_subj.0')
        self.assertEqual(m.nodes_db.loc['wfpt.0']['node'].parents['a'].parents['args'][1].__name__, 'a_cov2_subj.0')

    def test_group_only(self):
        params = hddm.generate.gen_rand_params()
        data, params_true = hddm.generate.gen_rand_data(params, size=10, subjs=4)
        data = pd.DataFrame(data)
        data['cov'] = 1.
        m = hddm.HDDMRegressor(data, 'v ~ cov', group_only_regressors=True)
        m.sample(self.iter, burn=self.burn)

        self.assertTrue(isinstance(m.nodes_db.loc['wfpt.0']['node'].parents['v'].parents['args'][0], pm.Normal))
        self.assertEqual(m.nodes_db.loc['wfpt.0']['node'].parents['v'].parents['args'][0].__name__, 'v_Intercept_subj.0')
        self.assertTrue(isinstance(m.nodes_db.loc['wfpt.0']['node'].parents['v'].parents['args'][1], pm.Normal))
        self.assertEqual(m.nodes_db.loc['wfpt.0']['node'].parents['v'].parents['args'][1].__name__, 'v_cov')
        self.assertEqual(len(np.unique(m.nodes_db.loc['wfpt.0']['node'].parents['v'].value)), 1)

    def test_group_only_depends(self):
        params = hddm.generate.gen_rand_params(cond_dict={'v': [1, 2, 3]})
        data, params_true = hddm.generate.gen_rand_data(params[0], size=10, subjs=4)
        data = pd.DataFrame(data)
        data['cov'] = 1.
        # Create one merged column
        data['condition2'] = 'merged'
        data.loc[data.condition == 'c1', 'condition2'] = 'single'
        self.assertRaises(AssertionError, hddm.HDDMRegressor, data, 'v ~ cov', depends_on={'v_Intercept': 'condition2'}, group_only_regressors=True)

    def test_contrast_coding(self):
        params = hddm.generate.gen_rand_params(cond_dict={'v': [1, 2, 3]})
        data, params_true = hddm.generate.gen_rand_data(params[0], size=10, subjs=4)
        data = pd.DataFrame(data)
        data['cov'] = 1.
        m = hddm.HDDMRegressor(data, 'v ~ cov * C(condition)',
                               depends_on={'a': 'condition'},
                               group_only_regressors=False)
        m.sample(self.iter, burn=self.burn)

        self.assertTrue(isinstance(m.nodes_db.loc['wfpt(c1).0']['node'].parents['v'].parents['args'][0], pm.Normal))
        self.assertEqual(m.nodes_db.loc['wfpt(c1).0']['node'].parents['v'].parents['args'][0].__name__, 'v_Intercept_subj.0')
        self.assertTrue(isinstance(m.nodes_db.loc['wfpt(c1).0']['node'].parents['v'].parents['args'][1], pm.Normal))
        self.assertEqual(m.nodes_db.loc['wfpt(c1).0']['node'].parents['v'].parents['args'][1].__name__, 'v_C(condition)[T.c1]_subj.0')
        self.assertEqual(len(np.unique(m.nodes_db.loc['wfpt(c1).0']['node'].parents['v'].value)), 1)

    def test_categorical_wo_intercept(self):
        params = hddm.generate.gen_rand_params(cond_dict={'a': [1, 2, 3]})
        data, params_true = hddm.generate.gen_rand_data(params[0], size=10, subjs=4)
        data = pd.DataFrame(data)
        data['cov'] = 1.
        m = hddm.HDDMRegressor(data, 'a ~ 0 + C(condition) * cov',
                               group_only_regressors=False)
        m.sample(self.iter, burn=self.burn)

        self.assertIsInstance(m.nodes_db.loc['a_C(condition)[c0]_subj.0']['node'], pm.Gamma)
        self.assertIsInstance(m.nodes_db.loc['a_C(condition)[c1]_subj.0']['node'], pm.Gamma)
        self.assertIsInstance(m.nodes_db.loc['a_C(condition)[c2]_subj.0']['node'], pm.Gamma)
        self.assertNotIsInstance(m.nodes_db.loc['a_C(condition)[T.c1]:cov_subj.0']['node'], pm.Gamma)
        self.assertNotIsInstance(m.nodes_db.loc['a_C(condition)[T.c2]:cov_subj.0']['node'], pm.Gamma)
        self.assertNotIsInstance(m.nodes_db.loc['a_cov_subj.0']['node'], pm.Gamma)

def test_posterior_plots_breakdown():
    params = hddm.generate.gen_rand_params()
    data, params_subj = hddm.generate.gen_rand_data(params=params, subjs=4)
    m = hddm.HDDM(data)
    m.sample(2000, burn=10)
    m.plot_posterior_predictive()
    m.plot_posterior_quantiles()
    m.plot_posteriors()
    ppc = hddm.utils.post_pred_gen(m, samples=10)
    hddm.utils.post_pred_stats(data, ppc)


class TestRecovery(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestRecovery, self).__init__(*args, **kwargs)
        self.iter = 2000
        self.burn = 20
        np.random.seed(1)

    def runTest(self):
        return

def extend_params(params):
    # Find list
    extend_param = [param for param, val in params.items() if isinstance(val, (list, tuple))]
    if len(extend_param) > 1:
        raise ValueError('Only one parameter can be extended')
    extend_param = extend_param[0]

    fixed_params = [param for param, val in params.items() if not isinstance(val, (list, tuple))]

    out_extended = {}
    out_merged = {k: params[k] for k in fixed_params}
    for i_cond, extend_val in enumerate(params[extend_param]):
         cond_params = {k: params[k] for k in fixed_params}
         cond_params[extend_param] = extend_val
         out_extended['cond%i' % i_cond] = cond_params

         out_merged['%s(cond%i)' % (extend_param, i_cond)] = extend_val

    return out_extended, out_merged

if __name__=='__main__':
    print("Run nosetest.py")
