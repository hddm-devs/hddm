
from copy import copy
import itertools
import glob
import os

import unittest
import pymc as pm
import numpy as np
import pandas as pd
import nose
from nose import SkipTest

import hddm
from hddm.diag import check_model


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
                                         optimization_method='ML'):
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
            print("*** the true parameters ***")
            print(merged_params)

            #generate samples
            samples, _ = hddm.generate.gen_rand_data(cond_params, size=10000)

            h = hddm.models.HDDMTruncated(samples, include=include, depends_on={'v':'condition'})

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
            print(df)

            #assert
            np.testing.assert_allclose(df.values[0], df.values[1], atol=0.1)


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
    optimization_recovery_single_subject(repeats=5, seed=1, true_starting_point=True, optimization_method='ML')

def test_chisquare_recovery_single_subject_from_true_starting_point():
    optimization_recovery_single_subject(repeats=5, seed=1, true_starting_point=True, optimization_method='chisquare')

def test_gsquare_recovery_single_subject_from_true_starting_point():
    optimization_recovery_single_subject(repeats=5, seed=1, true_starting_point=True, optimization_method='gsquare')

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
            print("*** the true parameters ***")
            print(merged_params)

            #generate samples
            samples, _ = hddm.generate.gen_rand_data(cond_params, size=200)

            #get best recovered_params
            h = hddm.models.HDDMTruncated(samples, include=include, p_outlier=p_outlier, depends_on={'v':'condition'})
            best_params = h.optimize(method='ML')

            #add outliers
            samples = add_outliers(samples, p_outlier=p_outlier)

            #init model
            if random_p_outlier is None:
                h = hddm.models.HDDM(samples, include=include, depends_on={'v':'condition'})
            elif random_p_outlier:
                h = hddm.models.HDDM(samples, include=include.union(['p_outlier']), depends_on={'v':'condition'})
            else:
                h = hddm.models.HDDM(samples, include=include, p_outlier=p_outlier, depends_on={'v':'condition'})

            #optimize
            recovered_params = h.optimize(method='ML')

            #compare results to true values
            index = ['best_estimate', 'current_estimate']
            df = pd.DataFrame([best_params, recovered_params], index=index, dtype=np.float).dropna(1)
            print(df)

            #assert
            np.testing.assert_allclose(df.values[0], df.values[1], atol=0.15)

@nose.tools.raises(AssertionError)
def test_recovery_with_outliers():
    """test recovery of data with outliers without modeling them (should fail)"""
    recovery_with_outliers(repeats=5, seed=1, random_p_outlier=None)

def test_recovery_with_random_p_outlier():
    """test for recovery with o_outliers as random variable"""
    recovery_with_outliers(repeats=5, seed=1, random_p_outlier=True)

def test_recovery_with_fixed_p_outlier():
    """test for recovery with o_outliers as a fixed value"""
    recovery_with_outliers(repeats=5, seed=1, random_p_outlier=False)
