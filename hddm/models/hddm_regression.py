from copy import deepcopy
import numpy as np
import pymc as pm
import pandas as pd

import hddm
from hddm_gamma import HDDMGamma
import kabuki
from kabuki import Knode
from kabuki.utils import stochastic_from_dist

def generate_wfpt_reg_stochastic_class(wiener_params=None, sampling_method='cdf', cdf_range=(-5,5), sampling_dt=1e-4):

    #set wiener_params
    if wiener_params is None:
        wiener_params = {'err': 1e-4, 'n_st':2, 'n_sz':2,
                         'use_adaptive':1,
                         'simps_err':1e-3,
                         'w_outlier': 0.1}
    wp = wiener_params

    def wiener_multi_like(value, v, sv, a, z, sz, t, st, reg_outcomes, p_outlier=0):
        """Log-likelihood for the full DDM using the interpolation method"""
        params = {'v': v, 'sv': sv, 'a': a, 'z': z, 'sz': sz, 't': t, 'st': st}
        for reg_outcome in reg_outcomes:
            params[reg_outcome] = params[reg_outcome].ix[value['rt'].index].values
        return hddm.wfpt.wiener_like_multi(value['rt'].values,
                                           params['v'], params['sv'], params['a'], params['z'],
                                           params['sz'], params['t'], params['st'], 1e-4,
                                           reg_outcomes,
                                           p_outlier=p_outlier)


    def random(v, sv, a, z, sz, t, st, reg_outcomes, p_outlier, size=None):

        param_dict = {'v':v, 'z':z, 't':t, 'a':a, 'sz':sz, 'sv':sv, 'st':st}
        sampled_rts = np.empty(size)

        for i_sample in xrange(len(sampled_rts)):
            #get current params
            for p in reg_outcomes:
                param_dict[p] = locals()[p][i_sample]
            #sample
            sampled_rts[i_sample] = hddm.generate.gen_rts(param_dict,
                                                          method=sampling_method,
                                                          samples=1,
                                                          dt=sampling_dt)
        return sampled_rts

    return stochastic_from_dist('wfpt_reg', wiener_multi_like,
                                random=random)


wfpt_reg_like = generate_wfpt_reg_stochastic_class(sampling_method='drift')


################################################################################################

class KnodeRegress(kabuki.hierarchical.Knode):
    def create_node(self, name, kwargs, data):
        reg = kwargs['regressor']
        # order parents according to user-supplied args
        args = []
        for arg in reg['args']:
            for parent_name, parent in kwargs['parents'].iteritems():
                if parent_name.startswith(arg):
                    args.append(parent)

        cols = data if reg['covariates'] is None else data[reg['covariates']]
        parents = {'args': args, 'cols': cols}

        return self.pymc_node(reg['func'], kwargs['doc'], name, parents=parents)


class KnodeRegressPatsy(kabuki.hierarchical.Knode):
    def create_node(self, name, kwargs, data):
        from patsy import dmatrix
        reg = kwargs['regressor']
        # order parents according to user-supplied args
        args = []
        for arg in reg['args']:
            for parent_name, parent in kwargs['parents'].iteritems():
                if parent_name.startswith(arg):
                    args.append(parent)

        cols = data if reg['covariates'] is None else data[reg['covariates']]
        parents = {'args': args}

        def func(args, patsy_func=reg['func']):
            # convert parents to matrix
            params = np.matrix(args)
            # Apply design matrix to input data
            predictor = (dmatrix(patsy_func, cols) * params).sum(axis=1)
            return pd.DataFrame(predictor, index=cols.index)

        return self.pymc_node(func, kwargs['doc'], name, parents=parents)

class HDDMRegressor(HDDMGamma):
    """HDDMRegressor allows estimation of trial-by-trial influences of
    a covariate (e.g. a brain measure like fMRI) onto DDM parameters.

    For example, if your prediction is that activity of a particular
    brain area has a linear correlation with drift-rate, you could
    specify the following regression model (make sure you have a column
    with the brain activity in your data, in our example we name this
    column 'BOLD'):

    ::

        # Define regression function (linear in this case)
        reg_func = lambda args, cols: args[0] + args[1]*cols['BOLD']

        # Define regression descriptor
        # regression function to use (func, defined above)
        # args: parameter names (passed to reg_func; v_slope->args[0],
        #                                            v_inter->args[1])
        # outcome: DDM parameter that will be replaced by trial-by-trial
        #          regressor values (drift-rate v in this case)
        reg = {'func': reg_func,
               'args': ['v_inter','v_slope'],
               'outcome': 'v'}

        # construct regression model. Second argument must be the
        # regression descriptor. This model will have new parameters defined
        # in args above, these can be used in depends_on like any other
        # parameter.
        m = hddm.HDDMRegressor(data, reg, depends_on={'v_slope':'trial_type'})

    """
    def __init__(self, data, regressor=None, **kwargs):
        """Hierarchical Drift Diffusion Model with regressors
        """
        #create self.regressor and self.reg_outcome
        regressor = deepcopy(regressor)
        if isinstance(regressor, dict):
            regressor = [regressor]

        use_patsy = isinstance(regressor[0]['func'], str)
        self.knode_regress = KnodeRegressPatsy if use_patsy else KnodeRegress

        self.reg_outcomes = set() # holds all the parameters that are going to modeled as outcome
        self.reg_covariates = set()
        for reg in regressor:
            if isinstance(reg['args'], str):
                reg['args'] = [reg['args']]
            if 'covariates' in reg and isinstance(reg['covariates'], str):
                reg['covariates'] = [reg['covariates']]
            else:
                reg['covariates'] = None
            self.reg_outcomes.add(reg['outcome'])

        self.regressor = regressor

        #set wfpt_reg
        self.wfpt_reg_class = deepcopy(wfpt_reg_like)

        super(HDDMRegressor, self).__init__(data, **kwargs)

    def __getstate__(self):
        d = super(HDDMRegressor, self).__getstate__()
        del d['wfpt_reg_class']
        print 'WARNING: The regression function can not be saved. Still saving but the loaded model will not be functional!'
        for reg in d['regressor']:
            del reg['func']

        return d

    def __setstate__(self, d):
        d['wfpt_reg_class'] = deepcopy(wfpt_reg_like)
        for reg in d['regressor']:
            reg['func'] = lambda args, cols: cols
        print "WARNING: Can not load regression function. Replacing with a dummy function so do not try to sample from this model."
        super(HDDMRegressor, self).__setstate__(d)

    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)
        return Knode(self.wfpt_reg_class, 'wfpt', observed=True,
                     col_name=['rt'],
                     reg_outcomes=self.reg_outcomes, **wfpt_parents)

    def _create_stochastic_knodes(self, include):
        # Create all stochastic knodes except for the ones that we want to replace
        # with regressors.
        knodes = super(HDDMRegressor, self)._create_stochastic_knodes(include.difference(self.reg_outcomes))

        #create regressor params
        for i_reg, reg in enumerate(self.regressor):
            reg_parents = {}
            for arg in reg['args']:
                reg_family = self.create_family_normal(arg, value=0)
                reg_parents[arg] = reg_family['%s_bottom' % arg]
                if reg not in self.group_only_nodes:
                    reg_family['%s_subj_reg' % arg] = reg_family.pop('%s_bottom' % arg)
                knodes.update(reg_family)

            reg_knode = self.knode_regress(pm.Deterministic, "%s_reg" % reg['outcome'],
                                           regressor=reg,
                                           col_name=reg['covariates'],
                                           subj=self.is_group_model,
                                           plot=False,
                                           trace=False,
                                           hidden=True,
                                           **reg_parents)

            knodes['%s_bottom' % reg['outcome']] = reg_knode

        return knodes
