from copy import deepcopy
import numpy as np
from scipy import stats
import pymc as pm

import hddm
from hddm_transformed import HDDM
import kabuki
from kabuki import Knode
from kabuki.distributions import scipy_stochastic

class wfpt_regress_gen(stats.distributions.rv_continuous):

    wiener_params = {'err': 1e-4, 'nT':2, 'nsz':2,
                                  'use_adaptive':1,
                                  'simps_err':1e-3}
    wp = wiener_params
    sampling_method = 'drift'
    dt=1e-4

    def _argcheck(self, *args):
        return True

    def _logp(self, x, v, sv, a, z, sz, t, st, reg_outcomes, p_outlier):
        """Log-likelihood for the full DDM using the interpolation method"""
        return hddm.wfpt.wiener_like_multi(x, v, sv, a, z, sz, t, st, .001, reg_outcomes, p_outlier=p_outlier)

    def _pdf(self, x, v, sv, a, z, sz, t, st, reg_outcomes, p_outlier):
        raise NotImplementedError

    def _rvs(self, v, sv, a, z, sz, t, st, reg_outcomes, p_outlier):
        param_dict = {'v':v, 'z':z, 't':t, 'a':a, 'sz':sz, 'sv':sv, 'st':st}
        sampled_rts = np.empty(self._size)

        for i_sample in xrange(self._size):
            #get current params
            for p in reg_outcomes:
                param_dict[p] = locals()[p][i_sample]
            #sample
            sampled_rts[i_sample] = hddm.generate.gen_rts(param_dict, method=self.sampling_method,
                                                   samples=1, dt=self.dt)
        return sampled_rts

    def random(self, v=1., sv=0., a=2, z=.5, sz=.1, t=.3, st=.1, reg_outcomes=None, size=None, p_outlier=0):
        self._size = len(locals()[reg_outcomes[0]])
        return self._rvs(v, sv, a, z, sz, t, st, reg_outcomes)


wfpt_reg_like = scipy_stochastic(wfpt_regress_gen, name='wfpt_reg',
                                 longname="""Wiener first passage time with regressors likelihood function""",
                                 extradoc="")



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

        parents = {'args': args, 'cols': data[reg['covariates']].values}
        return self.pymc_node(reg['func'], kwargs['doc'], name, parents=parents)


class HDDMRegressor(HDDM):
    """HDDMRegressor allows estimation of trial-by-trial influences of
    a covariate (e.g. a brain measure like fMRI) onto DDM parameters.

    For example, if your prediction is that activity of a particular
    brain area has a linear correlation with drift-rate, you could
    specify the following regression model (make sure you have a column
    with the brain activity in your data, in our example we name this
    column 'BOLD'):

    ::

        # Define regression function (linear in this case)
        reg_func = lambda args, cols: args[0] + args[1]*cols[:,0]

        # Define regression descriptor
        # regression function to use (func, defined above)
        # args: parameter names (passed to reg_func; v_slope->args[0],
        #                                            v_inter->args[1])
        # covariates: data column to use as the covariate
        #             (in this example, expects a column named
        #             BOLD in the data)
        # outcome: DDM parameter that will be replaced by trial-by-trial
        #          regressor values (drift-rate v in this case)
        reg = {'func': reg_func,
               'args': ['v_inter','v_slope'],
               'covariates': 'BOLD',
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

        self.reg_outcomes = set() # holds all the parameters that are going to modeled as outcome
        for reg in regressor:
            if isinstance(reg['args'], str):
                reg['args'] = [reg['args']]
            if isinstance(reg['covariates'], str):
                reg['covariates'] = [reg['covariates']]
            self.reg_outcomes.add(reg['outcome'])

        self.regressor = regressor

        #set wfpt_reg
        self.wfpt_reg_class = deepcopy(wfpt_reg_like)
        self.wfpt_reg_class.rv.wiener_params

        super(HDDMRegressor, self).__init__(data, **kwargs)

    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)
        return Knode(self.wfpt_reg_class, 'wfpt', observed=True,
                     col_name='rt', reg_outcomes=self.reg_outcomes, **wfpt_parents)

    def _create_stochastic_knodes(self, include):
        # Create all stochastic knodes except for the ones that we want to replace
        # with regressors.
        knodes = super(HDDMRegressor, self)._create_stochastic_knodes(include.difference(self.reg_outcomes))

        #create regressor params
        for i_reg, reg in enumerate(self.regressor):
            reg_parents = {}

            for arg in reg['args']:
                # Create family for each regressor
                reg_family = self.create_family_normal(arg, value=0)
                # Store parents to pass to reg node below
                reg_parents[arg] = reg_family['%s_bottom' % arg]
                if reg not in self.group_only_nodes and self.is_group_model:
                    # Rename bottom nodes to subj nodes (reg nodes will be bottom)
                    reg_family['%s_subj_reg' % arg] = reg_family.pop('%s_bottom' % arg)
                knodes.update(reg_family)

            reg_knode = KnodeRegress(pm.Deterministic, "%s_reg" % reg['outcome'],
                                     regressor=reg,
                                     col_name=reg['covariates'],
                                     subj=self.is_group_model,
                                     plot=False,
                                     trace=False,
                                     hidden=True,
                                     **reg_parents)
            knodes['%s_bottom' % reg['outcome']] = reg_knode

        return knodes
