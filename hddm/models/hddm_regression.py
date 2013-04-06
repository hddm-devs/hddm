from copy import deepcopy
import math
import numpy as np
import pymc as pm
import pandas as pd
from patsy import dmatrix

import hddm
from hddm.models import HDDM
import kabuki
from kabuki import Knode
from kabuki.utils import stochastic_from_dist
import kabuki.step_methods as steps

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
        for arg in reg['params']:
            for parent_name, parent in kwargs['parents'].iteritems():
                if parent_name == arg:
                    args.append(parent)

        parents = {'args': args}

        # Make sure design matrix is kosher
        dm = dmatrix(reg['model'], data=data)
        if math.isnan(dm.sum()):
            raise NotImplementedError, 'DesignMatrix contains NaNs.'

        def func(args, design_matrix=dmatrix(reg['model'], data=data), link_func=reg['link_func']):
            # convert parents to matrix
            params = np.matrix(args)
            # Apply design matrix to input data
            if design_matrix.shape[1] != params.shape[1]:
                raise NotImplementedError, 'Missing columns in design matrix. You need data for all conditions for all subjects.'
            predictor = link_func(pd.DataFrame((design_matrix * params).sum(axis=1), index=data.index))

            return pd.DataFrame(predictor, index=data.index)

        return self.pymc_node(func, kwargs['doc'], name, parents=parents)

class HDDMRegressor(HDDM):
    """HDDMRegressor allows estimation of the DDM where parameter
    values are linear models of a covariate (e.g. a brain measure like
    fMRI or different conditions).
    """

    def __init__(self, data, models, group_only_regressors=True, **kwargs):
        """Instantiate a regression model.

        :Arguments:

            * data : pandas.DataFrame
                data containing 'rt' and 'response' column and any
                covariates you might want to use.
            * models : str or list of str
                Patsy linear model specifier.
                E.g. 'v ~ cov'
                You can include multiple linear models that influence
                separate DDM parameters.

        :Optional:

            * group_only_regressors : bool
                Do not estimate individual subject parameters for all regressors.
            * Additional keyword args are passed on to HDDMGamma.

        :Note:

            Internally, HDDMRegressor uses patsy which allows for
            simple yet powerful model specification. For more information see:
            http://patsy.readthedocs.org/en/latest/overview.html

        :Example:

            Consider you have a trial-by-trial brain measure
            (e.g. fMRI) as an extra column called 'BOLD' in your data
            frame. You want to estimate whether BOLD has an effect on
            drift-rate. The corresponding model might look like
            this:
                ```python
                HDDMRegressor(data, 'v ~ BOLD')
                ```

            This will estimate an v_Intercept and v_BOLD. If v_BOLD is
            positive it means that there is a positive correlation
            between BOLD and drift-rate on a trial-by-trial basis.

            This type of mechanism also allows within-subject
            effects. If you have two conditions, 'cond1' and 'cond2'
            in the 'conditions' column of your data you may
            specify:
                ```python
                HDDMRegressor(data, 'v ~ C(condition)')
                ```
            This will lead to estimation of 'v_Intercept' for cond1
            and v_C(condition)[T.cond2] for cond1+cond2.

        """
        if isinstance(models, (basestring, dict)):
            models = [models]

        group_only_nodes = list(kwargs.get('group_only_nodes', ()))
        self.reg_outcomes = set() # holds all the parameters that are going to modeled as outcome

        self.model_descrs = []

        for model in models:
            if isinstance(model, dict):
                try:
                    model_str = model['model']
                    link_func = model['link_func']
                except KeyError:
                    raise KeyError, "HDDMRegressor requires a model specification either like {'model': 'v ~ 1 + C(your_variable)', 'link_func' lambda x: np.exp(x)} or just a model string"
            else:
                model_str = model
                link_func = lambda x: x

            separator = model_str.find('~')
            assert separator != -1, 'No outcome variable specified.'
            outcome = model_str[:separator].strip(' ')
            model_stripped = model_str[(separator+1):]
            covariates = dmatrix(model_stripped, data).design_info.column_names

            # Build model descriptor
            model_descr = {'outcome': outcome,
                           'model': model_stripped,
                           'params': ['{out}_{reg}'.format(out=outcome, reg=reg) for reg in covariates],
                           'link_func': link_func
            }
            self.model_descrs.append(model_descr)

            print "Adding these covariates:"
            print model_descr['params']
            if group_only_regressors:
                group_only_nodes += model_descr['params']
                kwargs['group_only_nodes'] = group_only_nodes
            self.reg_outcomes.add(outcome)

        #set wfpt_reg
        self.wfpt_reg_class = deepcopy(wfpt_reg_like)

        super(HDDMRegressor, self).__init__(data, **kwargs)

        # Sanity checks
        for model_descr in self.model_descrs:
            for param in model_descr['params']:
                assert len(self.depends[param]) == 0, "When using patsy, you can not use any model parameter in depends_on."

    def __getstate__(self):
        d = super(HDDMRegressor, self).__getstate__()
        del d['wfpt_reg_class']
        print "WARNING: Will not save custom link functions."
        for model in d['model_descrs']:
            del model['link_func']
        return d

    def __setstate__(self, d):
        d['wfpt_reg_class'] = deepcopy(wfpt_reg_like)
        print "WARNING: Custom link functions will not be loaded."
        for model in d['model_descrs']:
            model['link_func'] = lambda x: x
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
        for reg in self.model_descrs:
            reg_parents = {}
            for param in reg['params']:
                inter = param.find('Intercept')
                if inter != -1:
                    param_lookup = param[:inter-1]
                    reg_family = super(HDDMRegressor, self)._create_stochastic_knodes(set(param_lookup))
                else:
                    reg_family = self._create_family_normal(param)
                    param_lookup = param

                reg_parents[param] = reg_family['%s_bottom' % param_lookup]
                if reg not in self.group_only_nodes:
                    reg_family['%s_subj_reg' % param] = reg_family.pop('%s_bottom' % param_lookup)
                knodes.update(reg_family)
                self.slice_widths[param] = .05

            reg_knode = KnodeRegress(pm.Deterministic, "%s_reg" % reg['outcome'],
                                     regressor=reg,
                                     subj=self.is_group_model,
                                     plot=False,
                                     trace=False,
                                     hidden=True,
                                     **reg_parents)

            knodes['%s_bottom' % reg['outcome']] = reg_knode

        return knodes
