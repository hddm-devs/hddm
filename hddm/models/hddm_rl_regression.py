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

def generate_wfpt_rl_reg_stochastic_class(wiener_params=None, sampling_method='cdf', cdf_range=(-5,5), sampling_dt=1e-4):

    #set wiener_params
    if wiener_params is None:
        wiener_params = {'err': 1e-4, 'n_st':2, 'n_sz':2,
                         'use_adaptive':1,
                         'simps_err':1e-3,
                         'w_outlier': 0.1}
    wp = wiener_params

    def wienerRL_multi_like(value, v, sv, a, z, sz, t, st, alpha, reg_outcomes, p_outlier=0):
        """Log-likelihood for the full DDM using the interpolation method"""
        response = value['response'].values.astype(int)
        q = value['q_init'].iloc[0]
        feedback = value['feedback'].values.astype(float)
        split_by = value['split_by'].values.astype(int)
        params = {'v': v, 'sv': sv, 'a': a, 'z': z, 'sz': sz, 't': t, 'st': st, 'alpha': alpha}
        for reg_outcome in reg_outcomes:
            params[reg_outcome] = params[reg_outcome].loc[value['rt'].index].values
        return hddm.wfpt.wiener_like_multi_rlddm(value['rt'].values, response, feedback, split_by, q,
                                           params['v'], params['sv'], params['a'], params['z'],
                                           params['sz'], params['t'], params['st'],params['alpha'], 1e-4,
                                           reg_outcomes,
                                           p_outlier=p_outlier)


    def random(self):
        param_dict = deepcopy(self.parents.value)
        del param_dict['reg_outcomes']
        sampled_rts = self.value.copy()

        for i in self.value.index:
            #get current params
            for p in self.parents['reg_outcomes']:
                param_dict[p] = np.asscalar(self.parents.value[p].loc[i])
            #sample
            samples = hddm.generate.gen_rts(method=sampling_method,
                                            size=1, dt=sampling_dt, **param_dict)

            sampled_rts.loc[i, 'rt'] = hddm.utils.flip_errors(samples).rt

        return sampled_rts

    stoch = stochastic_from_dist('wfpt_reg', wienerRL_multi_like)
    stoch.random = random

    return stoch


wfpt_reg_like = generate_wfpt_rl_reg_stochastic_class(sampling_method='drift')


################################################################################################

class KnodeRegress(kabuki.hierarchical.Knode):
    def __init__(self, *args, **kwargs):
        self.keep_regressor_trace = kwargs.pop('keep_regressor_trace', False)
        super(KnodeRegress, self).__init__(*args, **kwargs)

    def create_node(self, name, kwargs, data):
        reg = kwargs['regressor']
        # order parents according to user-supplied args
        args = []
        for arg in reg['params']:
            for parent_name, parent in kwargs['parents'].items():
                if parent_name == arg:
                    args.append(parent)

        parents = {'args': args}

        # Make sure design matrix is kosher
        dm = dmatrix(reg['model'], data=data)
        if math.isnan(dm.sum()):
            raise NotImplementedError('DesignMatrix contains NaNs.')

        def func(args, design_matrix=dmatrix(reg['model'], data=data), link_func=reg['link_func']):
            # convert parents to matrix
            params = np.matrix(args)
            # Apply design matrix to input data
            if design_matrix.shape[1] != params.shape[1]:
                raise NotImplementedError('Missing columns in design matrix. You need data for all conditions for all subjects.')
            predictor = link_func(pd.DataFrame((design_matrix * params).sum(axis=1), index=data.index))

            return pd.DataFrame(predictor, index=data.index)

        return self.pymc_node(func, kwargs['doc'], name, parents=parents, trace=self.keep_regressor_trace)

class HDDMrlRegressor(HDDM):
    """HDDMrlRegressor allows estimation of the RLDDM where parameter
    values are linear models of a covariate (e.g. a brain measure like
    fMRI or different conditions).
    """

    def __init__(self, data, models, group_only_regressors=True, keep_regressor_trace=False, **kwargs):
        """Instantiate a regression model.

        :Arguments:

            * data : pandas.DataFrame
                data containing 'rt', 'response','feedback','split_by' and 'q_init' column and any
                covariates you might want to use.
            * models : str or list of str
                Patsy linear model specifier.
                E.g. 'v ~ cov'
                You can include multiple linear models that influence
                separate DDM parameters.

        :Optional:

            * group_only_regressors : bool (default=True)
                Do not estimate individual subject parameters for all regressors.
            * keep_regressor_trace : bool (default=False)
                Whether to keep a trace of the regressor. This will use much more space,
                but needed for posterior predictive checks.
            * Additional keyword args are passed on to HDDM.

        :Note:

            Internally, HDDMrlRegressor uses patsy which allows for
            simple yet powerful model specification. For more information see:
            http://patsy.readthedocs.org/en/latest/overview.html

        :Example:

            Consider you have a trial-by-trial brain measure
            (e.g. fMRI) as an extra column called 'BOLD' in your data
            frame. You want to estimate whether BOLD has an effect on
            drift-rate. The corresponding model might look like
            this:
                ```python
                HDDMrlRegressor(data, 'v ~ BOLD')
                ```

            This will estimate an v_Intercept and v_BOLD. If v_BOLD is
            positive it means that there is a positive correlation
            between BOLD and drift-rate on a trial-by-trial basis.

            This type of mechanism also allows within-subject
            effects. If you have two conditions, 'cond1' and 'cond2'
            in the 'conditions' column of your data you may
            specify:
                ```python
                HDDMrlRegressor(data, 'v ~ C(condition)')
                ```
            This will lead to estimation of 'v_Intercept' for cond1
            and v_C(condition)[T.cond2] for cond1+cond2.

        """

        self.keep_regressor_trace = keep_regressor_trace
        if isinstance(models, (str, dict)):
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
                    raise KeyError("HDDMrlRegressor requires a model specification either like {'model': 'v ~ 1 + C(your_variable)', 'link_func' lambda x: np.exp(x)} or just a model string")
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

            print("Adding these covariates:")
            print(model_descr['params'])
            if group_only_regressors:
                group_only_nodes += model_descr['params']
                kwargs['group_only_nodes'] = group_only_nodes
            self.reg_outcomes.add(outcome)

        #set wfpt_reg
        self.wfpt_reg_class = deepcopy(wfpt_reg_like)

        super(HDDMrlRegressor, self).__init__(data, **kwargs)

        # Sanity checks
        for model_descr in self.model_descrs:
            for param in model_descr['params']:
                assert len(self.depends[param]) == 0, "When using patsy, you can not use any model parameter in depends_on."

    def __getstate__(self):
        d = super(HDDMrlRegressor, self).__getstate__()
        del d['wfpt_reg_class']
        for model in d['model_descrs']:
            if 'link_func' in model:
                print("WARNING: Will not save custom link functions.")
                del model['link_func']
        return d

    def __setstate__(self, d):
        d['wfpt_reg_class'] = deepcopy(wfpt_reg_like)
        print("WARNING: Custom link functions will not be loaded.")
        for model in d['model_descrs']:
            model['link_func'] = lambda x: x
        super(HDDMrlRegressor, self).__setstate__(d)

    def _create_stochastic_knodes_rl(self, include):
        knodes = super(HDDMrlRegressor, self)._create_stochastic_knodes(include)
        if 'alpha' in include:            
            knodes.update(self._create_family_normal_normal_hnormal('alpha', value=0, g_tau=50**-2, std_std=10))
        return knodes

    def _create_wfpt_knode(self, knodes):
        wfpt_parents = super(HDDMrlRegressor, self)._create_wfpt_parents_dict(knodes)
        wfpt_parents['alpha'] = knodes['alpha_bottom']
        return Knode(self.wfpt_reg_class, 'wfpt', observed=True,
                     col_name=['split_by', 'feedback', 'response', 'rt', 'q_init'],
                     reg_outcomes=self.reg_outcomes, **wfpt_parents)

    def _create_stochastic_knodes(self, include):
        # Create all stochastic knodes except for the ones that we want to replace
        # with regressors.
        knodes = self._create_stochastic_knodes_rl(include.difference(self.reg_outcomes))
        
        # This is in dire need of refactoring. Like any monster, it just grew over time.
        # The main problem is that it's not always clear which prior to use. For the intercept
        # we want to use the original parameters' prior. Also for categoricals that do not
        # have an intercept, but not when the categorical is part of an interaction....

        # create regressor params
        for reg in self.model_descrs:
            reg_parents = {}
            # Find intercept parameter
            intercept = np.asarray([param.find('Intercept') for param in reg['params']]) != -1
            # If no intercept specified (via 0 + C()) assume all C() are different conditions
            # -> all are intercepts
            if not np.any(intercept):
                # Has categorical but no interaction
                intercept = np.asarray([(param.find('C(') != -1) and (param.find(':') == -1)
                                        for param in reg['params']])

            for inter, param in zip(intercept, reg['params']):
                if inter:
                    # Intercept parameter should have original prior (not centered on 0)
                    param_lookup = param[:param.find('_')]
                    reg_family = self._create_stochastic_knodes_rl([param_lookup])
                    # Rename nodes to avoid collissions
                    names = list(reg_family.keys())
                    for name in names:
                        knode = reg_family.pop(name)
                        knode.name = knode.name.replace(param_lookup, param, 1)
                        reg_family[name.replace(param_lookup, param, 1)] = knode
                    param_lookup = param

                else:
                    #param_lookup = param[:param.find('_')]
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
                                     keep_regressor_trace=self.keep_regressor_trace,
                                     **reg_parents)

            knodes['%s_bottom' % reg['outcome']] = reg_knode

        return knodes