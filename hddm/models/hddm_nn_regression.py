#from collections import OrderedDict
from copy import deepcopy
#import math
import numpy as np
#import pymc as pm
#import pandas as pd
#from patsy import dmatrix
#import pickle

import hddm
#from hddm.models import HDDM
from hddm.models import HDDMRegressor
from hddm.keras_models import load_mlp
from hddm.cnn.wrapper import load_cnn
from hddm.models.hddm_regression import KnodeRegress

#import kabuki
from kabuki import Knode
#from kabuki.utils import stochastic_from_dist
#import kabuki.step_methods as steps

#from functools import partial
#import wfpt
class HDDMnnRegressor(HDDMRegressor):
    """HDDMnnRegressor allows estimation of the NNDDM where parameter
    values are linear models of a covariate (e.g. a brain measure like
    fMRI or different conditions).
    """

    def __init__(self, data, models, group_only_regressors = True, keep_regressor_trace = False, **kwargs):
        """Instantiate a regression model, with neural network based likelihoods.
        
        :Arguments:

            * data : pandas.DataFrame
                data containing 'rt', 'response', column and any
                covariates you might want to use.
            * models : str or list of str
                Patsy linear model specifier.
                E.g. 'v ~ cov'
                You can include multiple linear models that influence
                separate DDM parameters.

        :Optional:

            * group_only_regressors : bool (default = True)
                Do not estimate individual subject parameters for all regressors.
            * keep_regressor_trace : bool (default = False)
                Whether to keep a trace of the regressor. This will use much more space,
                but needed for posterior predictive checks.
            * Additional keyword args are passed on to HDDM.

        :Note:

            Internally, HDDMnnRegressor uses patsy which allows for
            simple yet powerful model specification. For more information see:
            http://patsy.readthedocs.org/en/latest/overview.html

        :Example:

            Consider you have a trial-by-trial brain measure
            (e.g. fMRI) as an extra column called 'BOLD' in your data
            frame. You want to estimate whether BOLD has an effect on
            drift-rate. The corresponding model might look like
            this:
                ```python
                HDDMnnRegressor(data, 'v ~ BOLD')
                ```

            This will estimate an v_Intercept and v_BOLD. If v_BOLD is
            positive it means that there is a positive correlation
            between BOLD and drift-rate on a trial-by-trial basis.

            This type of mechanism also allows within-subject
            effects. If you have two conditions, 'cond1' and 'cond2'
            in the 'conditions' column of your data you may
            specify:
                ```python
                HDDMnnRegressor(data, 'v ~ C(condition)')
                ```
            This will lead to estimation of 'v_Intercept' for cond1
            and v_C(condition)[T.cond2] for cond1 + cond2.

        """
        # Signify as neural net class for later super() inits
        kwargs['nn'] = True
        self.model = kwargs.pop('model', 'ddm')
        self.w_outlier = kwargs.pop('w_outlier', 0.1)
        self.network_type = kwargs.pop('network_type', 'mlp')
        self.network = None
        self.nbin = kwargs.pop('nbin', 512)
        self.is_informative = kwargs.pop('informative', False)

        if self.nbin == 512:
            self.cnn_pdf_multiplier = 51.2
        elif self.nbin == 256:
            self.cnn_pdf_multiplier = 25.6

        # Load Network
        if self.network_type == 'mlp':
            self.network = load_mlp(model = self.model)
            network_dict = {'network': self.network}
            # Make likelihood function
            self.wfpt_nn_reg_class = hddm.likelihoods_mlp.generate_wfpt_nn_ddm_reg_stochastic_class(model = self.model, **network_dict)

        if self.network_type == 'cnn':
            self.network = load_cnn(model = self.model, nbin = self.nbin)
            network_dict = {'network': self.network}
            # Make likelihood function
            self.wfpt_nn_reg_class = hddm.likelihoods_cnn.generate_wfpt_nn_ddm_reg_stochastic_class(model = self.model, **network_dict)
        
        super(HDDMnnRegressor, self).__init__(data, models, group_only_regressors, keep_regressor_trace, **kwargs)

    def _create_wfpt_knode(self, knodes):
        print('passed through highest class version of _create_wfpt_knode')
        wfpt_parents = self._create_wfpt_parents_dict(knodes)

        return Knode(self.wfpt_nn_reg_class,
                     'wfpt',
                     observed = True,
                     col_name = ['response', 'rt'],
                     reg_outcomes = self.reg_outcomes,
                     **wfpt_parents)

    def _create_stochastic_knodes(self, include):
        # Create all stochastic knodes except for the ones that we want to replace
        # with regressors.
        includes_remainder = set(include).difference(self.reg_outcomes)
        # knodes = super(HDDMRegressor, self)._create_stochastic_knodes(include.difference(self.reg_outcomes))
        knodes = super(HDDMRegressor, self)._create_stochastic_knodes(includes_remainder)

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
                    reg_family = super(HDDMRegressor, self)._create_stochastic_knodes([param_lookup])
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
        
    # May need debugging --> set_state(), get_state()
    def __getstate__(self):
        d = super(HDDMnnRegressor, self).__getstate__()
        #del d['wfpt_reg_class']
        del d['network']
        del d['wfpt_nn_reg_class']

        # for model in d['model_descrs']:
        #     if 'link_func' in model:
        #         print("WARNING: Will not save custom link functions.")
        #         del model['link_func']
        return d

    def __setstate__(self, d):
        #d['wfpt_reg_class'] = deepcopy(wfpt_reg_like)
        # print("WARNING: Custom link functions will not be loaded.")
        # for model in d['model_descrs']:
        #     model['link_func'] = lambda x: x

        if d['network_type'] == 'cnn':
            d['network'] =  load_cnn(model = d['model'], nbin = d['nbin'])
            network_dict = {'network': d['network']}
            d['wfpt_nn_reg_class'] = hddm.likelihoods_cnn.make_cnn_likelihood(model = d['model'], pdf_multiplier = d['cnn_pdf_multiplier'], **network_dict)
           
        if d['network_type'] == 'mlp':
            d['network'] = load_mlp(model = d['model'])
            network_dict = {'network': d['network']}
            d['wfpt_nn_reg_class'] = hddm.likelihoods_mlp.make_mlp_likelihood(model = d['model'], **network_dict)

        super(HDDMnnRegressor, self).__setstate__(d)


    # def __getstate__(self):
    #     d = super(HDDMnn, self).__getstate__()
    #     del d['network']
    #     del d['wfpt_nn']
    #     #del d['wfpt_class']
    #     #del d['wfpt_reg_class']
    #     # for model in d['model_descrs']:
    #     #     if 'link_func' in model:
    #     #         print("WARNING: Will not save custom link functions.")
    #     #         del model['link_func']
    #     return d

    # def __setstate__(self, d):
    #     if d['network_type'] == 'cnn':
    #         d['network'] =  load_cnn(model = d['model'], nbin = d['nbin'])
    #         network_dict = {'network': d['network']}
    #         d['wfpt_nn'] = hddm.likelihoods_cnn.make_cnn_likelihood(model = d['model'], **network_dict)
           
    #     if d['network_type'] == 'mlp':
    #         d['network'] = load_mlp(model = d['model'])
    #         network_dict = {'network': d['network']}
    #         d['wfpt_nn'] = hddm.likelihoods_mlp.make_mlp_likelihood(model = d['model'],pdf_multiplier = d['cnn_pdf_multiplier'], **network_dict)

    #     super(HDDMnn, self).__setstate__(d) 
