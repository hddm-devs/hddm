from collections import OrderedDict
from copy import deepcopy
#import math
#import numpy as np
#import pymc as pm
#import pandas as pd
from patsy import dmatrix
#import pickle

import hddm
#from hddm.models import HDDM
from hddm.models import HDDMRegressor
from hddm.keras_models import load_mlp
from hddm.cnn.wrapper import load_cnn
#from hddm.models.hddm_regression import KnodeRegress

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
        self.nn = False
        #kwargs['nn'] = True
        print('Setting priors uninformative (LANs only work with uninformative priors for now)')
        kwargs['informative'] = False
        self.model = kwargs.pop('model', 'ddm')
        self.w_outlier = kwargs.pop('w_outlier', 0.1)
        self.network_type = kwargs.pop('network_type', 'mlp')
        self.network = None
        self.nbin = kwargs.pop('nbin', 512)
        #self.is_informative = kwargs.pop('informative', False)

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
        wfpt_parents = self._create_wfpt_parents_dict(knodes)

        return Knode(self.wfpt_nn_reg_class,
                     'wfpt',
                     observed = True,
                     col_name = ['response', 'rt'],
                     reg_outcomes = self.reg_outcomes,
                     **wfpt_parents)
 
    # May need debugging --> set_state(), get_state()
    def __getstate__(self):
        d = super(HDDMnnRegressor, self).__getstate__()
        del d['network']
        del d['wfpt_nn_reg_class']
        return d

    def __setstate__(self, d):
        if d['network_type'] == 'cnn':
            d['network'] =  load_cnn(model = d['model'], nbin = d['nbin'])
            network_dict = {'network': d['network']}
            d['wfpt_nn_reg_class'] = hddm.likelihoods_cnn.generate_wfpt_nn_ddm_reg_stochastic_class(model = d['model'], pdf_multiplier = d['cnn_pdf_multiplier'], **network_dict)
           
        if d['network_type'] == 'mlp':
            d['network'] = load_mlp(model = d['model'])
            network_dict = {'network': d['network']}
            d['wfpt_nn_reg_class'] = hddm.likelihoods_mlp.generate_wfpt_nn_ddm_reg_stochastic_class(model = d['model'], **network_dict)

        super(HDDMnnRegressor, self).__setstate__(d)
