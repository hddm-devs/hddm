"""
.. module:: HDDM
   :platform: Agnostic
   :synopsis: Definition of HDDM models.

.. moduleauthor:: Thomas Wiecki <thomas.wiecki@gmail.com>
                  Imri Sofer <imrisofer@gmail.com>


"""

from __future__ import division
from collections import OrderedDict

import numpy as np
import pymc as pm

import hddm
import kabuki
import inspect

from kabuki.hierarchical import Knode
from scipy.optimize import fmin_powell, fmin

class AccumulatorModel(kabuki.Hierarchical):
    def __init__(self, data, **kwargs):
        # Flip sign for lower boundary RTs
        data = hddm.utils.flip_errors(data)

        self.group_only_nodes = kwargs.pop('group_only_nodes', ())

        super(AccumulatorModel, self).__init__(data, **kwargs)


class HDDMBase(AccumulatorModel):
    """Implements the hierarchical Ratcliff drift-diffusion model
    using the Navarro & Fuss likelihood and numerical integration over
    the variability parameters.

    :Arguments:
        data : numpy.recarray
            Input data with a row for each trial.
             Must contain the following columns:
              * 'rt': Reaction time of trial in seconds.
              * 'response': Binary response (e.g. 0->error, 1->correct)
             May contain:
              * 'subj_idx': A unique ID (int) of the subject.
              * Other user-defined columns that can be used in depends on keyword.
    :Example:
        >>> data, params = hddm.generate.gen_rand_data() # gen data
        >>> model = hddm.HDDM(data) # create object
        >>> mcmc = model.mcmc() # Create pymc.MCMC object
        >>> mcmc.sample() # Sample from posterior
    :Optional:
        include : iterable
            Optional parameters to include. These include all inter-trial
            variability parameters ('sv', 'sz', 'st'), as well as the bias parameter ('z'), and
            the probability for outliers ('p_outlier').
            Can be any combination of 'sv', 'sz', 'st', 'z', and 'p_outlier'.
            Passing the string 'all' will include all five.

            Note: Including 'sz' and/or 'st' will increase run time significantly!

            is_group_model : bool
                If True, this results in a hierarchical
                model with separate parameter distributions for each
                subject. The subject parameter distributions are
                themselves distributed according to a group parameter
                distribution.

                If not provided, this parameter is set to True if data
                provides a column 'subj_idx' and False otherwise.

            depends_on : dict
                Specifies which parameter depends on data
                of a column in data. For each unique element in that
                column, a separate set of parameter distributions will be
                created and applied. Multiple columns can be specified in
                a sequential container (e.g. list)

                :Example:

                    >>> hddm.HDDM(data, depends_on={'v':'difficulty'})

                    Separate drift-rate parameters will be estimated
                    for each difficulty. Requires 'data' to have a
                    column difficulty.


            bias : bool
                Whether to allow a bias to be estimated. This
                is normally used when the responses represent
                left/right and subjects could develop a bias towards
                responding right. This is normally never done,
                however, when the 'response' column codes
                correct/error.

            plot_var : bool
                 Plot group variability parameters when calling pymc.Matplot.plot()
                 (i.e. variance of Normal distribution.)

            wiener_params : dict
                 Parameters for wfpt evaluation and
                 numerical integration.

             :Parameters:
                 * err: Error bound for wfpt (default 1e-4)
                 * n_st: Maximum depth for numerical integration for st (default 2)
                 * n_sz: Maximum depth for numerical integration for Z (default 2)
                 * use_adaptive: Whether to use adaptive numerical integration (default True)
                 * simps_err: Error bound for Simpson integration (default 1e-3)

        p_outlier : double (default=0)
            The probability of outliers in the data. if p_outlier is passed in the
            'include' argument, then it is estimated from the data and the value passed
            using the p_outlier argument is ignored.

    """

    def __init__(self, data, bias=False, include=(),
                 wiener_params=None, p_outlier=0., **kwargs):

        self._kwargs = kwargs

        self.include = set(['v', 'a', 't'])
        if include is not None:
            if include == 'all':
                [self.include.add(param) for param in ('z', 'st','sv','sz', 'p_outlier')]
            elif isinstance(include, str):
                self.include.add(include)
            else:
                [self.include.add(param) for param in include]

        if bias:
            self.include.add('z')

        possible_parameters = ('v', 'a', 't', 'z', 'st', 'sz', 'sv', 'p_outlier')
        assert self.include.issubset(possible_parameters), """Received and invalid parameter using the 'include' keyword.
        parameters received: %s
        parameters allowed: %s """ % (tuple(self.include), possible_parameters)

        #set wiener params
        if wiener_params is None:
            self.wiener_params = {'err': 1e-4, 'n_st':2, 'n_sz':2,
                                  'use_adaptive':1,
                                  'simps_err':1e-3,
                                  'w_outlier': 0.1}
        else:
            self.wiener_params = wiener_params
        wp = self.wiener_params
        self.p_outlier = p_outlier

        #set cdf_range
        cdf_bound = max(np.abs(data['rt'])) + 1;
        self.cdf_range = (-cdf_bound, cdf_bound)

        #set wfpt class
        self.wfpt_class = hddm.likelihoods.generate_wfpt_stochastic_class(wp, cdf_range=self.cdf_range)

        super(HDDMBase, self).__init__(data, **kwargs)

    def __getstate__(self):
        d = super(HDDMBase, self).__getstate__()
        del d['wfpt_class']

        return d

    def __setstate__(self, d):
        self.wfpt_class = hddm.likelihoods.generate_wfpt_stochastic_class(d['wiener_params'], cdf_range=d['cdf_range'])
        super(HDDMBase, self).__setstate__(d)

    def _create_wfpt_parents_dict(self, knodes):
        wfpt_parents = OrderedDict()
        wfpt_parents['a'] = knodes['a_bottom']
        wfpt_parents['v'] = knodes['v_bottom']
        wfpt_parents['t'] = knodes['t_bottom']

        wfpt_parents['sv'] = knodes['sv_bottom'] if 'sv' in self.include else 0
        wfpt_parents['sz'] = knodes['sz_bottom'] if 'sz' in self.include else 0
        wfpt_parents['st'] = knodes['st_bottom'] if 'st' in self.include else 0
        wfpt_parents['z'] = knodes['z_bottom'] if 'z' in self.include else 0.5
        wfpt_parents['p_outlier'] = knodes['p_outlier_bottom'] if 'p_outlier' in self.include else self.p_outlier
        return wfpt_parents

    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)

        return Knode(self.wfpt_class, 'wfpt', observed=True, col_name='rt', **wfpt_parents)

    def create_knodes(self):
        knodes = self._create_stochastic_knodes(self.include)
        knodes['wfpt'] = self._create_wfpt_knode(knodes)

        return knodes.values()

    def plot_posterior_predictive(self, *args, **kwargs):
        if 'value_range' not in kwargs:
            kwargs['value_range'] = np.linspace(-5, 5, 100)
        kabuki.analyze.plot_posterior_predictive(self, *args, **kwargs)

    def plot_posterior_quantiles(self, *args, **kwargs):
        """Plot posterior predictive quantiles.

        :Arguments:

            model : HDDM model

        :Optional:

            value_range : numpy.ndarray
                Range over which to evaluate the CDF.

            samples : int (default=10)
                Number of posterior samples to use.

            alpha : float (default=.75)
               Alpha (transparency) of posterior quantiles.

            hexbin : bool (default=False)
               Whether to plot posterior quantile density
               using hexbin.

            data_plot_kwargs : dict (default=None)
               Forwarded to data plotting function call.

            predictive_plot_kwargs : dict (default=None)
               Forwareded to predictive plotting function call.

            columns : int (default=3)
                How many columns to use for plotting the subjects.

            save : bool (default=False)
                Whether to save the figure to a file.

            path : str (default=None)
                Save figure into directory prefix

        """
        hddm.utils.plot_posterior_quantiles(self, *args, **kwargs)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
