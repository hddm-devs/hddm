"""
.. module:: HDDM
   :platform: Agnostic
   :synopsis: Definition of HDDM models.

.. moduleauthor:: Thomas Wiecki <thomas.wiecki@gmail.com>
                  Imri Sofer <imrisofer@gmail.com>


"""

from __future__ import division
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

import hddm
import kabuki
import kabuki.step_methods as steps
from kabuki.hierarchical import Knode
from copy import deepcopy
import scipy as sp
from scipy import stats



class HDDM(kabuki.Hierarchical):
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
            Optional inter-trial variability parameters to include.
             Can be any combination of 'V', 'Z' and 'T'. Passing the string
            'all' will include all three.

            Note: Including 'Z' and/or 'T' will increase run time significantly!

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
                     * nT: Maximum depth for numerical integration for T (default 2)
                     * nZ: Maximum depth for numerical integration for Z (default 2)
                     * use_adaptive: Whether to use adaptive numerical integration (default True)
                     * simps_err: Error bound for Simpson integration (default 1e-3)

    """

    def __init__(self, data, bias=False,
                 include=(), wiener_params=None, **kwargs):

        # Flip sign for lower boundary RTs
        data = hddm.utils.flip_errors(data)

        include_params = set()

        if include is not None:
            if include == 'all':
                [include_params.add(param) for param in ('T','V','Z')]
            else:
                [include_params.add(param) for param in include]

        if bias:
            include_params.add('z')

        if wiener_params is None:
            self.wiener_params = {'err': 1e-4, 'nT':2, 'nZ':2,
                                  'use_adaptive':1,
                                  'simps_err':1e-3}
        else:
            self.wiener_params = wiener_params
        wp = self.wiener_params

        #set wfpt
        self.wfpt = deepcopy(hddm.likelihoods.wfpt_like)
        self.wfpt.rv.wiener_params = wp
        cdf_bound = max(np.abs(data['rt'])) + 1;
        self.wfpt.cdf_range = (-cdf_bound, cdf_bound)

        super(hddm.model.HDDM, self).__init__(data, include=include_params, **kwargs)

    def create_knodes_set(self, name, lower=None, upper=None, init=0):
        knodes = {}
        if lower is None and upper is None:
            g = Knode(pm.Uniform, '%s_group_mean' % name, lower=1e-3, upper=1e3, value=init, depends=self.depends[name])
        else:
            g = Knode(pm.Normal, '%s_group_mean' % name, mu=0, tau=15**-2, value=init, depends=self.depends[name])



        var = Knode(pm.Uniform, '%s_group_var' % name, lower=1e-10, upper=100, value=init)

        tau = Knode(pm.Deterministic, '%s_group_tau' % name, doc='%_group_tau' % name, eval=lambda x: x**-2, x=var, plot=False, trace=False)

        if lower is not None or upper is not None:
            subj = Knode(pm.TruncatedNormal, '%s_subj' % name, mu=g, tau=tau, a=lower, b=upper, value=init, depends=('subj_idx',), subj=True)
        else:
            subj = Knode(pm.Normal, '%s_subj' % name, mu=g, tau=tau, value=init, depends=('subj_idx',), subj=True)

        knodes['%s_group_mean'%name] = g
        knodes['%s_group_var'%name] = var
        knodes['%s_group_tau'%name] = tau
        knodes['%s_subj'%name] = subj

        return knodes

    def create_knodes(self):
        """Returns list of model parameters.
        """
        knodes = {}
        wfpt_parents = {}

        a = self.create_knodes_set('a', lower=1e-3, upper=1e3, init=1)
        knodes.update(a)
        wfpt_parents['a'] = a['a_subj']

        v = self.create_knodes_set('v', init=0)
        knodes.update(v)
        wfpt_parents['v'] = v['v_subj']

        t = self.create_knodes_set('t', lower=1e-3, upper=1e3, init=.01)
        knodes.update(t)
        wfpt_parents['t'] = t['t_subj']

        if 'V' in self.include:
            V = self.create_knodes_set('V', lower=0, upper=1e3, init=1)
            knodes.update(V)
            wfpt_parents['V'] = V['V_subj']
        else:
            wfpt_parents['V'] = 0.

        if 'Z' in self.include:
            Z = self.create_knodes_set('Z', lower=0, upper=1, init=.1)
            knodes.update(Z)
            wfpt_parents['Z'] = Z['Z_subj']
        else:
            wfpt_parents['Z'] = 0.

        if 'T' in self.include:
            T = self.create_knodes_set('T', lower=0, upper=1e3, init=.01)
            knodes.update(T)
            wfpt_parents['T'] = T['T_subj']
        else:
            wfpt_parents['T'] = 0.

        if 'z' in self.include:
            z = self.create_knodes_set('z', lower=0, upper=1, init=.5)
            knodes.update(z)
            wfpt_parents['z'] = z['z_subj']
        else:
            wfpt_parents['z'] = .5

        #wfpt
        wfpt = Knode(self.wfpt, 'wfpt', observed=True, col_name='rt', **wfpt_parents)

        knodes['wfpt'] = wfpt

        return knodes.values

    def plot_posterior_predictive(self, *args, **kwargs):
        if 'value_range' not in kwargs:
            kwargs['value_range'] = np.linspace(-5, 5, 100)
        kabuki.analyze.plot_posterior_predictive(self, *args, **kwargs)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
