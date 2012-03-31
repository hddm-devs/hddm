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
from kabuki.hierarchical import Parameter, Knode
from copy import deepcopy



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

        self.kwargs = kwargs

        super(hddm.model.HDDM, self).__init__(data, include=include_params, **kwargs)

    def create_params(self):
        """Returns list of model parameters.
        """
        # These boundaries are largely based on a meta-analysis of
        # reported fit values.
        # See: Matzke & Wagenmakers 2009
        basic_var = Knode(pm.Uniform, lower=1e-10, upper=100, value=1)

        a_g = Knode(pm.Uniform, lower=1e-3, upper=1e3, value=1)
        a_subj = Knode(pm.TruncatedNormal, a=0.3, b=1e3, value=1)
        # a
        a = Parameter('a', group_knode=a_g, var_knode=deepcopy(basic_var), subj_knode=a_subj,
                      group_label = 'mu', var_label = 'tau', var_type='std',
                      transform=lambda mu,var:(mu, var**-2))

        # v
        v_g = Knode(pm.Normal, mu=0, tau=15**-2, value=0, step_method=kabuki.steps.kNormalNormal)
        v_subj = Knode(pm.Normal, value=0)
        v = Parameter('v', group_knode=v_g, var_knode=deepcopy(basic_var), subj_knode=v_subj,
                      group_label = 'mu', var_label = 'tau', var_type='std',
                      transform=lambda mu,var:(mu, var**-2))

        # t
        t_g = Knode(pm.Uniform, lower=1e-3, upper=1e3, value=0.01)
        t_subj = Knode(pm.TruncatedNormal, a=0.1, b=1e3, value=0.1)
        t = Parameter('t', group_knode=t_g, var_knode=deepcopy(basic_var), subj_knode=t_subj,
                      group_label = 'mu', var_label = 'tau', var_type='std',
                      transform=lambda mu,var:(mu, var**-2))

        # z
        z_g = Knode(pm.Beta, alpha=1, beta=1, value=0.5)
        z_var = Knode(pm.Uniform, lower=1, upper=1e5, value=1)
        z_subj = Knode(pm.Beta, value=0.5)
        z = Parameter('z', group_knode=z_g, var_knode=z_var, subj_knode=z_subj,
                      group_label='alpha', var_label='beta', var_type='sample_size',
                      transform=lambda mu,n: (mu*n, (1-mu)*n),
                      optional=True, default=0.5)

        #V
        V_g = Knode(pm.Uniform, lower=0, upper=1e3, value=1)
        V_subj = Knode(pm.TruncatedNormal, a=0, b=1e3, value=1)
        V = Parameter('V', group_knode=V_g, var_knode=deepcopy(basic_var), subj_knode=V_subj,
                      group_label = 'mu', var_label = 'tau', var_type='std',
                      transform=lambda mu,var:(mu, var**-2),
                      optional=True, default=0)

        #Z
        Z_g = Knode(pm.Uniform, lower=0, upper=1, value=0.1)
        Z_subj = Knode(pm.TruncatedNormal, a=0, b=1, value=1)
        Z = Parameter('Z', group_knode=Z_g, var_knode=deepcopy(basic_var), subj_knode=Z_subj,
                      group_label = 'mu', var_label = 'tau', var_type='std',
                      transform=lambda mu,var:(mu, var**-2),
                      optional=True, default=0)

        #T
        T_g = Knode(pm.Uniform, lower=0, upper=1e3, value=0.01)
        T_subj = Knode(pm.TruncatedNormal, a=0, b=1e3, value=0.01)
        T = Parameter('T', group_knode=T_g, var_knode=deepcopy(basic_var), subj_knode=T_subj,
                      group_label = 'mu', var_label = 'tau', var_type='std',
                      transform=lambda mu,var:(mu, var**-2),
                      optional=True, default=0)

        #wfpt
        wfpt_knode = Knode(self.wfpt)
        wfpt = Parameter('wfpt', is_bottom_node=True, subj_knode=wfpt_knode)


        return [a, v, t, z, V, T, Z, wfpt]


    def get_bottom_node(self, param, params):
        """Create and return the wiener likelihood distribution
        supplied in 'param'.

        'params' is a dictionary of all parameters on which the data
        depends on (i.e. condition and subject).

        """
        if param.name == 'wfpt':
            return self.wfpt(param.full_name,
                             value=param.data['rt'].flatten(),
                             v=params['v'],
                             a=params['a'],
                             z=self.get_node('z',params),
                             t=params['t'],
                             Z=self.get_node('Z',params),
                             T=self.get_node('T',params),
                             V=self.get_node('V',params),
                             observed=True)

        else:
            raise KeyError, "Groupless parameter named %s not found." % param.name


    def plot_posterior_predictive(self, *args, **kwargs):
        if 'value_range' not in kwargs:
            kwargs['value_range'] = np.linspace(-5, 5, 100)
        kabuki.analyze.plot_posterior_predictive(self, *args, **kwargs)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
