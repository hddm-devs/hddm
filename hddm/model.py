"""
.. module:: HDDM
   :platform: Agnostic
   :synopsis: Defvalueion of HDDM models.

.. moduleauthor:: Thomas Wiecki <thomas.wiecki@gmail.com>
                  Imri Sofer <imrisofer@gmail.com>


"""

from __future__ import division
from collections import OrderedDict

import numpy as np
import pymc as pm

import hddm
import kabuki
import kabuki.step_methods as steps
from kabuki.hierarchical import Knode
from copy import deepcopy

class AccumulatorModel(kabuki.Hierarchical):
    def __init__(self, data, **kwargs):
        # Flip sign for lower boundary RTs
        data = hddm.utils.flip_errors(data)

        self.group_only_nodes = kwargs.pop('group_only_nodes', ())

        super(AccumulatorModel, self).__init__(data, **kwargs)


    def create_family_normal(self, name, value=0, g_mu=None,
                             g_tau=15**-2, var_lower=1e-10,
                             var_upper=100, var_value=.1):
        if g_mu is None:
            g_mu = value

        knodes = OrderedDict()

        if self.is_group_model and name not in self.group_only_nodes:
            g = Knode(pm.Normal, '%s' % name, mu=g_mu, tau=g_tau,
                      value=value, depends=self.depends[name])
            var = Knode(pm.Uniform, '%s_var' % name, lower=var_lower,
                        upper=var_upper, value=var_value)
            tau = Knode(pm.Deterministic, '%s_tau' % name,
                        doc='%s_tau' % name, eval=lambda x: x**-2, x=var,
                        plot=False, trace=False, hidden=True)
            subj = Knode(pm.Normal, '%s_subj' % name, mu=g, tau=tau,
                         value=value, depends=('subj_idx',),
                         subj=True, plot=self.plot_subjs)
            knodes['%s'%name] = g
            knodes['%s_var'%name] = var
            knodes['%s_tau'%name] = tau
            knodes['%s_bottom'%name] = subj

        else:
            subj = Knode(pm.Normal, name, mu=g_mu, tau=g_tau,
                         value=value, depends=self.depends[name])

            knodes['%s_bottom'%name] = subj

        return knodes


    def create_family_trunc_normal(self, name, value=0, lower=None,
                                   upper=None, var_lower=1e-10,
                                   var_upper=100, var_value=.1):
        knodes = OrderedDict()

        if self.is_group_model and name not in self.group_only_nodes:
            g = Knode(pm.Uniform, '%s' % name, lower=lower,
                      upper=upper, value=value, depends=self.depends[name])
            var = Knode(pm.Uniform, '%s_var' % name, lower=var_lower,
                        upper=var_upper, value=var_value)
            tau = Knode(pm.Deterministic, '%s_tau' % name,
                        doc='%s_tau' % name, eval=lambda x: x**-2, x=var,
                        plot=False, trace=False, hidden=True)
            subj = Knode(pm.TruncatedNormal, '%s_subj' % name, mu=g,
                         tau=tau, a=lower, b=upper, value=value,
                         depends=('subj_idx',), subj=True, plot=self.plot_subjs)

            knodes['%s'%name] = g
            knodes['%s_var'%name] = var
            knodes['%s_tau'%name] = tau
            knodes['%s_bottom'%name] = subj

        else:
            subj = Knode(pm.Uniform, name, lower=lower,
                         upper=upper, value=value,
                         depends=self.depends[name])
            knodes['%s_bottom'%name] = subj

        return knodes


    def create_family_invlogit(self, name, value, g_mu=None, g_tau=15**-2,
                               var_lower=1e-10, var_upper=100, var_value=.1):
        if g_mu is None:
            g_mu = value

        # logit transform values
        value_trans = np.log(value) - np.log(1-value)
        g_mu_trans = np.log(g_mu) - np.log(1-g_mu)

        knodes = OrderedDict()

        if self.is_group_model and name not in self.group_only_nodes:
            g_trans = Knode(pm.Normal,
                            '%s_trans'%name,
                            mu=g_mu_trans,
                            tau=g_tau,
                            value=value_trans,
                            depends=self.depends[name],
                            plot=False,
                            hidden=True
            )

            g = Knode(pm.InvLogit, name, ltheta=g_trans, plot=True,
                      trace=True)

            var = Knode(pm.Uniform, '%s_var'%name, lower=var_lower,
                        upper=var_upper, value=var_value)

            tau = Knode(pm.Deterministic, '%s_tau'%name, doc='%s_tau'
                        % name, eval=lambda x: x**-2, x=var,
                        plot=False, trace=False, hidden=True)

            subj_trans = Knode(pm.Normal, '%s_subj_trans'%name,
                               mu=g_trans, tau=tau, value=value_trans,
                               depends=('subj_idx',), subj=True,
                               plot=False, hidden=True)

            subj = Knode(pm.InvLogit, '%s_subj'%name,
                         ltheta=subj_trans, depends=('subj_idx',),
                         plot=self.plot_subjs, trace=True, subj=True)

            knodes['%s_trans'%name]      = g_trans
            knodes['%s'%name]            = g
            knodes['%s_var'%name]        = var
            knodes['%s_tau'%name]        = tau

            knodes['%s_subj_trans'%name] = subj_trans
            knodes['%s_bottom'%name]     = subj

        else:
            g_trans = Knode(pm.Normal, '%s_trans'%name, mu=g_mu,
                            tau=g_tau, value=value_trans,
                            depends=self.depends[name], plot=False, hidden=True)

            g = Knode(pm.InvLogit, '%s'%name, ltheta=g_trans, plot=True,
                      trace=True )

            knodes['%s_trans'%name] = g_trans
            knodes['%s_bottom'%name] = g

        return knodes

    def create_family_exp(self, name, value=0, g_mu=None,
                          g_tau=15**-2, var_lower=1e-10, var_upper=100, var_value=.1):
        if g_mu is None:
            g_mu = value

        value_trans = np.log(value)
        g_mu_trans = np.log(g_mu)

        knodes = OrderedDict()
        if self.is_group_model and name not in self.group_only_nodes:
            g_trans = Knode(pm.Normal, '%s_trans' % name, mu=g_mu_trans,
                            tau=g_tau, value=value_trans,
                            depends=self.depends[name], plot=False, hidden=True)

            g = Knode(pm.Deterministic, '%s'%name, eval=lambda x: np.exp(x),
                      x=g_trans, plot=True)

            var = Knode(pm.Uniform, '%s_var' % name,
                        lower=var_lower, upper=var_upper, value=var_value)

            tau = Knode(pm.Deterministic, '%s_tau' % name, eval=lambda x: x**-2,
                        x=var, plot=False, trace=False, hidden=True)

            subj_trans = Knode(pm.Normal, '%s_subj_trans'%name, mu=g_trans,
                         tau=tau, value=value_trans, depends=('subj_idx',),
                         subj=True, plot=False, hidden=True)

            subj = Knode(pm.Deterministic, '%s_subj'%name, eval=lambda x: np.exp(x),
                         x=subj_trans,
                         depends=('subj_idx',), plot=self.plot_subjs,
                         trace=True, subj=True)

            knodes['%s_trans'%name]      = g_trans
            knodes['%s'%name]            = g
            knodes['%s_var'%name]        = var
            knodes['%s_tau'%name]        = tau
            knodes['%s_subj_trans'%name] = subj_trans
            knodes['%s_bottom'%name]     = subj

        else:
            g_trans = Knode(pm.Normal, '%s_trans' % name, mu=g_mu_trans,
                            tau=g_tau, value=value_trans,
                            depends=self.depends[name], plot=False, hidden=True)

            g = Knode(pm.Deterministic, '%s'%name, doc='%s'%name, eval=lambda x: np.exp(x), x=g_trans, plot=True)
            knodes['%s_trans'%name] = g_trans
            knodes['%s_bottom'%name] = g

        return knodes


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
            Optional inter-trial variability parameters to include.
             Can be any combination of 'sv', 'sz' and 'st'. Passing the string
            'all' will include all three.

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

    """

    def __init__(self, data, bias=False, include=(),
                 wiener_params=None, **kwargs):

        self.include = set()
        if include is not None:
            if include == 'all':
                [self.include.add(param) for param in ('st','sv','sz')]
            else:
                [self.include.add(param) for param in include]

        if bias:
            self.include.add('z')

        if wiener_params is None:
            self.wiener_params = {'err': 1e-4, 'n_st':2, 'n_sz':2,
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

        super(HDDMBase, self).__init__(data, **kwargs)

    def create_wfpt_knode(self, knodes):
        wfpt_parents = OrderedDict()
        wfpt_parents['a'] = knodes['a_bottom']
        wfpt_parents['v'] = knodes['v_bottom']
        wfpt_parents['t'] = knodes['t_bottom']

        wfpt_parents['sv'] = knodes['sv_bottom'] if 'sv' in self.include else 0
        wfpt_parents['sz'] = knodes['sz_bottom'] if 'sz' in self.include else 0
        wfpt_parents['st'] = knodes['st_bottom'] if 'st' in self.include else 0
        wfpt_parents['z'] = knodes['z_bottom'] if 'z' in self.include else 0.5

        return Knode(self.wfpt, 'wfpt', observed=True, col_name='rt', **wfpt_parents)

    def plot_posterior_predictive(self, *args, **kwargs):
        if 'value_range' not in kwargs:
            kwargs['value_range'] = np.linspace(-5, 5, 100)
        kabuki.analyze.plot_posterior_predictive(self, *args, **kwargs)

class HDDMTruncated(HDDMBase):
   def create_knodes(self):
        """Returns list of model parameters.
        """

        knodes = OrderedDict()
        knodes.update(self.create_family_trunc_normal('a', lower=1e-3, upper=1e3, value=1))
        knodes.update(self.create_family_normal('v', value=0))
        knodes.update(self.create_family_trunc_normal('t', lower=1e-3, upper=1e3, value=.01))
        if 'sv' in self.include:
            knodes.update(self.create_family_trunc_normal('sv', lower=0, upper=1e3, value=1))
        if 'sz' in self.include:
            knodes.update(self.create_family_trunc_normal('sz', lower=0, upper=1, value=.1))
        if 'st' in self.include:
            knodes.update(self.create_family_trunc_normal('st', lower=0, upper=1e3, value=.01))
        if 'z' in self.include:
            knodes.update(self.create_family_trunc_normal('z', lower=0, upper=1, value=.5))

        knodes['wfpt'] = self.create_wfpt_knode(knodes)

        return knodes.values()


class HDDM(HDDMBase):
    def __init__(self, *args, **kwargs):
        self.use_gibbs = kwargs.pop('use_gibbs_for_mean', True)
        self.use_slice = kwargs.pop('use_slice_for_std', True)

        super(self.__class__, self).__init__(*args, **kwargs)

    def pre_sample(self):
        if not self.is_group_model:
            return

        # apply gibbs sampler to normal group nodes
        for name, node_descr in self.iter_group_nodes():
            node = node_descr['node']
            knode_name = node_descr['knode_name'].replace('_trans', '')
            if self.use_gibbs and isinstance(node, pm.Normal) and knode_name not in self.group_only_nodes:
                self.mc.use_step_method(steps.kNormalNormal, node)
            if self.use_slice and isinstance(node, pm.Uniform) and knode_name not in self.group_only_nodes:
                self.mc.use_step_method(steps.UniformPriorNormalstd, node)

    def create_knodes(self):
        knodes = OrderedDict()
        knodes.update(self.create_family_exp('a', value=1))
        knodes.update(self.create_family_normal('v', value=0))
        knodes.update(self.create_family_exp('t', value=.01))
        if 'sv' in self.include:
            # TW: Use kabuki.utils.HalfCauchy, S=10, value=1 instead?
            knodes.update(self.create_family_trunc_normal('sv', lower=0, upper=1e3, value=1))
            #knodes.update(self.create_family_exp('sv', value=1))
        if 'sz' in self.include:
            knodes.update(self.create_family_exp('sz', value=.1))
        if 'st' in self.include:
            knodes.update(self.create_family_exp('st', value=.01))
        if 'z' in self.include:
            knodes.update(self.create_family_invlogit('z', value=.5))

        knodes['wfpt'] = self.create_wfpt_knode(knodes)

        return knodes.values()

if __name__ == "__main__":
    import doctest
    doctest.testmod()
