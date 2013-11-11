from __future__ import division
from collections import OrderedDict

import numpy as np
from numpy.random import rand, randn
import pymc as pm
import pandas as pd

import hddm
import kabuki
from base import AccumulatorModel
from kabuki.hierarchical import Knode
from kabuki.utils import stochastic_from_dist

def lba_like(value, t, A, b, s, v, p_outlier, w_outlier=.1):
    """LBA likelihood.
    """
    return hddm.lba.lba_like(value['rt'], A, b, t, s, v, 0, normalize_v=True, p_outlier=p_outlier, w_outlier=w_outlier)

def pdf(self, x):#value, t, A, b, s, v, p_outlier, w_outlier=.1):
    """LBA likelihood.
    """
    p = np.empty_like(x)
    for i, xi in enumerate(x):
        p[i] = hddm.lba.lba_like(np.atleast_1d(xi), self.parents['A'], self.parents['b'], self.parents['t'], self.parents['s'], self.parents['v'], 0, normalize_v=True, p_outlier=self.parents['p_outlier'], w_outlier=.1)

    return np.exp(p)

def lba_random(size=100, **params):
    """Generate RTs from LBA process.
    """
    t = params['t']
    A = params['A']
    b = params['b']
    s = params['s']
    v0 = params['v']
    v1 = 1 - v0
    sampled_rts = np.empty(size)
    for i_sample in xrange(size):
        positive = False
        while not positive:
            i_v0 = randn()*s + v0
            i_v1 = randn()*s + v1
            if (i_v0 > 0) or (i_v1 > 0):
                positive = True
                if i_v0 > 0:
                    rt0 = (b - rand()*A) / i_v0 + t
                else:
                    rt0 = np.inf

                if i_v1 > 0:
                    rt1 = (b - rand()*A) / i_v1 + t
                else:
                    rt1 = np.inf

                if rt0 < rt1:
                    sampled_rts[i_sample] = rt0
                else:
                    sampled_rts[i_sample] = -rt1

    data = pd.DataFrame(sampled_rts, columns=['rt'])
    data['response'] = 1.
    data['response'][data['rt']<0] = 0.
    data['rt'] = np.abs(data['rt'])

    return data


lba_class = stochastic_from_dist(name='lba_like', logp=lba_like, random=lba_random)
lba_class.pdf = pdf

def gen_single_params_set():
    """Returns a dict of DDM parameters with random values for a singel conditin
    the function is used by gen_rand_params.
    """
    params = {}
    params['s'] = 2.5*rand()
    params['b'] = 0.5+rand()*1.5
    params['A'] = rand() * params['b']
    params['v'] = rand()
    params['t'] = 0.2+rand()*0.3

    return params


def gen_rand_params(cond_dict=None, seed=None):
    """Returns a dict of DDM parameters with random values.

        :Optional:
            cond_dict : dictionary
                cond_dict is used when multiple conditions are desired.
                the dictionary has the form of {param1: [value_1, ... , value_n], param2: [value_1, ... , value_n]}
                and the function will output n sets of parameters. each set with values from the
                appropriate place in the dictionary
                for instance if cond_dict={'v': [0, 0.5, 1]} then 3 parameters set will be created.
                the first with v=0 the second with v=0.5 and the third with v=1.

            seed: float
                random seed

            Output:
            if conditions is None:
                params: dictionary
                    a dictionary holding the parameters values
            else:
                cond_params: a dictionary holding the parameters for each one of the conditions,
                    that has the form {'c1': params1, 'c2': params2, ...}
                    it can be used directly as an argument in gen_rand_data.
                merged_params:
                     a dictionary of parameters that can be used to validate the optimization
                     and learning algorithms.
    """


    #set seed
    if seed is not None:
        np.random.seed(seed)

    #if there is only a single condition then we can use gen_single_params_set
    if cond_dict is None:
       return gen_single_params_set()

    #generate original parameter set
    org_params = gen_single_params_set()

    #create a merged set
    merged_params = org_params.copy()
    for name in cond_dict.iterkeys():
        del merged_params[name]

    cond_params = {};
    n_conds = len(cond_dict.values()[0])
    for i in range(n_conds):
        #create a set of parameters for condition i
        #put them in i_params, and in cond_params[c#i]
        i_params = org_params.copy()
        for name in cond_dict.iterkeys():
            i_params[name] = cond_dict[name][i]
            cond_params['c%d' %i] = i_params

            #update merged_params
            merged_params['%s(c%d)' % (name, i)] = cond_dict[name][i]

    return cond_params, merged_params


def gen_rand_data(params=None, **kwargs):
    """Generate simulated RTs with random parameters.

       :Optional:
            params : dict <default=generate randomly>
                Either dictionary mapping param names to values.

                Or dictionary mapping condition name to parameter
                dictionary (see example below).

                If not supplied, takes random values.

            n_fast_outliers : int <default=0>
                How many fast outliers to add (outlier_RT < ter)

            n_slow_outliers : int <default=0>
                How many late outliers to add.

            The rest of the arguments are forwarded to kabuki.generate.gen_rand_data

       :Returns:
            data array with RTs
            parameter values

       :Example:
            # Generate random data set
            >>> data, params = hddm.generate.gen_rand_data({'v':0, 'a':2, 't':.3},
                                                           size=100, subjs=5)

            # Generate 2 conditions
            >>> data, params = hddm.generate.gen_rand_data({'cond1': {'v':0, 'a':2, 't':.3},
                                                            'cond2': {'v':1, 'a':2, 't':.3}})

       :Notes:
            Wrapper function for kabuki.generate.gen_rand_data. See
            the help doc of that function for more options.

    """

    if params is None:
        params = gen_rand_lba_params()

    from numpy import inf

    # set valid param ranges
    bounds = {'b': (0, inf),
              'A': (0, inf),
              'v': (0, 1),
              's': (0, inf),
              't': (0, inf)
    }

    if 'share_noise' not in kwargs:
        kwargs['share_noise'] = set(['b','A','t','s','v'])

    # Create RT data
    data, subj_params = kabuki.generate.gen_rand_data(lba_random, params,
                                                      bounds=bounds, **kwargs)
    return data, subj_params

class HLBA(AccumulatorModel):
    def __init__(self, data, **kwargs):
        self.informative = kwargs.pop('informative', True)
        self.std_depends = kwargs.get('std_depends', False)
        self.slice_widths = {'b': 1, 'b_std': .5,
                             'A':1, 'A_std': .5,
                             'v': 1., 'v_std': .5, 'v_certainty': 5,
                             't': 0.05, 't_std': .1,
                             'p_outlier':1.,
                             's': .2, 's_std': .05}
        self.emcee_dispersions = {'b': 1, 'b_std': .5,
                             'A':.5, 'A_std': .5,
                             'v': 1., 'v_std': .5, 'v_certainty': 2,
                             't': 0.05, 't_std': .1,
                             'p_outlier':1.,
                             's': .2, 's_std': .05}
        self.p_outlier = kwargs.pop('p_outlier', 0.)
        self.fix_A = kwargs.pop('fix_A', False)

        if self.p_outlier is True:
            self.include = ['p_outlier']
        else:
            self.include = []

        super(HLBA, self).__init__(data, **kwargs)

    def pre_sample(self, use_slice=True):
        from kabuki import steps
        for name, node_descr in self.iter_stochastics():
            node = node_descr['node']
            knode_name = node_descr['knode_name'].replace('_subj', '')
            self.mc.use_step_method(steps.SliceStep, node, width=self.slice_widths.get(knode_name, 1),
                                    left=0, maxiter=5000)

    def _create_family_beta(self, name, value=.5, g_value=.5, g_mean=.5, g_certainty=2,
                           var_alpha=1, var_beta=1, var_value=.1):
        """Similar to create_family_normal() but beta for the subject
        and group mean nodes. This is useful when the parameter space
        is restricted from [0, 1].

        See create_family_normal() help for more information.

        """

        knodes = OrderedDict()
        if self.is_group_model and name not in self.group_only_nodes:
            g_mean = Knode(pm.Beta, name, alpha=g_mean*g_certainty, beta=(1-g_mean)*g_certainty,
                      value=g_value, depends=self.depends[name])

            g_certainty = Knode(pm.Gamma, '%s_certainty' % name,
                                alpha=var_alpha, beta=var_beta, value=var_value)

            alpha = Knode(pm.Deterministic, '%s_alpha' % name, eval=lambda mean, certainty: mean*certainty,
                      mean=g_mean, certainty=g_certainty, plot=False, trace=False, hidden=True)

            beta = Knode(pm.Deterministic, '%s_beta' % name, eval=lambda mean, certainty: (1-mean)*certainty,
                      mean=g_mean, certainty=g_certainty, plot=False, trace=False, hidden=True)

            subj = Knode(pm.Beta, '%s_subj'%name, alpha=alpha, beta=beta,
                         value=value, depends=('subj_idx',),
                         subj=True, plot=False)

            knodes['%s'%name] = g_mean
            knodes['%s_certainty'%name] = g_certainty
            knodes['%s_alpha'%name] = alpha
            knodes['%s_beta'%name] = beta
            knodes['%s_bottom'%name] = subj

        else:
            g = Knode(pm.Beta, name, alpha=g_mean*g_certainty, beta=(1-g_mean)*g_certainty, value=value,
                      depends=self.depends[name])

            knodes['%s_bottom'%name] = g

        return knodes

    def _create_knodes_informative(self):
        knodes = OrderedDict()
        knodes.update(self._create_family_gamma_gamma_hnormal('t', g_mean=.4, g_std=1., value=0.001, std_std=.5, std_value=0.2))
        knodes.update(self._create_family_gamma_gamma_hnormal('b', g_mean=1.5, g_std=.75, std_std=2, std_value=0.1, value=1.5))
        if self.fix_A is False:
            knodes.update(self._create_family_gamma_gamma_hnormal('A', g_mean=.5, g_std=1, std_std=2, std_value=0.1, value=.2))
        knodes['s_bottom'] = Knode(pm.HalfNormal, 's', tau=0.3**-2, value=1., depends=self.depends['s'])
        #knodes.update(self._create_family_gamma_gamma_hnormal('s', g_mean=1, g_std=2, std_std=2, value=1.))
        knodes.update(self._create_family_beta('v', value=.75, g_mean=.75, g_certainty=0.75**-2))
        if 'p_outlier' in self.include:
            knodes['p_outlier_bottom'] = Knode(pm.Beta, 'p_outlier', alpha=1, beta=15, value=0.01, depends=self.depends['p_outlier'])
        #knodes.update(self._create_family_gamma_gamma_hnormal('v', g_mean=1, g_std=2, std_std=2, value=0.5))

        return knodes

    def _create_knodes_noninformative(self):
        """Returns list of model parameters.
        """
        knodes = OrderedDict()
        knodes.update(self._create_family_trunc_normal('t', lower=0, upper=1, value=.001))
        if self.fix_A is False:
            knodes.update(self._create_family_trunc_normal('A', lower=1e-3, upper=10, value=.2))
        knodes.update(self._create_family_trunc_normal('b', lower=1e-3, upper=10, value=1.5))
        knodes['s_bottom'] = Knode(pm.Uniform, 's', lower=1e-6, upper=3, value=0.1, depends=self.depends['s'])
        #knodes.update(self._create_family_trunc_normal('s', lower=0, upper=10, value=1.))
        knodes.update(self._create_family_trunc_normal('v', lower=0, upper=1, value=.5))
        if 'p_outlier' in self.include:
            knodes['p_outlier_bottom'] = Knode(pm.Beta, 'p_outlier', alpha=1, beta=15, value=0.01, depends=self.depends['p_outlier'])

        return knodes

    def _create_lba_knode(self, knodes):
        lba_parents = OrderedDict()
        lba_parents['t'] = knodes['t_bottom']
        lba_parents['v'] = knodes['v_bottom']
        lba_parents['A'] = knodes['A_bottom'] if self.fix_A is False else self.fix_A
        lba_parents['b'] = knodes['b_bottom']
        lba_parents['s'] = knodes['s_bottom']
        lba_parents['p_outlier'] = knodes['p_outlier_bottom'] if 'p_outlier' in self.include else self.p_outlier

        return Knode(lba_class, 'lba', observed=True, col_name='rt', **lba_parents)

    def create_knodes(self):
        if self.informative:
            knodes = self._create_knodes_informative()
        else:
            knodes = self._create_knodes_noninformative()

        knodes['lba'] = self._create_lba_knode(knodes)

        return knodes.values()

    def plot_posterior_predictive(self, *args, **kwargs):
        if 'value_range' not in kwargs:
            kwargs['value_range'] = np.linspace(-5, 5, 100)
        kabuki.analyze.plot_posterior_predictive(self, *args, **kwargs)
