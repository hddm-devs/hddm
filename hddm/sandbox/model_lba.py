from __future__ import division
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import hddm
import kabuki
import kabuki.step_methods as steps
import scipy as sp

from scipy import stats
from kabuki.hierarchical import Parameter, Knode
from kabuki.distributions import scipy_stochastic
from copy import deepcopy
from numpy.random import rand, randn


class lba_gen(stats.distributions.rv_continuous):

    def _argcheck(self, *args):
        return True

    def _logp(self, x, t, A, b, s, v):
        return hddm.lba.lba_like(x, A, b, t, s, v, 0, logp=1, normalize_v=1)

    def _pdf(self, x, t, A, b, s, v):
        raise NotImplementedError

    def _rvs(self, t, A, b, s, v):
        v0 = v
        v1 = 1 - v;
        sampled_rts = np.empty(self._size)
        for i_sample in xrange(self._size):
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

        return sampled_rts

    def random(self, t, A, b, s, v, size):
        self._size = size
        return self._rvs(t, A, b, s, v)

lba_like = scipy_stochastic(lba_gen, name='lba', longname="", extradoc="")


class HLBA(kabuki.Hierarchical):

    def __init__(self, data, include=(), **kwargs):

        # Flip sign for lower boundary RTs
        data = hddm.utils.flip_errors(data)

        include_params = set()

        #set wfpt
        self.lba = deepcopy(hddm.likelihoods.wfpt_like)

        self.kwargs = kwargs

        super(self.__class__, self).__init__(data, include=include_params, **kwargs)

        # LBA model
        self.init_params = {}

        self.param_ranges = {'a_lower': .2,
                             'a_upper': 4.,
                             'v_lower': 0.1,
                             'v_upper': 3.,
                             'z_lower': .0,
                             'z_upper': 2.,
                             't_lower': .05,
                             't_upper': 2.,
                             'V_lower': .2,
                             'V_upper': 2.}

    def create_params(self):
        """Returns list of model parameters.
        """

        basic_var = Knode(pm.Uniform, lower=1e-10, upper=100, value=1)

        # A
        A_g = Knode(pm.Uniform, lower=1e-3, upper=1e3, value=1)
        A_subj = Knode(pm.TruncatedNormal, a=1e-3, b=np.inf, value=1)
        A = Parameter('A', group_knode=A_g, var_knode=deepcopy(basic_var), subj_knode=A_subj,
                      group_label='mu', var_label='tau', var_type='std',
                      transform=lambda mu,var:(mu, var**-2))

        # b
        b_g = Knode(pm.Uniform, lower=1e-3, upper=1e3, value=1.5)
        b_subj = Knode(pm.TruncatedNormal, a=1e-3, b=np.inf, value=1.5)
        b = Parameter('b', group_knode=b_g, var_knode=deepcopy(basic_var), subj_knode=b_subj,
                      group_label='mu', var_label='tau', var_type='std',
                      transform=lambda mu,var:(mu, var**-2))


        # v
        v_g = Knode(pm.Normal, mu=0.5, tau=2**-2, value=0.5, step_method=kabuki.steps.kNormalNormal)
        v_subj = Knode(pm.Normal, value=0.5)
        v = Parameter('v', group_knode=v_g, var_knode=deepcopy(basic_var), subj_knode=v_subj,
                      group_label = 'mu', var_label = 'tau', var_type='std',
                      transform=lambda mu,var:(mu, var**-2))

        # t
        t_g = Knode(pm.Uniform, lower=1e-3, upper=1e3, value=0.01)
        t_subj = Knode(pm.TruncatedNormal, a=1e-3, b=1e3, value=0.01)
        t = Parameter('t', group_knode=t_g, var_knode=deepcopy(basic_var), subj_knode=t_subj,
                      group_label = 'mu', var_label = 'tau', var_type='std',
                      transform=lambda mu,var:(mu, var**-2))

        #s
        s_g = Knode(pm.Uniform, lower=0, upper=1e3, value=1)
        s_subj = Knode(pm.TruncatedNormal, a=0, b=1e3, value=1)
        s = Parameter('s', group_knode=s_g, var_knode=deepcopy(basic_var), subj_knode=s_subj,
                      group_label = 'mu', var_label = 'tau', var_type='std',
                      transform=lambda mu,var:(mu, var**-2))
        #lba
        lba_knode = Knode(lba_like)
        lba = Parameter('lba', is_bottom_node=True, subj_knode=lba_knode)


        return [A, b, v, t, s, lba]


    def get_bottom_node(self, param, params):
        return lba_like(param.full_name,
                        value=param.data['rt'],
                        A=params['A'],
                        b=params['b'],
                        t=params['t'],
                        v=params['v'],
                        s=params['s'],
                        observed=True)

    def plot_posterior_predictive(self, *args, **kwargs):
        if 'value_range' not in kwargs:
            kwargs['value_range'] = np.linspace(-1.5, 1.5, 100)
        kabuki.analyze.plot_posterior_predictive(self, *args, **kwargs)
