from __future__ import division
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import hddm
import kabuki
import kabuki.step_methods as steps
import scipy as sp
from collections import OrderedDict

from scipy import stats
from kabuki.hierarchical import Knode
from kabuki.distributions import scipy_stochastic
from copy import deepcopy
from numpy.random import rand, randn


class lba_gen(stats.distributions.rv_continuous):

    def _argcheck(self, *args):
        return True

    def _logp(self, x, t, A, b, s, v):
        return hddm.lba.lba_like(x, A, b, t, s, v, 0, logp=True, normalize_v=True)

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


class HLBA(hddm.model.AccumulatorModel):
    def create_lba_knode(self, knodes):
        lba_parents = OrderedDict()
        lba_parents['t'] = knodes['t_bottom']
        lba_parents['A'] = knodes['A_bottom']
        lba_parents['b'] = knodes['b_bottom']
        lba_parents['s'] = knodes['s_bottom']
        lba_parents['v'] = knodes['v_bottom']

        return Knode(lba_like, 'lba', observed=True, col_name='rt', **lba_parents)

    def create_knodes(self):
        """Returns list of model parameters.
        """

        knodes = OrderedDict()
        knodes.update(self.create_family_trunc_normal('t', lower=1e-3, upper=1e3, value=.01))
        knodes.update(self.create_family_trunc_normal('A', lower=1e-3, upper=1e3, value=.2))
        knodes.update(self.create_family_trunc_normal('b', lower=1e-3, upper=1e3, value=1.5))
        knodes.update(self.create_family_trunc_normal('s', lower=0, upper=1e3, value=1.))
        knodes.update(self.create_family_trunc_normal('v', lower=0, upper=1, value=.5))

        knodes['wfpt'] = self.create_lba_knode(knodes)

        return knodes.values()

    def plot_posterior_predictive(self, *args, **kwargs):
        if 'value_range' not in kwargs:
            kwargs['value_range'] = np.linspace(-1.5, 1.5, 100)
        kabuki.analyze.plot_posterior_predictive(self, *args, **kwargs)
