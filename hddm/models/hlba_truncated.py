from __future__ import division
from collections import OrderedDict

import numpy as np
from numpy.random import rand, randn

import hddm
import kabuki
from base import AccumulatorModel
from kabuki.hierarchical import Knode
from kabuki.utils import stochastic_from_dist

def lba_like(value, t, A, b, s, v):
    """LBA likelihood.
    """
    return hddm.lba.lba_like(value['rt'], A, b, t, s, v, 0, logp=True, normalize_v=True)

def lba_random(t, A, b, s, v, size=1):
    """Generate RTs from LBA process.
    """
    v0 = v
    v1 = 1 - v
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

    return sampled_rts

lba_class = stochastic_from_dist(name='lba_like', logp=lba_like, random=lba_random)

class HLBA(AccumulatorModel):
    def _create_lba_knode(self, knodes):
        lba_parents = OrderedDict()
        lba_parents['t'] = knodes['t_bottom']
        lba_parents['A'] = knodes['A_bottom']
        lba_parents['b'] = knodes['b_bottom']
        lba_parents['s'] = knodes['s_bottom']
        lba_parents['v'] = knodes['v_bottom']

        return Knode(lba_class, 'lba', observed=True, col_name='rt', **lba_parents)

    def create_knodes(self):
        """Returns list of model parameters.
        """

        knodes = OrderedDict()
        knodes.update(self._create_family_trunc_normal('t', lower=0, upper=1, value=.01))
        knodes.update(self._create_family_trunc_normal('A', lower=1e-3, upper=10, value=.2))
        knodes.update(self._create_family_trunc_normal('b', lower=1e-3, upper=10, value=1.5))
        knodes.update(self._create_family_trunc_normal('s', lower=0, upper=10, value=1.))
        knodes.update(self._create_family_trunc_normal('v', lower=0, upper=1, value=.5))

        knodes['wfpt'] = self._create_lba_knode(knodes)

        return knodes.values()

    def plot_posterior_predictive(self, *args, **kwargs):
        if 'value_range' not in kwargs:
            kwargs['value_range'] = np.linspace(-1.5, 1.5, 100)
        kabuki.analyze.plot_posterior_predictive(self, *args, **kwargs)
