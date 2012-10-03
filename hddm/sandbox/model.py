import hddm
from hddm.model import HDDM
import pymc as pm
from kabuki.distributions import scipy_stochastic
import kabuki

import numpy as np
from scipy import stats

try:
    import wfpt_switch
except:
    pass

class wfpt_switch_gen(stats.distributions.rv_continuous):
    err = 1e-4
    evals = 100
    def _argcheck(self, *args):
        return True

    def _logp(self, x, v, vcc, a, z, t, tcc):
        """Log-likelihood for the simple DDM switch model"""
        logp = wfpt_switch.wiener_like_antisaccade_precomp(x, float(v), float(vcc), 0., float(a), .5, float(t), float(tcc), 0., float(self.err))
        return logp

    def _pdf(self, x, v, vcc, a, z, t, tcc, st):
        if np.isscalar(x):
            out = wfpt_switch.pdf_switch(np.array([x]), v, vcc, 0., a, z, t, tcc, st, 1e-4)
        else:
            out = np.empty_like(x)
            for i in xrange(len(x)):
                out[i] = wfpt_switch.pdf_switch(np.array([x[i]]), v[i], vcc[i], 0., a[i], z[i], t[i], tcc[i], st[i], 1e-4)

        return out

    def _rvs(self, v, vcc, a, z, t, tcc, st):
        all_rts_generated=False
        while(not all_rts_generated):
            out = hddm.generate.gen_antisaccade_rts({'v':v, 'z':z, 't':t, 'a':a, 'vcc':vcc, 'sv':0, 'tcc':tcc, 'Z':0, 'V':0, 'st': st}, samples_anti=self._size, samples_pro=0)[0]
            if (len(out) == self._size):
                all_rts_generated=True
        return hddm.utils.flip_errors(out)['rt']

wfpt_switch_like = scipy_stochastic(wfpt_switch_gen, name='wfpt switch', longname="""Wiener first passage time likelihood function""", extradoc="")

class KnodePro(kabuki.hierarchical.Knode):
    def create_node(self, name, kwargs):
        data = kwargs.pop('value')
        data = data[data[:,1] == 0]
        if len(data) == 0:
            return None

        return self.pymc_node(name, value=data[:,0], **kwargs)

class KnodeAnti(kabuki.hierarchical.Knode):
    def create_node(self, name, kwargs):
        data = kwargs.pop('value')
        data = data[data[:,1] == 1]
        if len(data) == 0:
            return None

        return self.pymc_node(name, value=data[:,0], **kwargs)

class HDDMSwitch(hddm.model.HDDMBase):
    def create_wfpt_knode(self, knodes):
        wfpt_pro = kabuki.OrderedDict()
        wfpt_pro['a'] = knodes['a_bottom']
        wfpt_pro['v'] = knodes['vpp_bottom']
        wfpt_pro['t'] = knodes['t_bottom']
        wfpt_pro['sv'] = 0
        wfpt_pro['sz'] = 0
        wfpt_pro['st'] = 0
        wfpt_pro['z'] = 0.5

        wfpt_anti = kabuki.OrderedDict()
        wfpt_anti['a'] = knodes['a_bottom']
        wfpt_anti['v'] = knodes['vpp_bottom']
        wfpt_anti['t'] = knodes['t_bottom']
        wfpt_anti['vcc'] = knodes['vcc_bottom']
        wfpt_anti['tcc'] = knodes['tcc_bottom']

        wfpt_pro = KnodePro(self.wfpt_class, 'wfpt_pro', observed=True, col_name=['rt', 'instruct'], **wfpt_pro)
        wfpt_anti = KnodeAnti(wfpt_switch_like, 'wfpt_anti', observed=True, col_name=['rt', 'instruct'], **wfpt_anti)

        return wfpt_pro, wfpt_anti

    def create_knodes(self):
        """Returns list of model parameters.
        """

        knodes = kabuki.OrderedDict()
        knodes.update(self.create_family_normal('vpp', value=0))
        knodes.update(self.create_family_normal('vcc', value=0))
        knodes.update(self.create_family_trunc_normal('a', lower=1e-3, upper=1e3, value=1))
        knodes.update(self.create_family_trunc_normal('tcc', lower=1e-3, upper=1e3, value=.1))
        knodes.update(self.create_family_trunc_normal('t', lower=1e-3, upper=1e3, value=.1))

        knodes['wfpt_pro'], knodes['wfpt_anti'] = self.create_wfpt_knode(knodes)

        return knodes.values()
