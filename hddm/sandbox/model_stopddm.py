from copy import copy, deepcopy

import numpy as np
import pymc as pm
import pandas as pd

import hddm
from kabuki.hierarchical import Parameter, Knode


def ss_error_logp(value, ssd, v, V, a, z, Z, t, T, ssrt):
    """ss_error_logp"""
    data = value[value < ssd+ssrt]
    num_outliers = np.sum([value > ssd+ssrt])
    #if num_outliers > 0:
    #    return -np.inf
    logp_error = hddm.wfpt.wiener_like(data, v, V, a, z, Z, t, T, 1e-4)

    return logp_error + num_outliers * np.log(.01)

ss_error_like = pm.stochastic_from_dist(name="ss_error_like",
                                        logp=ss_error_logp,
                                        dtype=np.float,
                                        mv=True)

def ss_inhib_logp(value, v, V, a, z, Z, t, T, ssrt):
    """ss_error_logp"""
    assert V>=0 and a>0 and z>0 and z<1 and Z>=0 and t>=0 and T>=0 and ssrt>=0, "bad"

    # Compute probability of single SSD
    x, cdf = hddm.wfpt.gen_cdf(v, V, a, z, Z, t, T)
    x_lb, lb_cdf, x_ub, ub_cdf = hddm.wfpt.split_cdf(x, cdf)

    logp = 0
    for ssd, n_trials in value:
        assert ssd != -999
        # find ssd+ssrt
        lb_cutoff = np.where([x_lb > ssd+ssrt])[1]
        if len(lb_cutoff) == 0:
            return -np.inf
        lb_cutoff = lb_cutoff[0]
        p_lb = 1-lb_cdf[lb_cutoff]
        ub_cutoff = np.where([x_ub > ssd+ssrt])[1][0]
        p_ub = 1-ub_cdf[ub_cutoff]

        logp += n_trials * np.log(p_lb + p_ub)

    return logp

ss_inhib_like = pm.stochastic_from_dist(name="Ex-Gauss GoRT",
                                        logp=ss_inhib_logp,
                                        dtype=np.float,
                                        mv=False)

def gen_ss_data(params, samples=1000):
    data = pd.DataFrame(hddm.generate.gen_rand_data(params, samples=samples)[0])
    ssrt = params['ssrt']
    p_sst = params['p_sst']
    data['ss_presented'] = np.random.rand(len(data)) < p_sst
    data['ssd'] = np.NaN
    data['inhibited'] = False

    num_ss_presented = np.sum(data.ss_presented)
    ssds = np.arange(.75, 2.25, .25)
    data['ssd'][data.ss_presented] = np.repeat(ssds, num_ss_presented/len(ssds))[:num_ss_presented]

    inhibited = data.ss_presented & (data.rt > data.ssd + ssrt)
    data['rt'][inhibited] = np.NaN
    data['inhibited'][inhibited] = True

    assert np.all(data[(inhibited == False) & data.ss_presented]['rt'] < data[(inhibited == False) & data.ss_presented]['ssd'] + ssrt), "RTs larger than SSD+SSRT"

    return data


class StopDDM(hddm.HDDM):
    def create_params(self):
        """Returns list of model parameters.
        """
        # These boundaries are largely based on a meta-analysis of
        # reported fit values.
        # See: Matzke & Wagenmakers 2009

        params = super(StopDDM, self).create_params()
        wfpt = params.pop()

        basic_var = Knode(pm.Uniform, lower=1e-10, upper=100, value=1)

        # ssrt
        ssrt_g = Knode(pm.Uniform, lower=1e-3, upper=1, value=0.2)
        ssrt_subj = Knode(pm.TruncatedNormal, a=1e-3, b=1, value=0.2)
        ssrt = Parameter('ssrt', group_knode=ssrt_g,
                         var_knode=deepcopy(basic_var),
                         subj_knode=ssrt_subj, group_label='mu',
                         var_label='tau', var_type='std',
                         transform=lambda mu,var: (mu, var**-2))

        params.append(ssrt)

        ss_inhib = Parameter('ss_inhib', is_bottom_node=True, subj_knode=Knode(ss_inhib_like))
        params.append(ss_inhib)

        ss_error = Parameter('ss_error', is_bottom_node=True, subj_knode=Knode(ss_error_like))
        params.append(ss_error)

        params.append(wfpt)

        return params

    def get_bottom_node(self, param, params):
        if param.name == 'wfpt':
            data = copy(param.data[param.data['ss_presented'] == False])
            return hddm.likelihoods.wfpt_like(param.full_name,
                                              value=data['rt'],
                                              v=params['v'],
                                              a=params['a'],
                                              z=self.get_node('z',params),
                                              t=params['t'],
                                              Z=self.get_node('Z',params),
                                              T=self.get_node('T',params),
                                              V=self.get_node('V',params),
                                              observed=True)

        elif param.name == 'ss_error':
            data = copy(param.data[(param.data['ss_presented'] == True) & (param.data['inhibited'] == False)])

            return ss_error_like(param.full_name,
                                 value=data['rt'],
                                 ssd=data['ssd'],
                                 v=params['v'],
                                 a=params['a'],
                                 z=self.get_node('z',params),
                                 t=params['t'],
                                 Z=self.get_node('Z',params),
                                 T=self.get_node('T',params),
                                 V=self.get_node('V',params),
                                 ssrt=params['ssrt'],
                                 observed=True)

        elif param.name == 'ss_inhib':
            data = copy(param.data[(param.data['ss_presented'] == 1) & (param.data['inhibited'] == 1)])
            uniq_ssds = np.unique(data['ssd'])
            ssd_inhib_trials = []
            for uniq_ssd in uniq_ssds:
                ssd_inhib_trials.append((uniq_ssd, len(data[data['ssd'] == uniq_ssd])))
            ssd_inhib_trials = np.array(ssd_inhib_trials, dtype=np.float32)

            return ss_inhib_like(param.full_name,
                                 value=ssd_inhib_trials,
                                 v=params['v'],
                                 a=params['a'],
                                 z=self.get_node('z',params),
                                 t=params['t'],
                                 Z=self.get_node('Z',params),
                                 T=self.get_node('T',params),
                                 V=self.get_node('V',params),
                                 ssrt=params['ssrt'],
                                 observed=True)


