
"""
"""

from copy import copy
import numpy as np
import pymc
import wfpt

from kabuki.hierarchical import Knode
from kabuki.utils import stochastic_from_dist
from hddm.models import HDDM
from wfpt import wiener_like_rlddm

class HDDMrl(HDDM):
    """HDDM model that can be used for two-armed bandit tasks.

    """
    def __init__(self,uncertainty=True,q_up=0.5,q_low=0.5, *args, **kwargs):
        self.alpha = kwargs.pop('alpha', True)
        self.dual_alpha = kwargs.pop('dual_alpha', False)
        self.q_up = q_up
        self.q_low = q_low
        self.uncertainty = uncertainty
        self.wfpt_rl_class = WienerRL
        
        super(HDDMrl, self).__init__(*args, **kwargs)

    def _create_stochastic_knodes(self, include):
        knodes = super(HDDMrl, self)._create_stochastic_knodes(include)
        if self.alpha:
            # Add learning rate parameter
            knodes.update(self._create_family_normal('alpha',
                                                                    value=0,
                                                                    g_mu=0.2,
                                                                    g_tau=3**-2,
                                                                    std_lower=1e-10,
                                                                    std_upper=10, 
                                                                    std_value=.1))
        return knodes

    def _create_wfpt_parents_dict(self, knodes):
        wfpt_parents = super(HDDMrl, self)._create_wfpt_parents_dict(knodes)

        wfpt_parents['alpha'] = knodes['alpha_bottom']
        wfpt_parents['dual_alpha'] = knodes['dual_alpha_bottom'] if 'dual_alpha' in self.include else 0
        
        return wfpt_parents

    #use own wfpt_class, defined in the init
    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)
        return Knode(self.wfpt_rl_class, 'wfpt',
                                   observed=True, col_name=['split_by','feedback', 'response', 'rt'],
                                   **wfpt_parents)

def wienerRL_like(x, v, alpha,dual_alpha, sv, a, z, sz, t, st, uncertainty,q_up,q_low,p_outlier=0.1):
    
    wiener_params = {'err': 1e-4, 'n_st':2, 'n_sz':2,
                         'use_adaptive':1,
                         'simps_err':1e-3,
                         'w_outlier': 0.1}
    sum_logp = 0
    wp = wiener_params

    response = x['response'].values.astype(int)
    q = np.array([q_up,q_low])
    feedback = x['feedback'].values
    split_by = x['split_by'].values
    unique = np.unique(split_by).shape[0]
    # could use something like the line below to avoid sending exp_up and exp_low as arrays. want to access the different values of 
    # of by u (below), but not using for-loop.
    #u, split_by = np.unique(x['split_by'].values, return_inverse=True)
    print("v = %.2f alpha = %.2f dual_alpha = %.2f a = %.2f qup = %.2f qlow = %.2f unique = %.2f uncertainty = %.2f t = %.2f z = %.2f sv = %.2f st = %.2f p_outlier = %.2f" 
          % (v,alpha,dual_alpha,a,q[1],q[0],unique,uncertainty,t,z,sv,st, p_outlier))
    return wiener_like_rlddm(x['rt'].values, response,feedback,q,split_by,unique,alpha,dual_alpha,v,sv, a, z, sz, t, st,uncertainty, p_outlier=p_outlier, **wp)
WienerRL = stochastic_from_dist('wienerRL', wienerRL_like)
