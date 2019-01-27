
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
    def __init__(self,*args, **kwargs): #q=0.5,uncertainty=0,*args, **kwargs):
        #self.uncertainty = uncertainty
        #self.q = q
        self.alpha = kwargs.pop('alpha', True)
        self.dual_alpha = kwargs.pop('dual_alpha', False)
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
        #if self.uncertainty:
        #    knodes.update(self._create_family_normal('uncertainty',
        #                                                            value=0,
        #                                                            g_mu=0.2,
        #                                                            g_tau=3**-2,
        #                                                            std_lower=1e-10,
        #                                                            std_upper=10, 
        #                                                            std_value=.1))
        #if self.q:
        #    knodes.update(self._create_family_normal('q',
        #                                                            value=0,
        #                                                            g_mu=0.2,
        #                                                            g_tau=3**-2,
        #                                                            std_lower=1e-10,
        #                                                            std_upper=10, 
        #                                                            std_value=.1))
        return knodes

    def _create_wfpt_parents_dict(self, knodes):
        wfpt_parents = super(HDDMrl, self)._create_wfpt_parents_dict(knodes)
        
        #wfpt_parents['uncertainty'] = self.uncertainty
        wfpt_parents['alpha'] = knodes['alpha_bottom']
        wfpt_parents['dual_alpha'] = knodes['dual_alpha_bottom'] if 'dual_alpha' in self.include else 0
        #wfpt_parents['uncertainty'] = knodes['uncertainty_bottom'] if 'dual_alpha' in self.include else self.uncertainty
        #wfpt_parents['q'] = knodes['q_bottom'] if 'q' in self.include else self.q
        
        return wfpt_parents

    #use own wfpt_class, defined in the init
    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)
        return Knode(self.wfpt_rl_class, 'wfpt',
                                   observed=True, col_name=['split_by','feedback', 'response', 'rt'],
                                   **wfpt_parents)

def wienerRL_like(x, v, alpha,dual_alpha, sv, a, z, sz, t, st,q=0.5,uncertainty=0,p_outlier=0):
    
    wiener_params = {'err': 1e-4, 'n_st':2, 'n_sz':2,
                         'use_adaptive':1,
                         'simps_err':1e-3,
                         'w_outlier': 0.1}
    sum_logp = 0
    #print(uncertainty)
    #print(p_outlier)
    #print(q)
    wp = wiener_params
    #uncertainty = x['uncertainty'].iloc[0]
    response = x['response'].values.astype(int)
    #q = np.array([q,q])
    #print(q)
    feedback = x['feedback'].values
    split_by = x['split_by'].values
    #print(split_by)
    #unique = np.array([np.unique(split_by)])
    #print("v = %.2f alpha = %.2f dual_alpha = %.2f a = %.2f qup = %.2f qlow = %.2f uncertainty = %.2f t = %.2f z = %.2f sv = %.2f st = %.2f p_outlier = %.2f" % (v,alpha,dual_alpha,a,q[1],q[0],uncertainty,t,z,sv,st, p_outlier))
    return wiener_like_rlddm(x['rt'].values, response,feedback,split_by,q,alpha,dual_alpha,v,sv, a, z, sz, t, st,uncertainty, p_outlier=p_outlier, **wp)
WienerRL = stochastic_from_dist('wienerRL', wienerRL_like)
