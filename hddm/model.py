#!/usr/bin/python
from __future__ import division
import numpy as np
import numpy.lib.recfunctions as rec
import pymc as pm
import matplotlib.pyplot as plt
from copy import copy

import hddm

import kabuki

from kabuki.hierarchical import Parameter

class Base(kabuki.Hierarchical):
    """
    
    """
    
    def __init__(self, data, trace_subjs=True, no_bias=True, 
                 init=False, include=(), wiener_params = None,
                 init_values=None, **kwargs):
        
        # Flip sign for lower boundary RTs
        self.data = hddm.utils.flip_errors(data)

        include = set(include)
        
        if include is not None:
            if include == 'all':
                [include.add(param) for param in ('T','V','Z')]
            else:
                [include.add(param) for param in include]

        self.no_bias = no_bias
        
        if not self.no_bias:
            include.add('z')

        if init:
            raise NotImplementedError, "TODO"
            # Compute ranges based on EZ method
            param_ranges = hddm.utils.EZ_param_ranges(self.data)
            # Overwrite set parameters
            for param,value in param_ranges.iteritems():
                self.param_ranges[param] = value
            self.init_params = hddm.utils.EZ_subjs(self.data)
        
        if init_values is not None:
            raise NotImplementedError, "TODO"

        if wiener_params == None:
            self.wiener_params = {'err': 1e-4, 'nT':2, 'nZ':2, 'use_adaptive':1, 'simps_err':1e-3}
        else:          
            self.wiener_params = wiener_params
        
        self.params = self.get_params()

        super(hddm.model.Base, self).__init__(data, include=include, **kwargs)

    def get_params(self):
        params = [Parameter('a', True, lower=.5, upper=4.5),
                  Parameter('v', True, lower=-10., upper=10.), 
                  Parameter('t', True, lower=.1, upper=2., init=.1),
                  Parameter('z', True, lower=0., upper=1., init=.5, default=.5, optional=True),
                  Parameter('V', True, lower=0., upper=6., default=0, optional=True),
                  Parameter('Z', True, lower=0., upper=1., init=.1, default=0, optional=True),
                  Parameter('T', True, lower=0., upper=2., init=.1, default=0, optional=True),
                  Parameter('wfpt', False)]
        
        return params
    
    def get_root_node(self, param):
        """Create and return a prior distribution for [param].
        """
        return pm.Uniform(param.full_name,
                          lower=param.lower,
                          upper=param.upper,
                          value=param.init,
                          verbose=param.verbose)

    def get_tau_node(self, param):
        return pm.Uniform(param.full_name, lower=0., upper=1., value=.1, plot=self.plot_tau)

    def get_child_node(self, param):
        if param.name.startswith('e') or param.name.startswith('v'):
            return pm.Normal(param.full_name,
                             mu=param.root,
                             tau=param.tau**-2,
                             plot=self.plot_subjs, trace=self.trace_subjs,
                             value=param.init)

        else:
            return pm.TruncatedNormal(param.full_name,
                                      a=param.lower,
                                      b=param.upper,
                                      mu=param.root, 
                                      tau=param.tau**-2,
                                      plot=self.plot_subjs, trace=self.trace_subjs,
                                      value=param.init)
    
    def get_rootless_child(self, param, params):
        if param.name == 'wfpt':
            if self.wiener_params is not None:
                wp = self.wiener_params
                WienerFullIntrp = hddm.likelihoods.general_WienerFullIntrp_variable(err=wp['err'], nT=wp['nT'], nZ=wp['nZ'],
                                                                                    use_adaptive=wp['use_adaptive'], simps_err=wp['simps_err'])
            else:
                WienerFullIntrp =  hddm.likelihoods.WienerFullIntrp
            
            return WienerFullIntrp(param.full_name,
                                   value=param.data['rt'].flatten(),
                                   v = params['v'],
                                   a = params['a'],
                                   z = self.get_node('z',params),
                                   t = params['t'],
                                   Z = self.get_node('Z',params),
                                   T = self.get_node('T',params),
                                   V = self.get_node('V',params),
                                   observed=True)

        else:
            raise KeyError, "Rootless parameter named %s not found." % param.name


class HDDM(Base):
    pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
