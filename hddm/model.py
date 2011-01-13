#!/usr/bin/python
from __future__ import division
import numpy as np
import numpy.lib.recfunctions as rec
import pymc as pm
import matplotlib.pyplot as plt
from copy import copy

import hddm
import kabuki

@kabuki.hierarchical
class HDDM(object):
    """
    This class can generate different hddms:
    - simple DDM (without inter-trial variabilities)
    - full averaging DDM (with inter-trial variabilities)
    - subject param DDM (each subject get's it's own param, see EJ's book 8.3)
    - parameter dependent on data (e.g. drift rate is dependent on stimulus
    """
    
    def __init__(self, data, model_type=None, trace_subjs=True, normalize_v=True, no_bias=True, fix_sv=None, init=True):
        self.trace_subjs = trace_subjs

        if model_type is None:
            self.model_type = 'simple'
        else:
            self.model_type = model_type
        
        self.no_bias = no_bias
        self.fix_sv = fix_sv

        # Flip sign for lower boundary RTs
        self.data = hddm.utils.flip_errors(data)


        # Set function map
        self._models = {'simple': self._get_simple,
                        'simple_gpu': self._get_simple_gpu,
                        'full_mc': self._get_full_mc,
                        'full': self._get_full,
                        'lba':self._get_lba}

        if self.model_type != 'lba':
            self.param_ranges = {'a_lower': .5,
                                 'a_upper': 4.5,
                                 'z_lower': .0,
                                 'z_upper': 1.,
                                 't_lower': .1,
                                 't_upper': 1.,
                                 'v_lower': -3.,
                                 'v_upper': 3.,
                                 'T_lower': 0.,
                                 'T_upper': 1.,
                                 'Z_lower': 0.,
                                 'Z_upper': 1.,
                                 'e_lower': -.5,
                                 'e_upper': .5
                                 }
            if not init:
                # Default param ranges
                self.init_params = {}
            else:
                # Compute ranges based on EZ method
                param_ranges = hddm.utils.EZ_param_ranges(self.data)
                # Overwrite set parameters
                for param,value in param_ranges.iteritems():
                    self.param_ranges[param] = value
                self.init_params = hddm.utils.EZ_subjs(self.data)
                
            self.normalize_v = False
            self.fix_sv = None
            
        else:
            # LBA model
            self.normalize_v = normalize_v
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
            
            if self.normalize_v:
                self.param_ranges['v_lower'] = 0.
                self.param_ranges['v_upper'] = 1.


    def get_param_names(self):
        if self.model_type == 'simple' or self.model_type == 'simple_gpu':
            return ('a', 'v', 'z', 't')
        elif self.model_type == 'full_mc' or self.model_type == 'full':
            return ('a', 'v', 'V', 'z', 'Z', 't', 'T')
        elif self.model_type == 'lba':
            return ('a', 'z', 't', 'V', 'v0', 'v1')
        else:
            raise ValueError('Model %s not recognized' % self.model_type)

    def get_model(self, *args, **kwargs):
        return self._models[self.model_type](*args, **kwargs)
    
    def get_root_param(self, param, tag=None):
        """Create and return a prior distribution for [param]. [tag] is
        used in case of dependent parameters.
        """
        if tag is None:
            tag = ''
        else:
            tag = '_' + tag

        if param in self.init_params:
            init_val = self.init_params[param]
        else:
            init_val = None
            
        if param == 'V' and self.fix_sv is not None: # drift rate variability
            return pm.Lambda("V%s"%tag, lambda x=self.fix_sv: x)

        elif param == 'z' and self.no_bias: # starting point position (bias)
            return pm.Lambda("z%s"%tag, lambda x=param: .5) #None # z = a/2.

        else:
            return pm.Uniform("%s%s"%(param, tag),
                              lower=self.param_ranges['%s_lower'%param],
                              upper=self.param_ranges['%s_upper'%param],
                              value=init_val)

    def get_tau_param(self, param, tag=None):
        if tag is None:
            tag = '_tau'
        else:
            tag = tag + '_tau'

        return pm.Uniform(param + tag, lower=0, upper=800, plot=False)

    def get_subj_param(self, param_name, parent_mean, parent_tau, subj_idx):
        if len(param_name) != 1: # if there is a tag attached to the param
            param = param_name[0]
            tag = param_name[1:] + '_' + str(subj_idx) # create new name for the subj parameter
        else:
            param = param_name
            tag = '_' + str(subj_idx)

        init_param_name = '%s_%i'%(param_name,subj_idx)
        if init_param_name in self.init_params:
            init_val = self.init_params[init_param_name]
        else:
            init_val = None

        return self.get_child_param(param, parent_mean, parent_tau, tag=tag, init_val=init_val)

    def get_child_param(self, param, parent_mean, parent_tau, tag=None, init_val=None, plot=False):
        if tag is None:
            tag = ''

        if not tag.startswith('_'):
            tag = '_'+tag
    
        if param == 'V' and self.fix_sv is not None:
            return pm.Lambda("V%s"%tag, lambda x=parent_mean: parent_mean,
                             plot=plot, trace=self.trace_subjs)

        elif param == 'z' and self.no_bias:
            return pm.Lambda("z%s"%tag, lambda x=parent_mean: .5,
                             plot=plot, trace=self.trace_subjs)

        elif param == 'e' or param.startswith('v'):
            return pm.Normal("%s%s"%(param,tag),
                             mu=parent_mean,
                             tau=parent_tau,
                             plot=plot, trace=self.trace_subjs,
                             value=init_val)

        else:
            return pm.TruncatedNormal("%s%s"%(param,tag),
                                      a=self.param_ranges['%s_lower'%param],
                                      b=self.param_ranges['%s_upper'%param],
                                      mu=parent_mean, tau=parent_tau,
                                      plot=plot, trace=self.trace_subjs,
                                      value=init_val)



    def _get_simple(self, name, data, params, idx=None):
        if idx is None:
            return hddm.likelihoods.WienerSimple(name,
                                                 value=data['rt'].flatten(), 
                                                 v=params['v'], 
                                                 ter=params['t'], 
                                                 a=params['a'], 
                                                 z=params['z'],
                                                 observed=True)
        else:
            return hddm.likelihoods.WienerSimple(name,
                                value=data['rt'].flatten(), 
                                v=params['v'][idx], 
                                ter=params['t'][idx], 
                                a=params['a'][idx], 
                                z=params['z'][idx],
                                observed=True)


    def _get_simple_gpu(self, name, data, params, idx=None):
        if idx is None:
            return hddm.likelihoods.WienerGPUSingle(name,
                                   value=data['rt'].flatten(), 
                                   v=params['v'], 
                                   ter=params['t'], 
                                   a=params['a'], 
                                   z=params['z'],
                                   observed=True)
        else:
            return hddm.likelihoods.WienerGPUSingle(name,
                                   value=data['rt'].flatten(), 
                                   v=params['v'][idx], 
                                   ter=params['t'][idx], 
                                   a=params['a'][idx],
                                   z=params['z'][idx],
                                   observed=True)

    def _get_full_mc(self, name, data, params, idx=None):
        if idx is None:
            return hddm.likelihoods.WienerAvg(name,
                             value=data['rt'].flatten(), 
                             v=params['v'], 
                             sv=params['V'],
                             ter=params['t'],
                             ster=params['T'], 
                             a=params['a'],
                             z=params['z'],
                             sz=params['Z'],
                             observed=True)

        else:
            return hddm.likelihoods.WienerAvg(name,
                             value=data['rt'].flatten(), 
                             v=params['v'][idx], 
                             sv=params['V'][idx],
                             ter=params['t'][idx],
                             ster=params['T'][idx], 
                             a=params['a'][idx],
                             z=params['z'][idx],
                             sz=params['Z'][idx],
                             observed=True)


    def _get_full(self, name, data, params, idx=None):
        if idx is None:
            trials = data.shape[0]
            ddm[i] = np.empty(trials, dtype=object)
            z_trial = np.empty(trials, dtype=object)
            v_trial = np.empty(trials, dtype=object)
            ter_trial = np.empty(trials, dtype=object)
            for trl in range(trials):
                z_trial[trl] = hddm.likelihoods.CenterUniform("z_%i"%trl,
                                             center=params['z'],
                                             width=params['sz'],
                                             plot=False, observed=False, trace=False)
                v_trial[trl] = pm.Normal("v_%i"%trl,
                                         mu=params['v'],
                                         tau=1/(params['sv']**2),
                                         plot=False, observed=False, trace=False)
                ter_trial[trl] = hddm.likelihoods.CenterUniform("ter_%i"%trl,
                                               center=params['ter'],
                                               width=params['ster'],
                                               plot=False, observed=False, trace=False)
                ddm[i][trl] = hddm.likelihoods.Wiener2("ddm_%i_%i"%(trl, i),
                                      value=data['rt'].flatten()[trl],
                                      v=v_trial[trl],
                                      ter=ter_trial[trl], 
                                      a=param['a'],
                                      z=z_trial[trl],
                                      observed=True, trace=False)

            return ddm

        else:
            trials = data.shape[0]
            ddm = np.empty(trials, dtype=object)
            z_trial = np.empty(trials, dtype=object)
            v_trial = np.empty(trials, dtype=object)
            ter_trial = np.empty(trials, dtype=object)
            for trl in range(trials):
                z_trial[trl] = hddm.likelihoods.CenterUniform("z_%i"%trl,
                                             center=params['z'],
                                             width=params['sz'],
                                             plot=False, observed=False)
                v_trial[trl] = pm.Normal("v_%i"%trl,
                                         mu=params['v'],
                                         tau=1/(params['sv']**2),
                                         plot=False, observed=False)
                ter_trial[trl] = hddm.likelihoods.CenterUniform("ter_%i"%trl,
                                               center=params['ter'],
                                               width=params['ster'],
                                               plot=False, observed=False)
                ddm[trl] = hddm.likelihoods.Wiener2("ddm_%i"%trl,
                                   value=data['rt'].flatten()[trl],
                                   v=v_trial[trl],
                                   ter=ter_trial[trl], 
                                   a=param['a'],
                                   z=z_trial[trl],
                                   observed=True)

            return ddm

    def _get_lba(self, name, data, params, idx=None):
        if idx is None:
            return hddm.likelihoods.LBA(name,
                                        value=data['rt'].flatten(),
                                        a=params['a'],
                                        z=params['z'],
                                        ter=params['t'],
                                        v0=params['v0'],
                                        v1=params['v1'],
                                        sv=params['V'],
                                        normalize_v=self.normalize_v,
                                        observed=True)
        else:
            return hddm.likelihoods.LBA(name,
                                        value=data['rt'].flatten(),
                                        a=params['a'][idx],
                                        z=params['z'][idx],
                                        ter=params['t'][idx],
                                        v0=params['v0'][idx],
                                        v1=params['v1'][idx],
                                        sv=params['V'][idx],
                                        normalize_v=self.normalize_v,
                                        observed=True)    



if __name__ == "__main__":
    import doctest
    doctest.testmod()