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
    This class can generate different hddms:
    - simple DDM (without inter-trial variabilities)
    - full averaging DDM (with inter-trial variabilities)
    - subject param DDM (each subject get's it's own param, see EJ's book 8.3)
    - parameter dependent on data (e.g. drift rate is dependent on stimulus
    """
    
    def __init__(self, data, model_type=None, trace_subjs=True, no_bias=True, 
                 init=False, exclude=None, wiener_params = None,
                 init_values = None, **kwargs):
        
        self.trace_subjs = trace_subjs
        self.no_bias = no_bias

        if model_type is None:
            self.model_type = 'simple'
        else:
            self.model_type = model_type
        
        # Flip sign for lower boundary RTs
        self.data = hddm.utils.flip_errors(data)

        if exclude is None:
            self.exclude = []
        else:
            self.exclude = exclude

        if init:
            raise NotImplemented, "TODO"
            # Compute ranges based on EZ method
            param_ranges = hddm.utils.EZ_param_ranges(self.data)
            # Overwrite set parameters
            for param,value in param_ranges.iteritems():
                self.param_ranges[param] = value
            self.init_params = hddm.utils.EZ_subjs(self.data)
        
        if init_values is not None:
            raise NotImplemented, "TODO"

        if wiener_params == None:
            self.wiener_params = {'err': 1e-4, 'nT':2, 'nZ':2, 'use_adaptive':1, 'simps_err':1e-3}
        else:          
            self.wiener_params = wiener_params
        
        self.params = self.get_params()
        super(hddm.model.Base, self).__init__(data, **kwargs)

    def get_params(self):
        params = [Parameter('a',True, lower=.5, upper=4.5),
                  Parameter('v',True, lower=-6., upper=6.), 
                  Parameter('t',True, lower=.1, upper=2., init=.1)]
        if not self.no_bias:
            params.append(Parameter('z', True, lower=0., upper=1., init=.5))
        if self.model_type == 'simple':
            pass
        elif self.model_type.startswith('full'):
            if 'V' not in self.exclude:
                params.append(Parameter('V',True, lower=0., upper=6.))
            if 'Z' not in self.exclude:
                params.append(Parameter('Z',True, lower=0., upper=1., init=.1))
            if 'T' not in self.exclude:
                params.append(Parameter('T',True, lower=0., upper=1., init=.1))
        else:
            raise ValueError('Model %s was not recognized' % self.model_type)
        
        params.append(Parameter('wfpt', False)) # Append likelihood parameter
        return params
    
    def get_root_node(self, param):
        """Create and return a prior distribution for [param].
        """
        return pm.Uniform(param.full_name,
                          lower=param.lower,
                          upper=param.upper,
                          value=param.init)

    def get_tau_node(self, param):
        return pm.Uniform(param.full_name, lower=0, upper=10)

    def get_child_node(self, param, plot=False):
        if param.name.startswith('e') or param.name.startswith('v'):
            return pm.Normal(param.full_name,
                             mu=param.root,
                             tau=param.tau**-2,
                             plot=plot, trace=self.trace_subjs,
                             value=param.init)

        else:
            return pm.TruncatedNormal(param.full_name,
                                      a=param.lower,
                                      b=param.upper,
                                      mu=param.root, 
                                      tau=param.tau**-2,
                                      plot=plot, trace=self.trace_subjs,
                                      value=param.init)
    
    def get_rootless_child(self, param, params):
        if param.name.startswith('wfpt'):
            if self.model_type == 'simple':
                return self._get_simple(param.full_name, param.data, params)
            elif self.model_type == 'full_intrp':
                return self._get_full_mc(param.full_name, param.data, params)
            else:
                raise KeyError, "Model type %s not found." % self.model_type
        else:
            raise KeyError, "Rootless parameter named %s not found." % param.name
        
    def _get_simple(self, name, data, params):
        return hddm.likelihoods.WienerSimple(name,
                                             value=data['rt'].flatten(),
                                             v=params['v'], 
                                             a=params['a'], 
                                             z=self._get_node('z',params),
                                             t=params['t'], 
                                             observed=True)

    def _get_full_intrp(self, name, data, params):
        if self.wiener_params is not None:
            wp = self.wiener_params
            WienerFullIntrp = hddm.likelihoods.general_WienerFullIntrp_variable(err=wp['err'], nT=wp['nT'], nZ=wp['nZ'],
                                                               use_adaptive=wp['use_adaptive'], simps_err=wp['simps_err'])
        else:
            WienerFullIntrp =  hddm.likelihoods.WienerFullIntrp
            
        return WienerFullIntrp(name,
                             value=data['rt'].flatten(),
                             v = params['v'],
                             a = params['a'],
                             z = self._get_node('z',params),
                             t = params['t'],
                             Z = self._get_node('Z',params),
                             T = self._get_node('T',params),
                             V = self._get_node('V',params),
                             observed=True)

    def _get_node(self, node_name, params):
        if node_name in self.exclude:
            return 0
        elif node_name=='z' and self.no_bias:
            return 0.5
        else:
            return params[node_name]



class HDDM(Base):
    pass

class HDDMFullExtended(Base):
    def get_param_names(self):
        self.model_type = 'full_expanded'
        params = [Parameter('z_trls', False),
                  Parameter('v_trls', False),
                  Parameter('t_trls', False)]
        params += list(super(self.__class__, self).get_param_names())

        return params

    def get_rootless_child(self, param, params):
        trials = len(param.data)

        if param.name.startswith('z_trls'):
            return hddm.likelihoods.CenterUniform(param.full_name,
                                                  center=[params['z'] for i in range(trials)],
                                                  width=[params['Z'] for i in range(trials)])

        elif param.name.startswith('v_trls'):
            return pm.Normal(param.full_name,
                             mu=[params['v'] for i in range(trials)],
                             tau=[params['V']**-2 for i in range(trials)])

        elif param.name.startswith('t_trls'):
            return hddm.likelihoods.CenterUniform(param.full_name,
                                                  center=[params['t'] for i in range(trials)],
                                                  width=[params['T'] for i in range(trials)])

        elif param.name.startswith('wfpt'):
            return hddm.likelihoods.WienerSingleTrial(param.full_name,
                                                      value=param.data['rt'],
                                                      v=params['v_trls'],
                                                      t=params['t_trls'], 
                                                      a=[params['a'] for i in range(trials)],
                                                      z=params['z_trls'],
                                                      observed=True)
        else:
            raise KeyError, "Rootless child node named %s not found." % param.name
        
class HLBA(Base):
    param_names = (('a',True), ('z',True), ('t',True), ('V',True), ('v0',True), ('v1',True), ('lba',False))

    def __init__(self, data, model_type=None, trace_subjs=True, normalize_v=True, no_bias=True, fix_sv=None, init=False, exclude=None, **kwargs):
        super(self.__class__, self).__init__(data, **kwargs)

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

    def get_rootless_child(self, param, params):
        return hddm.likelihoods.LBA(param.full_name,
                                    value=param.data['rt'],
                                    a=params['a'],
                                    z=params['z'],
                                    t=params['t'],
                                    v0=params['v0'],
                                    v1=params['v1'],
                                    V=params['V'],
                                    normalize_v=self.normalize_v,
                                    observed=True)

    def get_root_node(self, param):
        """Create and return a prior distribution for [param]. [tag] is
        used in case of dependent parameters.
        """
        if param.name == 'V' and self.fix_sv is not None: # drift rate variability
            return pm.Lambda(param.full_name, lambda x=self.fix_sv: x)
        else:
            return super(self.__class__, self).get_root_param(self, param)

    def get_child_node(self, param, plot=False):
        if param.name.startswith('V') and self.fix_sv is not None:
            return pm.Lambda(param.full_name, lambda x=param.root: x,
                             plot=plot, trace=self.trace_subjs)
        else:
            return super(self.__class__, self).get_child_node(param, plot=plot)
    
class HDDMContaminant(Base):
    def __init__(self, *args, **kwargs):
        super(HDDMContaminant, self).__init__(self, *args, **kwargs)
        self.params = (Parameter('a',True, lower=.5, upper=4.5),
                       Parameter('v',True, lower=-6., upper=6.), 
                       Parameter('z',True, lower=0., upper=1., init=.5), 
                       Parameter('t',True, lower=.1, upper=2., init=.1),
                       Parameter('pi',True, lower=0.001, upper=0.2),
                       Parameter('gamma',True, lower=0.001, upper=0.999),
                       Parameter('x', False), 
                       Parameter('wfpt', False))

    def get_rootless_child(self, param, params):
        if param.name.startswith('wfpt'):
            return hddm.likelihoods.WienerSimpleContaminant(param.full_name,
                                                            value=param.data['rt'],
                                                            cont_x=params['x'],
                                                            gamma=params['gamma'],
                                                            v=params['v'],
                                                            t=params['t'],
                                                            a=params['a'],
                                                            z=params['z'],
                                                            observed=True)
        elif param.name.startswith('x'):
            return pm.Bernoulli(param.full_name, p=params['pi'], size=len(param.data['rt'])) 
        else:
            raise KeyError, "Rootless child parameter %s not found" %name


class HDDMOneRegressor(Base):
    def __init__(self, *args, **kwargs):
        """Hierarchical Drift Diffusion Model analyses for Cavenagh et al, IP.

        Arguments:
        ==========
        data: structured numpy array containing columns: subj_idx, response, RT, theta, dbs

        Keyword Arguments:
        ==================
        effect_on <list>: theta and dbs effect these DDM parameters.
        depends_on <list>: separate stimulus distributions for these parameters.

        Example:
        ========
        The following will create and fit a model on the dataset data, theta and dbs affect the threshold. For each stimulus,
        there are separate drift parameter, while there is a separate HighConflict and LowConflict threshold parameter. The effect coding type is dummy.

        model = HDDM_regress_multi(data, effect_on=['a'], depend_on=['v', 'a'], effect_coding=False, HL_on=['a'])
        model.mcmc()
        """
        # Fish out keyword arguments that should not get passed on
        # to the parent.
        if kwargs.has_key('effect_on'):
            self.effect_on = kwargs['effect_on']
            del kwargs['effect_on']
        else:
            self.effect_on = []

        if kwargs.has_key('e_data'):
            self.e1_data = kwargs['e_data']
            del kwargs['e_data']
        else:
            raise ValueError, "Provide e_data parameter"

        self.effect_id = 0

        super(self.__class__, self).__init__(*args, **kwargs)
        
    def get_param_names(self):
        param_names = list(super(self.__class__, self).get_param_names())
        param_names += ('e', True)
        return tuple(param_names)

    def get_model(self, model_name, data, params, idx=None):
        """Generate the HDDM."""
        data = copy(data)
        params_subj = {}
        for name, param in params.iteritems():
            params_subj[name] = param[idx]
        self.effect_id += 1

        for effect in self.effect_on:
            # Create actual effect on base values, result is a matrix.
            params_subj[effect] = pm.Lambda('e_inst_%s_%i_%i'%(effect,idx,self.effect_id),
                                            lambda base=params_subj[effect],
                                            e=params_subj['e']:
                                            base + data[self.e_data]*e,
                                            plot=False)

        model = hddm.likelihoods.WienerSimpleMulti(model_name,
                                                   value=data['rt'],
                                                   v=params_subj['v'],
                                                   t=params_subj['t'],
                                                   a=params_subj['a'],
                                                   z=params_subj['z'],
                                                   multi=self.effect_on,
                                                   observed=True)

        return model, params_subj, data

class HDDMTwoRegressor(Base):
    def __init__(self, *args, **kwargs):
        """Hierarchical Drift Diffusion Model analyses for Cavenagh et al, IP.

        Arguments:
        ==========
        data: structured numpy array containing columns: subj_idx, response, RT, theta, dbs

        Keyword Arguments:
        ==================
        effect_on <list>: theta and dbs effect these DDM parameters.
        depends_on <list>: separate stimulus distributions for these parameters.

        Example:
        ========
        The following will create and fit a model on the dataset data, theta and dbs affect the threshold. For each stimulus,
        there are separate drift parameter, while there is a separate HighConflict and LowConflict threshold parameter. The effect coding type is dummy.

        model = HDDM_regress_multi(data, effect_on=['a'], depend_on=['v', 'a'], effect_coding=False, HL_on=['a'])
        model.mcmc()
        """
        # Fish out keyword arguments that should not get passed on
        # to the parent.
        if kwargs.has_key('effect_on'):
            self.effect_on = kwargs['effect_on']
            del kwargs['effect_on']
        else:
            self.effect_on = []

        if kwargs.has_key('e1_data'):
            self.e1_data = kwargs['e1_data']
            del kwargs['e1_data']
        else:
            raise ValueError, "Provide e1_data parameter"

        if kwargs.has_key('e2_data'):
            self.e1_data = kwargs['e2_data']
            del kwargs['e2_data']
        else:
            raise ValueError, "Provide e2_data parameter"

        self.effect_id = 0

        super(self.__class__, self).__init__(*args, **kwargs)
        
    def get_param_names(self):
        param_names = list(super(self.__class__, self).get_param_names())
        param_names += ('e1','e2', 'e_inter')
        return tuple(param_names)

    def get_model(self, model_name, data, params, idx=None):
        """Generate the HDDM."""
        data = copy(data)
        params_subj = {}
        for name, param in params.iteritems():
            params_subj[name] = param[idx]
        self.effect_id += 1

        for effect in self.effect_on:
            # Create actual effect on base values, result is a matrix.
            params_subj[effect] = pm.Lambda('e_inst_%s_%i_%i'%(effect,idx,self.effect_id),
                                            lambda base=params_subj[effect],
                                            e1=params_subj['e1'],
                                            e2=params_subj['e2'],
                                            e_inter=params_subj['e_inter']:
                                            base + data[self.e1_data]*e1 + data[self.e2_data]*e2 + data[self.e1_data]*data[e2_data]*e_inter,
                                            plot=False)

        model = hddm.likelihoods.WienerSimpleMulti(model_name,
                                                   value=data['rt'],
                                                   v=params_subj['v'],
                                                   t=params_subj['t'],
                                                   a=params_subj['a'],
                                                   z=params_subj['z'],
                                                   multi=self.effect_on,
                                                   observed=True)

        return model, params_subj, data


class HDDMAntisaccade(Base):
    param_names = (('v',True),
                   ('v_switch', True),
                   ('a', True),
                   ('z', True),
                   ('t', True),
                   ('t_switch', True))

    def __init__(self, data, no_bias=True, init=True, **kwargs):
        super(self.__class__, self).__init__(data, **kwargs)
        
        if 'instruct' not in self.data.dtype.names:
            raise AttributeError, 'data has to contain a field name instruct.'

        if not init:
            # Default param ranges
            self.init_params = {'t':0.1, 'z':0.5, 'v':-2., 'v_switch':1, 't_switch':.2}

        self.param_ranges = {'a_lower': .5,
                             'a_upper': 4.5,
                             'z_lower': .0,
                             'z_upper': 1.,
                             't_lower': .1,
                             't_upper': 1.,
                             't_switch_lower': .1,
                             't_switch_upper': 1.,
                             'v_lower': -3.,
                             'v_upper': 0.,
                             'v_switch_lower': 0.,
                             'v_switch_upper': 3.,
                             'e_lower': -.3,
                             'e_upper': .3}
            
    def get_rootless_child(self, name, tag, data, params, idx=None):
        return hddm.likelihoods.WienerAntisaccade(name+tag,
                                                  value=data['rt'],
                                                  instruct=data['instruct'],
                                                  v=params['v'],
                                                  v_switch=params['v_switch'],
                                                  a=params['a'],
                                                  z=params['z'],
                                                  t=params['t'],
                                                  t_switch=params['t_switch'],
                                                  observed=True)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
