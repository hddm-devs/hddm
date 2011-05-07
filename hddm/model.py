#!/usr/bin/python
from __future__ import division
import numpy as np
import numpy.lib.recfunctions as rec
import pymc as pm
import matplotlib.pyplot as plt
from copy import copy

import hddm
import kabuki

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
        
        super(hddm.model.Base, self).__init__(data, **kwargs)

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

        self.param_ranges = {'a_lower': .5,
                             'a_upper': 4.5,
                             'z_lower': .0,
                             'z_upper': 1.,
                             't_lower': .1,
                             't_upper': 2.,
                             'v_lower': -6.,
                             'v_upper': 6.,
                             'V_lower': 0.,
                             'V_upper': 3.,
                             'T_lower': 0.,
                             'T_upper': 1.,
                             'Z_lower': 0.,
                             'Z_upper': 1.,
                             'e_lower': -.3,
                             'e_upper': .3}

        if not init:
            # Default param ranges
            self.init_params = {'t':0.1, 'T':0.1, 'z':0.5, 'Z':0.1}
        else:
            # Compute ranges based on EZ method
            param_ranges = hddm.utils.EZ_param_ranges(self.data)
            # Overwrite set parameters
            for param,value in param_ranges.iteritems():
                self.param_ranges[param] = value
            self.init_params = hddm.utils.EZ_subjs(self.data)
        
        if init_values is not None:
            for param in init_values:
                self.init_params[param] = init_values[param]
        
        if wiener_params == None:
            self.wiener_params = {'err': 1e-4, 'nT':2, 'nZ':2, 'use_adaptive':1, 'simps_err':1e-3}
        else:          
            self.wiener_params = wiener_params

        self.param_names = self.get_param_names()
        
    def get_param_names(self):
        names = [('a',True), ('v',True), ('t',True)]
        if not self.no_bias:
            names.append(('z', True))
        if self.model_type == 'simple':
            pass
        elif self.model_type.startswith('full'):
            names += [('V',True),('Z',True),('T',True)]
            for ex in self.exclude:
                names.remove((ex, True))
            if self.no_bias and 'Z' in names:
                names.remove(('Z', True))
        else:
            raise ValueError('Model %s was not recognized' % self.model_type)
        
        names.append(('wfpt', False)) # Append likelihood parameter
        return tuple(names)



    def get_root_node(self, param, all_params, tag, data):
        """Create and return a prior distribution for [param]. [tag] is
        used in case of dependent parameters.
        """
        if param in self.init_params:
            init_val = self.init_params[param]
        else:
            init_val = None
        
        return pm.Uniform("%s%s"%(param, tag),
                          lower=self.param_ranges['%s_lower'%param],
                          upper=self.param_ranges['%s_upper'%param],
                          value=init_val)

    def get_tau_node(self, param_name, all_params, tag):
        return pm.Uniform(param_name + tag, lower=0, upper=1000)

    def get_child_node(self, param_name, parent_mean, parent_tau, subj_idx, all_params, tag, data, plot=False):
        param_full_name = '%s%s%i'%(param_name, tag, subj_idx)
        init_param_name = '%s%i'%(param_name, subj_idx)

        if init_param_name in self.init_params:
            init_val = self.init_params[init_param_name]
        else:
            init_val = None

        if param_name.startswith('e') or param_name.startswith('v'):
            return pm.Normal(param_full_name,
                             mu=parent_mean,
                             tau=parent_tau,
                             plot=plot, trace=self.trace_subjs,
                             value=init_val)

        else:
            return pm.TruncatedNormal(param_full_name,
                                      a=self.param_ranges['%s_lower'%param_name],
                                      b=self.param_ranges['%s_upper'%param_name],
                                      mu=parent_mean, tau=parent_tau,
                                      plot=plot, trace=self.trace_subjs,
                                      value=init_val)
    
    def get_rootless_child(self, param_name, tag, data, params, idx=None):
        if param_name.startswith('wfpt'):
            if self.model_type == 'simple':
                return self._get_simple(param_name+tag, data, params, idx)
            elif self.model_type == 'full_intrp':
                return self._get_full_mc(param_name+tag, data, params, idx)
            else:
                raise KeyError, "Model type %s not found." % self.model_type
        else:
            raise KeyError, "Rootless parameter named %s not found." % param_name
        
    def _get_simple(self, name, data, params, idx=None):
        return hddm.likelihoods.WienerSimple(name,
                                             value=data['rt'].flatten(), 
                                             v=params['v'], 
                                             t=params['t'], 
                                             a=params['a'], 
                                             z=self._get_idx_node('z',params),
                                             observed=True)

    def _get_full_intrp(self, name, data, params, idx=None):
        if self.wiener_params is not None:
            wp = self.wiener_params
            WienerFullIntrp = hddm.likelihoods.general_WienerFullIntrp_variable(err=wp['err'], nT=wp['nT'], nZ=wp['nZ'],
                                                               use_adaptive=wp['use_adaptive'], simps_err=wp['simps_err'])
        else:
            WienerFullIntrp =  hddm.likelihoods.WienerFullIntrp
            
        return WienerFullIntrp(name,
                             value=data['rt'].flatten(),
                             z = self._get_idx_node('z',params),
                             t = self._get_idx_node('t',params),
                             v = self._get_idx_node('v',params),
                             a = self._get_idx_node('a',params),       
                             Z = self._get_idx_node('Z',params),
                             T = self._get_idx_node('T',params),
                             V = self._get_idx_node('V',params),
                             observed=True)

    def _get_idx_node(self, node_name, params):
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
        names = list(super(self.__class__, self).get_param_names())
        names.remove(('wfpt', False))
        names += [('z_trls', False), ('v_trls', False), ('t_trls', False), ('wfpt', False)]
        return names

    def get_rootless_child(self, param_name, tag, data, params, idx=None):
        trials = data.shape[0]

        if param_name.startswith('z_trls'):
            return hddm.likelihoods.CenterUniform("z_trls%s"%tag,
                                                  center=[params['z'] for i in range(trials)],
                                                  width=[params['Z'] for i in range(trials)])

        elif param_name.startswith('v_trls'):
            return pm.Normal("v_trls%s"%tag,
                             mu=[params['v'] for i in range(trials)],
                             tau=[params['V']**-2 for i in range(trials)])

        elif param_name.startswith('t_trls'):
            return hddm.likelihoods.CenterUniform("t_trls%s"%tag,
                                                  center=[params['t'] for i in range(trials)],
                                                  width=[params['T'] for i in range(trials)])

        elif param_name.startswith('wfpt'):
            return hddm.likelihoods.WienerSingleTrial(name+tag,
                                                      value=data['rt'],
                                                      v=params['v_trls'],
                                                      t=params['t_trls'], 
                                                      a=[params['a'] for i in range(trials)],
                                                      z=params['z_trls'],
                                                      observed=True)

        else:
            raise KeyError, "Rootless child node named %s not found." % param_name
        
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

    def get_rootless_child(self, name, tag, data, params, idx=None):
        return hddm.likelihoods.LBA(name+tag,
                                    value=data['rt'].flatten(),
                                    a=params['a'],
                                    z=params['z'],
                                    t=params['t'],
                                    v0=params['v0'],
                                    v1=params['v1'],
                                    V=params['V'],
                                    normalize_v=self.normalize_v,
                                    observed=True)

    def get_root_node(self, param, all_params, tag, data):
        """Create and return a prior distribution for [param]. [tag] is
        used in case of dependent parameters.
        """
        if param == 'V' and self.fix_sv is not None: # drift rate variability
            return pm.Lambda("V%s"%tag, lambda x=self.fix_sv: x)
        else:
            return super(self.__class__, self).get_root_param(self, param, all_params, tag)

    def get_child_node(self, param_name, parent_mean, parent_tau, subj_idx, all_params, tag, data, plot=False):
        param_full_name = '%s%s%i'%(param_name, tag, subj_idx)

        if param_name.startswith('V') and self.fix_sv is not None:
            return pm.Lambda(param_full_name, lambda x=parent_mean: parent_mean,
                             plot=plot, trace=self.trace_subjs)
        else:
            return super(self.__class__, self).get_subj_param(self, param_name, parent_mean, parent_tau, subj_idx, all_params, tag, data, plot)
    
class HDDMContaminant(Base):
    
    def __init__(self, *args, **kwargs):
        Base.__init__(self, *args, **kwargs)
        self.param_names = (('a',True), ('v',True), ('z',True), ('t',True), \
                            ('pi',True), ('gamma',True), ('x', False), ('wfpt', False))
        self.param_ranges['pi_lower'] = 0.001;
        self.param_ranges['pi_upper'] = 0.2;
        self.param_ranges['gamma_lower'] = 0.001;
        self.param_ranges['gamma_upper'] = 0.999;
        
    def get_rootless_child(self, name, tag, data, params, idx=None):
        if name.startswith('wfpt'):
            return hddm.likelihoods.WienerSimpleContaminant(name+tag,
                                                            value=data['rt'],
                                                            cont_x=params['x'],
                                                            gamma=params['gamma'],
                                                            v=params['v'],
                                                            t=params['t'],
                                                            a=params['a'],
                                                            z=params['z'],
                                                            observed=True)
        elif name.startswith('x'):
            return pm.Bernoulli('x', params['pi'], size=len(data['rt']))        
        else:
            raise KeyError, "Rootless child parameter %s not found" %name

    def get_root_node(self, param_name, all_params, tag, data):
            return super(self.__class__, self).get_root_node(param_name, all_params, tag, data)

    def get_child_node(self, param_name, parent_mean, parent_tau, subj_idx, all_params, tag, data, plot=False):
            return super(self.__class__, self).get_child_node(param_name, parent_mean, parent_tau, subj_idx, all_params, tag, data, plot=False)
    

#@kabuki.hierarchical
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
