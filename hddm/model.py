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
                 init=False, exclude_inter_var_params=None, wiener_params = None,
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
            
        # Set function map
        self._models = {'simple': self._get_simple,
                        'full_mc': self._get_full_mc,
                        'full_intrp': self._get_full_intrp,
                        'full': self._get_full}

        if exclude_inter_var_params is None:
            self.exclude = []
        else:
            self.exclude = exclude_inter_var_params

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
            
        self.wiener_params = wiener_params

        
    def get_param_names(self):
        if self.model_type == 'simple':
            return ('a', 'v', 'z', 't')
        elif self.model_type == 'full_mc' or self.model_type == 'full' or self.model_type== 'full_intrp':
            names = set(['a', 'v', 'V', 'z', 'Z', 't', 'T'])
            for ex in self.exclude:
                names.remove(ex)
            return tuple(names)
        else:
            raise ValueError('Model %s not recognized' % self.model_type)

    param_names = property(get_param_names)
    
    def get_observed(self, *args, **kwargs):
        return self._models[self.model_type](*args, **kwargs)
    
    def get_root_node(self, param, all_params, tag, pos=None):
        """Create and return a prior distribution for [param]. [tag] is
        used in case of dependent parameters.
        """
        if param in self.init_params:
            init_val = self.init_params[param]
        else:
            init_val = None
        
        if param == 'z' and self.no_bias: # starting point position (bias)
            return pm.Deterministic(hddm.utils.return_fixed,
                                    'z%s'%tag,
                                    'z%s'%tag,
                                    parents={},
                                    plot=False)

        elif param == 'Z':
            return pm.Uniform("%s%s"%(param, tag),
                              lower=self.param_ranges['%s_lower'%param[0]],
                              upper=1.,
                              value=init_val)
        elif param == 'T':
            return pm.Uniform("%s%s"%(param, tag),
                              lower=self.param_ranges['%s_lower'%param[0]],
                              upper=self.param_ranges['%s_upper'%param[0]],
                              value=init_val)
        else:
            return pm.Uniform("%s%s"%(param, tag),
                              lower=self.param_ranges['%s_lower'%param[0]],
                              upper=self.param_ranges['%s_upper'%param[0]],
                              value=init_val)

    def get_tau_node(self, param_name, all_params, tag):
        return pm.Uniform(param_name + tag, lower=0, upper=1000, plot=False)

    def get_child_node(self, param_name, parent_mean, parent_tau, subj_idx, all_params, tag, data, pos=None, plot=False):
        param_full_name = '%s%s%i'%(param_name, tag, subj_idx)
        init_param_name = '%s%i'%(param_name, subj_idx)

        if init_param_name in self.init_params:
            init_val = self.init_params[init_param_name]
        else:
            init_val = None

        if param_name.startswith('z') and self.no_bias:
            return pm.Deterministic(hddm.utils.return_fixed,
                                    param_full_name,
                                    param_full_name,
                                    parents={},
                                    plot=False)

        elif param_name.startswith('Z'):
            return pm.TruncatedNormal(param_full_name,
                                      mu=parent_mean,
                                      tau=parent_tau,
                                      a=0,
                                      b=1.,
                                      plot=plot, trace=self.trace_subjs,
                                      value=init_val)

        elif param_name.startswith('T'):
            return pm.TruncatedNormal(param_full_name,
                                      mu=parent_mean,
                                      tau=parent_tau,
                                      a=0,
                                      b=self.param_ranges['T_upper'],
                                      plot=plot, trace=self.trace_subjs,
                                      value=init_val)

        elif param_name.startswith('e') or param_name.startswith('v'):
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

    def _get_simple(self, name, data, params, idx=None):
        return hddm.likelihoods.WienerSimple(name,
                                             value=data['rt'].flatten(), 
                                             v=params['v'], 
                                             t=params['t'], 
                                             a=params['a'], 
                                             z=params['z'],
                                             observed=True)

    def _get_full_mc(self, name, data, params, idx=None):
        return hddm.likelihoods.WienerFullMc(name,
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
        else:
            return params[node_name]

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


    def _get_full(self, name, data, params, idx=None):
        trials = data.shape[0]
        
        if idx is None:
            tag = ''
        else:
            tag = str(idx)

        z_trials = hddm.likelihoods.CenterUniform("z_trls%s"%tag,
                                                  center=[params['z'] for i in range(trials)],
                                                  width=[params['Z'] for i in range(trials)])

        v_trials = pm.Normal("v_trls%s"%tag,
                             mu=[params['v'] for i in range(trials)],
                             tau=[params['V']**-2 for i in range(trials)])
        
        ter_trials = hddm.likelihoods.CenterUniform("t_trls%s"%tag,
                                                    center=[params['t'] for i in range(trials)],
                                                    width=[params['T'] for i in range(trials)])

        ddm = hddm.likelihoods.WienerSingleTrial(name,
                                                 value=data['rt'],
                                                 v=v_trials,
                                                 t=ter_trials, 
                                                 a=[params['a'] for i in range(trials)],
                                                 z=z_trials,
                                                 observed=True)

        return [ddm, ter_trials, v_trials, z_trials]




#@kabuki.hierarchical
class HDDM(Base):
    pass

class HLBA(Base):
    param_names = ('a', 'z', 't', 'V', 'v0', 'v1')

    def __init__(self, data, model_type=None, trace_subjs=True, normalize_v=True, no_bias=True, fix_sv=None, init=False, exclude_inter_var_params=None, **kwargs):
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

    def get_observed(self, name, data, params, idx=None):
        return hddm.likelihoods.LBA(name,
                                    value=data['rt'].flatten(),
                                    a=params['a'],
                                    z=params['z'],
                                    t=params['t'],
                                    v0=params['v0'],
                                    v1=params['v1'],
                                    V=params['V'],
                                    normalize_v=self.normalize_v,
                                    observed=True)

    def get_root_node(self, param, all_params, tag, pos=None):
        """Create and return a prior distribution for [param]. [tag] is
        used in case of dependent parameters.
        """
        if param == 'V' and self.fix_sv is not None: # drift rate variability
            return pm.Lambda("V%s"%tag, lambda x=self.fix_sv: x)
        else:
            return super(self.__class__, self).get_root_param(self, param, all_params, tag, pos=None)

    def get_child_node(self, param_name, parent_mean, parent_tau, subj_idx, all_params, tag, data, pos=None, plot=False):
        param_full_name = '%s%s%i'%(param_name, tag, subj_idx)

        if param_name.startswith('V') and self.fix_sv is not None:
            return pm.Lambda(param_full_name, lambda x=parent_mean: parent_mean,
                             plot=plot, trace=self.trace_subjs)
        else:
            return super(self.__class__, self).get_subj_param(self, param_name, parent_mean, parent_tau, subj_idx, all_params, tag, data, pos, plot)
    
#@kabuki.hierarchical
class HDDMContaminant(Base):
    param_names = ('a', 'v', 'z', 't', 'pi', 'gamma')

    def get_observed(self, model_name, data, params, idx=None):
        return hddm.likelihoods.WienerSimpleContaminant(model_name,
                                                        value=data['rt'],
                                                        cont_x=params['pi'],
                                                        cont_y=params['gamma'],
                                                        v=params['v'],
                                                        t=params['t'],
                                                        a=params['a'],
                                                        z=params['z'],
                                                        observed=True)

    def get_root_node(self, param_name, all_params, tag, data, pos=None):
        if param_name == 'pi':
            return pm.Uniform('%s%s'%(param_name,tag), lower=0, upper=1)
        elif param_name == 'gamma':
            return pm.Uniform('%s%s'%(param_name,tag), lower=0, upper=1)
        else:
            return super(self.__class__, self).get_root_param(param_name, all_params, tag, pos=pos)

    def get_child_node(self, param_name, parent_mean, parent_tau, subj_idx, all_params, tag, data, pos=None, plot=False):
        if param_name == 'pi':
            return pm.Bernoulli('%s%s%i'%(param_name, tag, subj_idx), p=[parent_mean for i in range(len(data))])
        elif param_name == 'gamma':
            return pm.Bernoulli('%s%s%i'%(param_name, tag, subj_idx), p=[parent_mean for i in range(len(data))])
        else:
            return super(self.__class__, self).get_subj_param(param_name, parent_mean, parent_tau, subj_idx, all_params, tag, data, pos=None, plot=False)
    

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
        param_names = super(self.__class__, self).get_param_names()
        param_names += ('e')
        return param_names

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

#@kabuki.hierarchical
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
        param_names = super(self.__class__, self).get_param_names()
        param_names += ('e1','e2', 'e_inter')
        return param_names

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

if __name__ == "__main__":
    import doctest
    doctest.testmod()
