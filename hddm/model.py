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
    
    def __init__(self, data, trace_subjs=True, no_bias=True, 
                 init=False, include=None, wiener_params = None,
                 init_values=None, **kwargs):
        
        # Flip sign for lower boundary RTs
        self.data = hddm.utils.flip_errors(data)

        if include is None:
            self.include = set()
        else:
            if include == 'all':
                self.include = set(['T','V','Z'])
            else:
                self.include = set(include)

        self.no_bias = no_bias
        
        if not self.no_bias:
            self.include.add('z')
            
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
                  Parameter('v',True, lower=-10., upper=10.), 
                  Parameter('t',True, lower=.1, upper=2., init=.1)]

        if not self.no_bias:
            params.append(Parameter('z', True, lower=0., upper=1., init=.5))

        # Include inter-trial variability parameters
        if 'V' in self.include:
            params.append(Parameter('V',True, lower=0., upper=6.))
        if 'Z' in self.include:
            params.append(Parameter('Z',True, lower=0., upper=1., init=.1))
        if 'T' in self.include:
            params.append(Parameter('T',True, lower=0., upper=2., init=.1))

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
        if param.name.startswith('wfpt'):
            if self.wiener_params is not None:
                wp = self.wiener_params
                WienerFullIntrp = hddm.likelihoods.general_WienerFullIntrp_variable(err=wp['err'], nT=wp['nT'], nZ=wp['nZ'],
                                                                                    use_adaptive=wp['use_adaptive'], simps_err=wp['simps_err'])
            else:
                WienerFullIntrp =  hddm.likelihoods.WienerFullIntrp
            
            return WienerFullIntrp(param.name,
                                   value=param.data['rt'].flatten(),
                                   v = params['v'],
                                   a = params['a'],
                                   z = self._get_node('z',params),
                                   t = params['t'],
                                   Z = self._get_node('Z',params),
                                   T = self._get_node('T',params),
                                   V = self._get_node('V',params),
                                   observed=True)

        else:
            raise KeyError, "Rootless parameter named %s not found." % param.name

    def _get_node(self, node_name, params):
        if node_name in self.include:
            return params[node_name]
        elif node_name=='z' and self.no_bias and 'z' not in params:
            return 0.5
        else:
            return 0

class HDDM(Base):
    pass

class HDDMFullExtended(Base):
    def get_params(self):
        self.include = ['V','Z','T']
        params = [Parameter('z_trls', False),
                  Parameter('v_trls', False),
                  Parameter('t_trls', False)]
        params += list(super(self.__class__, self).get_params())

        return params

    def get_rootless_child(self, param, params):
        trials = len(param.data)

        if param.name.startswith('z_trls'):
            return [hddm.likelihoods.CenterUniform(param.full_name+str(i),
                                                   center=params['z'],
                                                   width=params['Z']) for i in range(trials)]

        elif param.name.startswith('v_trls'):
            return [pm.Normal(param.full_name+str(i),
                              mu=params['v'],
                              tau=params['V']**-2) for i in range(trials)]

        elif param.name.startswith('t_trls'):
            return [hddm.likelihoods.CenterUniform(param.full_name+str(i),
                                                   center=params['t'],
                                                   width=params['T']) for i in range(trials)]

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
        super(HDDMContaminant, self).__init__(*args, **kwargs)
        self.params = [Parameter('a',True, lower=.5, upper=4.5),
                       Parameter('v',True, lower=-6., upper=6.), 
                       Parameter('t',True, lower=.1, upper=2., init=.1),
                       Parameter('pi',True, lower=1e-3, upper=0.2),
                       Parameter('gamma',True, lower=1e-4, upper=1-1e-4),
                       Parameter('x', False), 
                       Parameter('dummy_gamma',False),
                       Parameter('dummy_pi',False),
                       Parameter('wfpt', False)]
        if not self.no_bias:
            self.params.append(Parameter('z',True, lower=0., upper=1., init=.5))
            
        self.t_min = 0
        self.t_max = max(self.data['rt'])

    def get_rootless_child(self, param, params):
        if param.name.startswith('wfpt'):
            return hddm.likelihoods.WienerSimpleContaminant(param.full_name,
                                                            value=param.data['rt'],
                                                            cont_x=params['x'],
                                                            gamma=params['gamma'],
                                                            v=params['v'],
                                                            t=params['t'],
                                                            a=params['a'],
                                                            z=self._get_node('z', params),
                                                            t_min=self.t_min,
                                                            t_max=self.t_max,
                                                            observed=True)
        elif param.name.startswith('x'):
            return pm.Bernoulli(param.full_name, p=params['pi'], size=len(param.data['rt']), plot=False)
        elif param.name.startswith('dummy_gamma'):
            return pm.Bernoulli(param.full_name, params['gamma'], value=[True,False], observed=True)
        elif param.name.startswith('dummy_pi'):
            return pm.Bernoulli(param.full_name, params['pi'], value=[True], observed=True)
        else:
            raise KeyError, "Rootless child parameter %s not found" % param.name

    def remove_outliers(self, cutoff=.5):
        data_dep = self._get_data_depend()

        data_out = []
        cont = []
        
        # Find x param
        for param in self.params:
            if param.name == 'x':
                break

        for i, (data, params_dep, dep_name) in enumerate(data_dep):
            dep_name = str(dep_name)
            # Contaminant probability
            print dep_name
            for subj_idx, subj in enumerate(self._subjs):
                data_subj = data[data['subj_idx'] == subj]
                cont_prob = np.mean(param.child_nodes[dep_name][subj_idx].trace(), axis=0)
            
                no_cont = np.where(cont_prob < cutoff)[0]
                cont.append(np.logical_not(no_cont))
                data_out.append(data_subj[no_cont])

        data_all = np.concatenate(data_out)
        data_all['rt'] = np.abs(data_all['rt'])
        
        return data_all, np.concatenate(cont)

class HDDMAntisaccade(Base):
    def __init__(self, data, no_bias=True, init=True, **kwargs):
        super(self.__class__, self).__init__(data, **kwargs)
        
        if 'instruct' not in self.data.dtype.names:
            raise AttributeError, 'data has to contain a field name instruct.'

        self.params = (Parameter('v',True, lower=-6, upper=0., init=-1.),
                       Parameter('v_switch', True, lower=0, upper=6., init=1.),
                       Parameter('a', True, lower=1, upper=4, init=2),
                       Parameter('t', True, lower=0.1, upper=1., init=0.1),
                       Parameter('t_switch', True, lower=0.05, upper=0.7, init=0.3),
                       Parameter('wfpt', False))
            
    def get_rootless_child(self, param, params):
        if param.name == 'wfpt':
            return hddm.likelihoods.WienerAntisaccade(param.full_name,
                                                      value=param.data['rt'],
                                                      instruct=param.data['instruct'],
                                                      v=params['v'],
                                                      v_switch=params['v_switch'],
                                                      a=params['a'],
                                                      z=.5,
                                                      t=params['t'],
                                                      t_switch=params['t_switch'],
                                                      observed=True)
        else:
            raise TypeError, "Parameter named %s not found." % param.name



class HDDMRegressor(Base):
    def __init__(self, data, effects_on=None, use_root_for_effects=False, **kwargs):
        """Hierarchical Drift Diffusion Model analyses for Cavenagh et al, IP.

        Arguments:
        ==========
        data: structured numpy array containing columns: subj_idx, response, RT, theta, dbs

        Keyword Arguments:
        ==================
        effect_on <list>: theta and dbs effect these DDM parameters.
        depend_on <list>: separate stimulus distributions for these parameters.

        Example:
        ========
        The following will create and fit a model on the dataset data, theta and dbs affect the threshold. For each stimulus,
        there are separate drift parameter, while there is a separate HighConflict and LowConflict threshold parameter. The effect coding type is dummy.

        model = Theta(data, effect_on=['a'], depend_on=['v', 'a'], effect_coding=False, HL_on=['a'])
        model.mcmc()
        """
        if effects_on is None:
            self.effects_on = {'a': 'theta'}
        else:
            self.effects_on = effects_on

        self.use_root_for_effects = use_root_for_effects
        
        super(self.__class__, self).__init__(data, **kwargs)
        
    def get_params(self):
        params = []

        # Add rootless nodes for effects
        for effect_on, col_names in self.effects_on.iteritems():
            if type(col_names) is str or (type(col_names) is list and len(col_names) == 1):
                if type(col_names) is list:
                    col_names = col_names[0]
                params.append(Parameter('e_%s_%s'%(col_names, effect_on), True, lower=-3., upper=3., init=0, no_childs=self.use_root_for_effects))
                params.append(Parameter('error_%s_%s'%(col_names, effect_on), True, lower=0., upper=10., init=0, no_childs=self.use_root_for_effects))
                params.append(Parameter('e_inst_%s_%s'%(col_names, effect_on), 
                                        False,
                                        vars={'col_name':col_names,
                                              'effect_on':effect_on,
                                              'e':'e_%s_%s'%(col_names, effect_on)}))
            elif len(col_names) == 2:
                for col_name in col_names:
                    params.append(Parameter('e_%s_%s'%(col_name, effect_on), True, lower=-3., upper=3., init=0, no_childs=self.use_root_for_effects))
                params.append(Parameter('error_%s_%s'%(col_names, effect_on), True, lower=0, upper=10., init=0, no_childs=self.use_root_for_effects))
                params.append(Parameter('e_inter_%s_%s_%s'%(col_names[0], col_names[1], effect_on), 
                                        True, lower=-3., upper=3., init=0, no_childs=self.use_root_for_effects))
                params.append(Parameter('e_inst_%s_%s_%s'%(col_names[0], col_names[1], effect_on), 
                                        False,
                                        vars={'col_name0': col_names[0],
                                              'col_name1': col_names[1],
                                              'effect_on': effect_on,
                                              'e1':'e_%s_%s'%(col_names[0], effect_on),
                                              'e2':'e_%s_%s'%(col_names[1], effect_on),
                                              'inter':'e_inter_%s_%s_%s'%(col_names[0], col_names[1], effect_on)}))
            else:
                raise NotImplementedError, "Only 1 or 2 regressors allowed per variable."

        params += super(self.__class__, self).get_params()

        return params

    def get_rootless_child(self, param, params):
        """Generate the HDDM."""
        if param.name.startswith('e_inst'):
            if not param.vars.has_key('inter'):
                # No interaction
                if param.vars['effect_on'] == 't':
                    func = effect1_nozero
                else:
                    func = effect1

                return pm.Deterministic(func, param.full_name, param.full_name,
                                        parents={'base': self._get_node(param.vars['effect_on'], params),
                                                 'e1': params[param.vars['e']],
                                                 'data': param.data[param.vars['col_name']]}, trace=False, plot=self.plot_subjs)
            else:
                    
                return pm.Deterministic(effect2, param.full_name, param.full_name,
                                        parents={'base': params[param.vars['effect_on']],
                                                 'e1': params[param.vars['e1']],
                                                 'e2': params[param.vars['e2']],
                                                 'e_inter': params[param.vars['inter']],
                                                 'data_e1': param.data[param.vars['col_name0']],
                                                 'data_e2': param.data[param.vars['col_name1']]}, trace=False)

        for effect_on, col_name in self.effects_on.iteritems():
            if type(col_name) is str:
                params[effect_on] = params['e_inst_%s_%s'%(col_name, effect_on)]
            else:
                params[effect_on] = params['e_inst_%s_%s_%s'%(col_name[0], col_name[1], effect_on)]

        if self.model_type == 'simple':
            model = hddm.likelihoods.WienerSimpleMulti(param.full_name,
                                                       value=param.data['rt'],
                                                       v=params['v'],
                                                       a=params['a'],
                                                       z=self._get_node('z',params),
                                                       t=params['t'],
                                                       multi=self.effects_on.keys(),
                                                       observed=True)
        elif self.model_type == 'full':
            model = hddm.likelihoods.WienerFullMulti(param.full_name,
                                                     value=param.data['rt'],
                                                     v=params['v'],
                                                     V=self._get_node('V', params),
                                                     a=params['a'],
                                                     z=self._get_node('z', params),
                                                     Z=self._get_node('Z', params),
                                                     t=params['t'],
                                                     T=self._get_node('T', params),
                                                     multi=self.effects_on.keys(),
                                                     observed=True)
        return model

def effect1(base, e1, error, data):
    """Effect distribution.
    """
    return base + e1 * data + error

def effect1_nozero(base, e1, error, data):
    """Effect distribution where values <0 will be set to 0.
    """
    value = base + e1 * data + error
    value[value < 0] = 0.
    value[value > .4] = .4
    return value

def effect2(base, e1, e2, e_inter, error, data_e1, data_e2):
    """2-regressor effect distribution
    """
    return base + data_e1*e1 + data_e2*e2 + data_e1*data_e2*e_inter + error

if __name__ == "__main__":
    import doctest
    doctest.testmod()
