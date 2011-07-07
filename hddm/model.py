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
    """HDDM Base Class. Implements the hierarchical Ratcliff
    drift-diffusion model using the Navarro & Fuss likelihood and
    numerical integration over the variability parameters.

    :Arguments:
        data <numpy.recarray>: Input data with a row for each trial.
            Must contain the following columns:
              * 'rt': Reaction time of trial in seconds.
              * 'response': Binary response (e.g. 0->error, 1->correct)
            May contain:
              * 'subj_idx': A unique ID (int) of the subject.
              * Other user-defined columns that can be used in depends_on
                keyword.

    :Keyword arguments:
        include <tuple=()>: Optional variability parameters to include.
            Can be any combination of 'V', 'Z' and 'T'. Passing the string
            'all' will include all three.
            
            :Note: Including 'Z' and/or 'T' will increase run time
            significantly!

        is_group_model <bool>: If True, this results in a hierarchical
            model with separate parameter distributions for each
            subject. The subject parameter distributions are
            themselves distributed according to a group parameter
            distribution.

            If not provided, this parameter is set to True if data
            provides a column 'subj_idx' and False otherwise.
        
        depends_on <dict>: Specifies which parameter depends on data
            of a column in data. For each unique element in that
            column, a separate set of parameter distributions will be
            created and applied. Multiple columns can be specified in
            a sequential container (e.g. list)

            :Example: 

            depends_on={'param1':['column1']}
    
            Suppose column1 has the elements 'element1' and
            'element2', then parameters 'param1('element1',)' and
            'param1('element2',)' will be created and the
            corresponding parameter distribution and data will be
            provided to the user-specified method get_liklihood().

        bias <bool=False>: Whether to allow a bias to be
            estimated. This is normally used when the responses represent
            left/right and subjects could develop a bias towards
            responding right. This is normally never done, however, when
            the 'response' column codes correct/error.

        trace_subjs <bool=True>: Save trace for subjs (needed for many
             statistics so probably a good idea.)

        plot_tau <bool=False>: Plot group variability parameters
             (i.e. variance of Normal distribution.)

        wiener_params <dict>: Parameters for wfpt evaluation and
             numerical integration.
             
             :Parameters: 
                 err: Error bound for wfpt (default 1e-4)
                 nT: Maximum depth for numerical integration 
                     for T (default 2)
                 nZ: Maximum depth for numerical integration 
                     for Z (default 2)
                 use_adaptive: Whether to use adaptive numerical 
                     integration (default True)
                 simps_err: Error bound for Simpson integration
                     (default 1e-3)

    """
    
    def __init__(self, data, trace_subjs=True, bias=False,
                 include=(), wiener_params = None, **kwargs):
        
        # Flip sign for lower boundary RTs
        self.data = hddm.utils.flip_errors(data)

        include = set(include)
        
        if include is not None:
            if include == 'all':
                [include.add(param) for param in ('T','V','Z')]
            else:
                [include.add(param) for param in include]

        if bias:
            include.add('z')

        if wiener_params is None:
            self.wiener_params = {'err': 1e-4, 'nT':2, 'nZ':2,
                                  'use_adaptive':1,
                                  'simps_err':1e-3}
            self.wfpt = hddm.likelihoods.WienerFullIntrp
        else:
            self.wiener_params = wiener_params
            wp = self.wiener_params
            self.wfpt = hddm.likelihoods.general_WienerFullIntrp_variable(err=wp['err'], nT=wp['nT'], nZ=wp['nZ'], use_adaptive=wp['use_adaptive'], simps_err=wp['simps_err'])

        self.params = self.get_params()

        super(hddm.model.Base, self).__init__(data, include=include, **kwargs)

    def get_params(self):
        """Returns list of model parameters.
        """
        params = [Parameter('a', True, lower=.5, upper=4.5),
                  Parameter('v', True, lower=-10., upper=10.), 
                  Parameter('t', True, lower=.1, upper=2., init=.1),
                  Parameter('z', True, lower=0., upper=1., init=.5, 
                            default=.5, optional=True),
                  Parameter('V', True, lower=0., upper=6., default=0, 
                            optional=True),
                  Parameter('Z', True, lower=0., upper=1., init=.1, 
                            default=0, optional=True),
                  Parameter('T', True, lower=0., upper=2., init=.1, 
                            default=0, optional=True),
                  Parameter('wfpt', False)]
        
        return params
    
    def get_root_node(self, param):
        """Create and return a uniform prior distribution for root
        parameter [param].

        This is used for the group distributions.

        """
        return pm.Uniform(param.full_name,
                          lower=param.lower,
                          upper=param.upper,
                          value=param.init,
                          verbose=param.verbose)

    def get_tau_node(self, param):
        """Create and return a Uniform prior distribution for the
        variability parameter [param].
        
        Note, that we chose a Uniform distribution rather than the
        more common Gamma (see Gelman 2006: "Prior distributions for
        variance parameters in hierarchical models").

        This is used for the variability fo the group distribution.

        """
        return pm.Uniform(param.full_name, lower=0., upper=1., value=.1, plot=self.plot_tau)

    def get_child_node(self, param):
        """Create and return a Normal (in case of an effect or
        drift-parameter) or Truncated Normal (otherwise) distribution
        for [param] centered around param.root with standard deviation
        param.tau and initialization value param.init.

        This is used for the individual subject distributions.

        """
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
        """Create and return the wiener likelihood distribution
        supplied in [param]. 

        [params] is a dictionary of all parameters on which the data
        depends on (i.e. condition and subject).

        """
        if param.name == 'wfpt':
            return self.wfpt(param.full_name,
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
                       Parameter('wfpt', False),
                       Parameter('z',True, lower=0., upper=1., init=.5, default=.5, optional=True)]
            
        self.t_min = 0
        self.t_max = max(self.data['rt'])

    def get_rootless_child(self, param, params):
        if param.name == 'wfpt':
            return hddm.likelihoods.WienerSimpleContaminant(param.full_name,
                                                            value=param.data['rt'],
                                                            cont_x=params['x'],
                                                            gamma=params['gamma'],
                                                            v=params['v'],
                                                            t=params['t'],
                                                            a=params['a'],
                                                            z=self.get_node('z', params),
                                                            t_min=self.t_min,
                                                            t_max=self.t_max,
                                                            observed=True)
        elif param.name == 'x':
            return pm.Bernoulli(param.full_name, p=params['pi'], size=len(param.data['rt']), plot=False)
        elif param.name == 'dummy_gamma':
            return pm.Bernoulli(param.full_name, p=params['gamma'], value=[True,False], observed=True)
        elif param.name == 'dummy_pi':
            return pm.Bernoulli(param.full_name, p=params['pi'], value=[True], observed=True)
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
    def __init__(self, data, init=True, **kwargs):
        super(self.__class__, self).__init__(data, **kwargs)
        
        if 'instruct' not in self.data.dtype.names:
            raise AttributeError, 'data has to contain a field name instruct.'

        self.params = [Parameter('v',True, lower=-4, upper=0.),
                       Parameter('v_switch', True, lower=0, upper=4.),
                       Parameter('a', True, lower=1, upper=4.5),
                       Parameter('t', True, lower=0., upper=.5, init=0.1),
                       Parameter('t_switch', True, lower=0.0, upper=1.0, init=0.3),
                       Parameter('T', True, lower=0, upper=.5, init=.1, default=0, optional=True),
                       Parameter('V_switch', True, lower=0, upper=2., default=0, optional=True),
                       Parameter('wfpt', False)]

    def get_rootless_child(self, param, params):
        if param.name == 'wfpt':
            return hddm.likelihoods.WienerAntisaccade(param.full_name,
                                                      value=param.data['rt'],
                                                      instruct=param.data['instruct'],
                                                      v=params['v'],
                                                      v_switch=params['v_switch'],
                                                      V_switch=self.get_node('V_switch',params),
                                                      a=params['a'],
                                                      z=.5,
                                                      t=params['t'],
                                                      t_switch=params['t_switch'],
                                                      T=self.get_node('T',params),
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
