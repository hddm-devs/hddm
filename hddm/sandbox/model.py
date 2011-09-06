import hddm
from hddm.model import HDDM
import pymc as pm
from kabuki import Parameter
import numpy as np
import matplotlib.pyplot as plt

try:
    import wfpt_switch
except:
    pass

def wiener_like_antisaccade(value, instruct, v, v_switch, V_switch, a, z, t, t_switch, T, err=1e-4):
    """Log-likelihood for the simple DDM switch model"""
    logp = wfpt_switch.wiener_like_antisaccade_precomp(value, instruct, v, v_switch, V_switch, a, z, t, t_switch, T, err)
    return logp

WienerAntisaccade = pm.stochastic_from_dist(name="Wiener Simple Diffusion Process",
                                            logp=wiener_like_antisaccade,
                                            dtype=np.float,
                                            mv=False)

class HDDMSwitch(HDDM):
    def __init__(self, data, init=True, **kwargs):
        super(self.__class__, self).__init__(data, **kwargs)

        if 'instruct' not in self.data.dtype.names:
            raise AttributeError, 'data has to contain a field name instruct.'

        self.params = [Parameter('v', lower=-4, upper=0.),
                       Parameter('v_switch', lower=0, upper=4.),
                       Parameter('a', lower=1, upper=4.5),
                       Parameter('t', lower=0., upper=.5, init=0.1),
                       Parameter('t_switch', lower=0.0, upper=1.0, init=0.3),
                       Parameter('T', lower=0, upper=.5, init=.1, default=0, optional=True),
                       Parameter('V_switch', lower=0, upper=2., default=0, optional=True),
                       Parameter('wfpt', is_bottom_node=True)]

    def get_bottom_node(self, param, params):
        if param.name == 'wfpt':
            return WienerAntisaccade(param.full_name,
                                     value=param.data['rt'],
                                     instruct=np.array(param.data['instruct'], dtype=np.int32),
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

class HDDMRegressor(HDDM):
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

        model = HDDMRegressor(data, effect_on=['a'], depend_on=['v', 'a'], effect_coding=False, HL_on=['a'])
        model.mcmc()
        """

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
                params.append(Parameter('e_%s_%s'%(col_names, effect_on), lower=-3., upper=3., init=0, create_subj_nodes=not self.use_root_for_effects))
                params.append(Parameter('e_inst_%s_%s'%(col_names, effect_on),
                                        False,
                                        vars={'col_name':col_names,
                                              'effect_on':effect_on,
                                              'e':'e_%s_%s'%(col_names, effect_on)}))
            elif len(col_names) == 2:
                for col_name in col_names:
                    params.append(Parameter('e_%s_%s'%(col_name, effect_on), True, lower=-3., upper=3., init=0, create_subj_nodes=not self.use_root_for_effects))
                params.append(Parameter('e_inter_%s_%s_%s'%(col_names[0], col_names[1], effect_on),
                                        True, lower=-3., upper=3., init=0, create_subj_nodes=not self.use_root_for_effects))
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

    def get_bottom_node(self, param, params):
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

        else:
            model = hddm.likelihoods.WienerMulti(param.full_name,
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

def effect1(base, e1, data):
    """Effect distribution.
    """
    return base + e1 * data

def effect1_nozero(base, e1, data):
    """Effect distribution where values <0 will be set to 0.
    """
    value = base + e1 * data
    value[value < 0] = 0.
    value[value > .4] = .4
    return value

def effect2(base, e1, e2, e_inter, data_e1, data_e2):
    """2-regressor effect distribution
    """
    return base + data_e1*e1 + data_e2*e2 + data_e1*data_e2*e_inter
