import hddm
from hddm.model import HDDM
import pymc as pm
from kabuki import Parameter
from kabuki.utils import scipy_stochastic
import numpy as np
from scipy import stats

try:
    import wfpt_switch
except:
    pass

def wiener_like_multi(value, v, V, a, z, Z, t, T, multi=None):
    """Log-likelihood for the simple DDM"""
    return hddm.wfpt.wiener_like_multi(value, v, V, a, z, Z, t, T, .001, multi=multi)

WienerMulti = pm.stochastic_from_dist(name="Wiener Simple Diffusion Process",
                                      logp=wiener_like_multi,
                                      dtype=np.float)

class wfpt_switch_gen(stats.distributions.rv_continuous):
    err = 1e-4
    evals = 100
    def _argcheck(self, *args):
        return True

    def _logp(self, x, v, v_switch, V_switch, a, z, t, t_switch, T):
        """Log-likelihood for the simple DDM switch model"""
        # if t < T/2 or t_switch < T/2 or t<0 or t_switch<0 or T<0 or a<=0 or z<=0 or z>=1 or T>.5:
        #     print "Condition not met"
        logp = wfpt_switch.wiener_like_antisaccade_precomp(x, v, v_switch, V_switch, a, z, t, t_switch, T, self.err, evals=self.evals)
        return logp

    def _pdf(self, x, v, v_switch, V_switch, a, z, t, t_switch, T):
        if np.isscalar(x):
            out = wfpt_switch.pdf_switch(np.array([x]), v, v_switch, V_switch, a, z, t, t_switch, T, 1e-4)
        else:
            out = np.empty_like(x)
            for i in xrange(len(x)):
                out[i] = wfpt_switch.pdf_switch(np.array([x[i]]), v[i], v_switch[i], V_switch[i], a[i], z[i], t[i], t_switch[i], T[i], 1e-4)

        return out

    def _rvs(self, v, v_switch, V_switch, a, z, t, t_switch, T):
        all_rts_generated=False
        while(not all_rts_generated):
            out = hddm.generate.gen_antisaccade_rts({'v':v, 'z':z, 't':t, 'a':a, 'v_switch':v_switch, 'V_switch':V_switch, 't_switch':t_switch, 'Z':0, 'V':0, 'T':T}, samples_anti=self._size, samples_pro=0)[0]
            if (len(out) == self._size):
                all_rts_generated=True
        return hddm.utils.flip_errors(out)['rt']

wfpt_switch_like = scipy_stochastic(wfpt_switch_gen, name='wfpt switch', longname="""Wiener first passage time likelihood function""", extradoc="""Wiener first passage time (WFPT) likelihood function of the Ratcliff Drift Diffusion Model (DDM). Models two choice decision making tasks as a drift process that accumulates evidence across time until it hits one of two boundaries and executes the corresponding response. Implemented using the Navarro & Fuss (2009) method.

Parameters:
***********
v: drift-rate
a: threshold
z: bias [0,1]
t: non-decision time

References:
***********
Fast and accurate calculations for first-passage times in Wiener diffusion models
Navarro & Fuss - Journal of Mathematical Psychology, 2009 - Elsevier
""")

class HDDMSwitch(HDDM):
    def __init__(self, data, init=True, **kwargs):
        super(self.__class__, self).__init__(data, **kwargs)

        if 'instruct' not in self.data.dtype.names:
            raise AttributeError, 'data has to contain a field name instruct.'

    def get_params(self):
        params = [Parameter('vpp', lower=-20, upper=20.),
                  Parameter('vcc', lower=-20, upper=20.),
                  Parameter('a', lower=.5, upper=4.5),
                  Parameter('t', lower=0., upper=.5, init=0.05),
                  Parameter('tcc', lower=0.01, upper=1.0),
                  Parameter('T', lower=0, upper=.5, init=.1, default=0, optional=True),
                  Parameter('Vcc', lower=0, upper=2., default=0, optional=True),
                  Parameter('wfpt_anti', is_bottom_node=True),
                  Parameter('wfpt_pro', is_bottom_node=True)]

        return params

    def get_subj_node(self, param):
        """Create and return a Normal (in case of an effect or
        drift-parameter) or Truncated Normal (otherwise) distribution
        for 'param' centered around param.group with standard deviation
        param.var and initialization value param.init.

        This is used for the individual subject distributions.

        """
        return pm.TruncatedNormal(param.full_name,
                                  a=param.lower,
                                  b=param.upper,
                                  mu=param.group,
                                  tau=param.var**-2,
                                  plot=self.plot_subjs,
                                  trace=self.trace_subjs,
                                  value=param.init)

    def get_var_node(self, param):
        #return pm.Gamma(param.full_name, alpha=.3, beta=.3)
        return pm.Uniform(param.full_name, lower=1e-7, upper=10)

    def get_bottom_node(self, param, params):
        if param.name == 'wfpt_anti':
            data = param.data[param.data['instruct'] == 1]
            if len(data) == 0:
                return None
            return wfpt_switch_like(param.full_name,
                                    value=data['rt'],
                                    v=params['vpp'],
                                    v_switch=params['vcc'],
                                    V_switch=self.get_node('Vcc',params),
                                    a=params['a'],
                                    z=.5,
                                    t=params['t'],
                                    t_switch=params['tcc'],
                                    T=self.get_node('T',params),
                                    observed=True)
        elif param.name == 'wfpt_pro':
            data = param.data[param.data['instruct'] == 0]
            if len(data) == 0:
                return None
            return hddm.likelihoods.wfpt_like(param.full_name,
                                              value=data['rt'],
                                              v=params['vpp'],
                                              V=0,
                                              a=params['a'],
                                              z=.5,
                                              Z=0.,
                                              t=params['t'],
                                              T=0.,
                                              observed=True)

        else:
            raise TypeError, "Parameter named %s not found." % param.name

class HDDMRegressor(HDDM):
    def __init__(self, data, effects_on=None, use_root_for_effects=False, **kwargs):
        """Hierarchical Drift Diffusion Model analyses for Cavenagh et al, IP.

        :Arguments:
            data : numpy.recarray
                structured numpy array containing columns: subj_idx, response, RT, theta, dbs
        :Optional:
            effects_on : dict
                theta and dbs effect these DDM parameters.
            depends_on : dict
                separate stimulus distributions for these parameters.
        :Example:
            >>> import hddm
            >>> data, params = hddm.generate.gen_correlated_rts()
            >>> model = hddm.sandbox.HDDMRegressor(data, effects_on={'a':'cov'})
            >>> model.sample(5000)
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
                params.append(Parameter('e_%s_%s'%(col_names, effect_on),
                                        lower=-3., upper=3., init=0,
                                        create_subj_nodes=not self.use_root_for_effects))
                params.append(Parameter('e_inst_%s_%s'%(col_names, effect_on),
                                        is_bottom_node=True,
                                        vars={'col_name':col_names,
                                              'effect_on':effect_on,
                                              'e':'e_%s_%s'%(col_names, effect_on)}))
            elif len(col_names) == 2:
                for col_name in col_names:
                    params.append(Parameter('e_%s_%s'%(col_name,
                                                       effect_on),
                                            lower=-3.,
                                            upper=3.,
                                            init=0,
                                            create_subj_nodes=not self.use_root_for_effects))
                params.append(Parameter('e_inter_%s_%s_%s'%(col_names[0],
                                                            col_names[1],
                                                            effect_on),
                                        lower=-3.,
                                        upper=3.,
                                        init=0,
                                        create_subj_nodes=not self.use_root_for_effects))
                params.append(Parameter('e_inst_%s_%s_%s'%(col_names[0], col_names[1], effect_on),
                                        is_bottom_node=True,
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
                                        parents={'base': params[param.vars['effect_on']],
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
            model = WienerMulti(param.full_name,
                                value=param.data['rt'],
                                v=params['v'],
                                V=self.get_node('V', params),
                                a=params['a'],
                                z=self.get_node('z', params),
                                Z=self.get_node('Z', params),
                                t=params['t'],
                                T=self.get_node('T', params),
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
