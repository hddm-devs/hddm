import hddm
from hddm.model import HDDM
import pymc as pm
from kabuki import Parameter
from kabuki.distributions import scipy_stochastic
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


class HDDMContUnif(HDDM):
    """Contaminant HDDM Uniform class

    Outliers are modeled using a uniform distribution over responses
    and reaction times.

    :Optional:
        init : bool
            Use EZ to initialize parameters (default: True)

    """
    def __init__(self, *args, **kwargs):
        super(hddm.model.HDDMContUnif, self).__init__(*args, **kwargs)

        self.cont_res = None
        self.t_min = 0
        self.t_max = max(abs(self.data['rt']))
        wp = self.wiener_params
        self.wfpt = hddm.likelihoods.general_WienerCont(err=wp['err'],
                                                        nT=wp['nT'],
                                                        nZ=wp['nZ'],
                                                        use_adaptive=wp['use_adaptive'],
                                                        simps_err=wp['simps_err'])

    def get_params(self):
        params = super(hddm.model.HDDMContUnif, self).get_params()
        params = params[:-1] + \
                 [Parameter('pi', lower=0.01, upper=0.1),
                  Parameter('x', is_bottom_node=True),
                  Parameter('wfpt', is_bottom_node=True)]

        return params

    def get_bottom_node(self, param, params):
        if param.name == 'wfpt':
            return self.wfpt(param.full_name,
                             value=param.data['rt'].flatten(),
                             cont_x=params['x'],
                             v=params['v'],
                             a=params['a'],
                             z=self.get_node('z',params),
                             t=params['t'],
                             Z=self.get_node('Z',params),
                             T=self.get_node('T',params),
                             V=self.get_node('V',params),
                             t_min=self.t_min,
                             t_max=self.t_max,
                             observed=True)

        elif param.name == 'x':
            rts = param.data['rt']
            outlier = np.empty(rts.shape, dtype=np.bool)
            outlier[np.abs(rts) < params['t'].value] = True
            outlier[np.abs(rts) >= params['t'].value] = False
            return pm.Bernoulli(param.full_name, p=params['pi'], size=len(param.data['rt']), plot=False, value=outlier)

        else:
            raise KeyError, "Groupless subj parameter %s not found" % param.name

    def cont_report(self, cont_threshold=0.5, plot=True):
        """Create conaminate report

        :Arguments:
            cont_threshold : float
                the threshold tthat define an outlier (default: 0.5)
            plot : bool
                should the result be plotted (default: True)

        :Returns:
            report : dict
                * cont_idx : index of outlier above threshold
                * probs : Probability RT is an outlier
                * rts : RTs of outliers

        """
        hm = self
        data_dep = hm._get_data_depend()

        self.cont_res = {}
        if self.is_group_model:
            subj_list = self._subjs
        else:
            subj_list = [0]

        conds = hm.params_dict['x'].subj_nodes.keys()


        #loop over subjects
        for subj_idx, subj in enumerate(subj_list):
            n_cont = 0
            rts = np.empty(0)
            probs = np.empty(0)
            cont_idx = np.empty(0, np.int)
            print "#$#$#$# outliers for subject %s #$#$#$#" % subj
            #loop over conds
            for cond in conds:
                print "*********************"
                print "looking at %s" % cond
                nodes = hm.params_dict['x'].subj_nodes[cond]
                if self.is_group_model:
                    node = nodes[subj_idx]
                else:
                    node = nodes
                m = np.mean(node.trace(), 0)

                #look for outliers with high probabilty
                idx = np.where(m > cont_threshold)[0]
                n_cont += len(idx)
                if idx.size > 0:
                    print "found %d probable outliers in %s" % (len(idx), cond)
                    wfpt = list(node.children)[0]
                    data_idx = [x for x in data_dep if x[3]==cond][0][0]['data_idx']
                    for i_cont in range(len(idx)):
                        print "rt: %8.5f prob: %.2f" % (wfpt.value[idx[i_cont]], m[idx[i_cont]])
                    cont_idx = np.r_[cont_idx, data_idx[idx]]
                    rts = np.r_[rts, wfpt.value[idx]]
                    probs = np.r_[probs, m[idx]]

                    #plot outliers
                    if plot:
                        plt.figure()
                        mask = np.ones(len(wfpt.value), dtype=bool)
                        mask[idx] = False
                        plt.plot(wfpt.value[mask], np.zeros(len(mask) - len(idx)), 'b.')
                        plt.plot(wfpt.value[~mask], np.zeros(len(idx)), 'ro')
                        plt.title(wfpt.__name__)
                        plt.xlabel('RTs for lower (negative) and upper (positive) boundary responses.')
                #report the next higest probability outlier
                next_outlier = max(m[m < cont_threshold])
                print "probability of the next most probable outlier: %.2f" % next_outlier

            print "!!!!!**** %d probable outliers were found in the data ****!!!!!" % n_cont
            single_cont_res = {}
            single_cont_res['cont_idx'] = cont_idx
            single_cont_res['rts'] = rts
            single_cont_res['probs'] = probs
            if self.is_group_model:
                self.cont_res[subj] = single_cont_res
            else:
                self.cont_res = single_cont_res

        if plot:
            plt.show()


        return self.cont_res


    def remove_conts(self, cont_threshold=.5):
        """
        Return the data without the contaminants.

        :Arguments:
            cutoff : float
                the probablity that defines an outlier (deafult 0.5)

        :Returns:
            cleaned_data : ndarray

        """
        # Generate cont_report
        if not self.cont_res:
            self.cont_report(cont_threshold = cont_threshold)

        new_data = []
        if self.is_group_model:
            subj_list = self._subjs
        else:
            subj_list = [0]

        # Loop over subjects and append cleaned data
        for subj in subj_list:
            if self.is_group_model:
                cont_res = self.cont_res[subj]
                data_subj = self.data[self.data['subj_idx'] == subj]
            else:
                cont_res = self.cont_res
                data_subj = self.data
            idx = np.ones(len(data_subj), bool)
            idx[cont_res['cont_idx'][cont_res['probs'] >= cont_threshold]] = 0
            new_data.append(data_subj[idx])

        data_all = np.concatenate(new_data)
        data_all['rt'] = np.abs(data_all['rt'])

        return data_all
