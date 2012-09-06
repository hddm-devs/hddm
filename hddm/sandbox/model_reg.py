import hddm
import kabuki
from hddm.model import HDDM
import pymc as pm
from kabuki import Knode
from kabuki.distributions import scipy_stochastic
import numpy as np
from scipy import stats
from copy import copy, deepcopy

class wfpt_regress_gen(stats.distributions.rv_continuous):

    wiener_params = {'err': 1e-4, 'nT':2, 'nZ':2,
                                  'use_adaptive':1,
                                  'simps_err':1e-3}
    wp = wiener_params
    sampling_method = 'drift'
    dt=1e-4

    def _argcheck(self, *args):
        return True

    def _logp(self, x, v, V, a, z, Z, t, T, reg_outcomes):
        """Log-likelihood for the full DDM using the interpolation method"""
        return hddm.wfpt.wiener_like_multi(x, v, V, a, z, Z, t, T, .001, reg_outcomes)

    def _pdf(self, x, v, V, a, z, Z, t, T, reg_outcomes):
        raise NotImplementedError

    def _rvs(self, v, V, a, z, Z, t, T, reg_outcomes):
        param_dict = {'v':v, 'z':z, 't':t, 'a':a, 'Z':Z, 'V':V, 'T':T}
        sampled_rts = np.empty(self._size)

        for i_sample in xrange(self._size):
            #get current params
            for p in reg_outcomes:
                param_dict[p] = locals()[p][i_sample]
            #sample
            sampled_rts[i_sample] = hddm.generate.gen_rts(param_dict, method=self.sampling_method,
                                                   samples=1, dt=self.dt)
        return sampled_rts

    def random(self, v=1., V=0., a=2, z=.5, Z=.1, t=.3, T=.1, reg_outcomes=None, size=None):
        print "in random"
        self._size = len(locals()[reg_outcomes[0]])
        return self._rvs(v, V, a, z, Z, t, T, reg_outcomes)


wfpt_reg_like = scipy_stochastic(wfpt_regress_gen, name='wfpt_reg',
                                 longname="""Wiener first passage time with regressors likelihood function""",
                                 extradoc="")



################################################################################################



class HDDMRegressor(hddm.model.HDDM):

    def __init__(self, data, regressor=None, **kwargs):
        """Hierarchical Drift Diffusion Model with regressors
        """

        #create self.regressor and self.reg_outcome
        regressor = deepcopy(regressor)
        if type(regressor) == dict:
            regressor = [regressor]

        self.reg_outcomes = [] # holds all the parameters that are going to modeled as outcome
        for reg in regressor:
            if type(reg['args']) == str:
                reg['args'] = [reg['args']]
            if type(reg['covariates']) == str:
                reg['covariates'] = [reg['covariates']]
            self.reg_outcomes.append(reg['outcome'])

        self.regressor = regressor

        #call HDDDM constractor
        super(self.__class__, self).__init__(data, **kwargs)

        #set wfpt_reg
        del self.wfpt
        self.wfpt_reg = deepcopy(wfpt_reg_like)
        self.wfpt_reg.rv.wiener_params = self.wiener_params


    def create_params(self):

        #get params from HDDM
        params = super(self.__class__, self).create_params()

        #remove wfpt and the outcome params
        remove_params = ['wfpt'] + self.reg_outcomes
        params = [x for x in params if x.name not in remove_params]

        #create regressor params
        for i_reg, reg in enumerate(self.regressor):
            for arg in reg['args']:
                reg_var = Knode(pm.Uniform, lower=1e-10, upper=100, value=1)
                reg_g = Knode(pm.Normal, mu=0, tau=10**-2, value=0, step_method=kabuki.steps.kNormalNormal)
                reg_subj = Knode(pm.Normal, value=0.5)
                reg_param = Parameter(arg, group_knode=reg_g, var_knode=reg_var, subj_knode=reg_subj,
                          group_label = 'mu', var_label = 'tau', var_type='std',
                          transform=lambda mu,var:(mu, var**-2))
                params.append(reg_param)

        #wfpt
        wfpt_knode = Knode('')
        wfpt = Parameter('wfpt', is_bottom_node=True, subj_knode=wfpt_knode)
        params.append(wfpt)

        return params

    def get_bottom_node(self, param, params):

        #create regressors
        for reg in self.regressor:
            outcome = reg['outcome']
            func = reg['func']
            cov = reg['covariates']

            #get predictors
            predictors = np.empty((len(param.data), len(cov)))
            for i_c, c in enumerate(cov):
                predictors[:,i_c] = param.data[c]

            #apply function to predictors
            try:
                nodes_args = []
                for arg in reg['args']:
                    nodes_args.append(params[arg])
                params[outcome] = func(nodes_args, predictors)
            except TypeError:
                errmsg = """the function of %s raised an error. make sure that the first argument is a list of
                of arguments, and the second is a numpy array or matrix""" % param.full_name
                raise TypeError(errmsg)

        model = self.wfpt_reg(param.full_name,
                            value=param.data['rt'],
                            v=params['v'],
                            V=self.get_node('V', params),
                            a=params['a'],
                            z=self.get_node('z', params),
                            Z=self.get_node('Z', params),
                            t=params['t'],
                            T=self.get_node('T', params),
                            reg_outcomes=self.reg_outcomes,
                            observed=True)
        return model
