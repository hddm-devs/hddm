import hddm
from hddm.model import HDDM
import pymc as pm
from kabuki import Parameter
import numpy as np
import matplotlib.pyplot as plt

class HDDMContaminant(HDDM):
    """
    Contaminant HDDM model
    outleirs are modeled using a uniform distribution over responses and reaction times.
    """
    def __init__(self, *args, **kwargs):
        super(HDDMContaminant, self).__init__(*args, **kwargs)
        self.params = self.params[:-1] + \
                 [Parameter('pi',True, lower=0.01, upper=0.1),
                  Parameter('x', False), 
                  Parameter('wfpt', False)]
                             
        self.t_min = 0
        self.t_max = max(self.data['rt'])
        wp = self.wiener_params
        self.wfpt = hddm.likelihoods.general_WienerCont(err=wp['err'], nT=wp['nT'], 
                                                        nZ=wp['nZ'], use_adaptive=wp['use_adaptive'], 
                                                        simps_err=wp['simps_err'])

    def get_rootless_child(self, param, params):
        if param.name == 'wfpt':
            return self.wfpt(param.full_name,
                             value=param.data['rt'].flatten(),
                             cont_x=params['x'],
                             v = params['v'],
                             a = params['a'],
                             z = self.get_node('z',params),
                             t = params['t'],
                             Z = self.get_node('Z',params),
                             T = self.get_node('T',params),
                             V = self.get_node('V',params),
                             t_min=self.t_min,
                             t_max=self.t_max,
                             observed=True)

        elif param.name == 'x':
            return pm.Bernoulli(param.full_name, p=params['pi'], size=len(param.data['rt']), plot=False)
        else:
            raise KeyError, "Rootless child parameter %s not found" % param.name
        
    def cont_report(self, cont_threshold = 0.5, plot= True):
        """create conaminate report.
        Input:
            hm -  HDDM model
            cont_threshold - the threshold tthat define an outlier (default: 0.5)
            plot - shoudl the result be plotted (default: True)
        """
        hm = self
        data_dep = hm._get_data_depend()
        conds = [str(x[2]) for x in data_dep]
        
        self.cont_res = {}
        if self.is_group_model:
            subj_list = self._subjs
        else:
            subj_list = [0]

        #loop over subjects
        for subj_idx, subj in enumerate(subj_list):
            n_cont = 0
            rts = np.empty(0)
            probs = np.empty(0)
            cont_idx = np.empty(0)
            print "#$#$#$# outliers for subject %s #$#$#$#" % subj
            #loop over conds
            for cond in conds:
                print "*********************"
                print "looking at %s" % cond
                nodes =hm.params_include['x'].child_nodes[cond]
                if self.is_group_model:
                    node = nodes[subj_idx]
                else:
                    node = nodes
                m = np.mean(node.trace(),0)
                
                #look for outliers with high probabilty
                idx = np.where(m > cont_threshold)[0]
                n_cont += len(idx)
                if idx.size > 0:
                    print "found %d probable outliers in %s" % (len(idx), cond)
                    wfpt = list(node.children)[0]
                    data_idx = [x for x in data_dep if str(x[2])==cond][0][0]['data_idx']
                    for i_cont in range(len(idx)):
                        print "rt: %8.5f prob: %.2f" % (wfpt.value[idx[i_cont]], m[idx[i_cont]])
                    cont_idx = np.r_[cont_idx, data_idx[idx]]
                    rts = np.r_[rts, wfpt.value[idx]]
                    probs = np.r_[probs, m[idx]]
                    
                    #plot outliers
                    if plot:
                        plt.figure()
                        mask = np.ones(len(wfpt.value),dtype=bool)
                        mask[idx] = False
                        plt.plot(wfpt.value[mask], np.zeros(len(mask) - len(idx)), 'b.')
                        plt.plot(wfpt.value[~mask], np.zeros(len(idx)), 'ro')
                        plt.title(wfpt.__name__)
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


#    def remove_outliers(self, cutoff=.5):
#        data_dep = self._get_data_depend()
#
#        data_out = []
#        cont = []
#        
#        # Find x param
#        for param in self.params:
#            if param.name == 'x':
#                break
#
#        for i, (data, params_dep, dep_name) in enumerate(data_dep):
#            dep_name = str(dep_name)
#            # Contaminant probability
#            print dep_name
#            for subj_idx, subj in enumerate(self._subjs):
#                data_subj = data[data['subj_idx'] == subj]
#                cont_prob = np.mean(param.child_nodes[dep_name][subj_idx].trace(), axis=0)
#            
#                no_cont = np.where(cont_prob < cutoff)[0]
#                cont.append(np.logical_not(no_cont))
#                data_out.append(data_subj[no_cont])
#
#        data_all = np.concatenate(data_out)
#        data_all['rt'] = np.abs(data_all['rt'])
#        
#        return data_all, np.concatenate(cont)

class HDDMAntisaccade(HDDM):
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
