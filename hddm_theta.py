from __future__ import division
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from copy import copy
import matplotlib.pyplot as plt
from ddm_likelihoods import *
from hddm import *

try:
    from IPython.Debugger import Tracer; debug_here = Tracer()
except:
    pass

#def get_HDDM_regress_multi(base=HDDM_simple_subjs)
def get_HDDM_regress_multi(base=HDDM_multi):
    class HDDM_regress_multi(base):
        def __init__(self, *args, **kwargs):
            """Hierarchical Drift Diffusion Model analyses for Cavenagh et al, IP.

            Arguments:
            ==========
            data: structured numpy array containing columns: subj_idx, response, RT, theta, dbs
            
            Keyword Arguments:
            ==================
            effect_on <list>: theta and dbs effect these DDM parameters.
            depend_on <list>: separate stimulus distributions for these parameters.
            dbs_global <bool>: True: Only one global effect distribution for dbs.
                               False: One dbs effect distribution for each stimulus.
            effect_coding <bool>: True: effects coding (default)
                          False: dummy coding
            HL_on <list>: Instead of individual stimuli, these paramaters depend on conflict type (WinWin, WinLose, LoseLose).
            theta_col <string>: Name of the column used for the values of theta (default: 'theta')
            single_effect <bool>: True: effects do not depend on stimuli
                                  False: only one effect distribution for all stimuli (but separate for dbs, theta and interaction) (default)

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
                self.effect_on = ['a']

            if kwargs.has_key('dbs_global'):
                self.dbs_global = kwargs['dbs_global']
                del kwargs['dbs_global']
            else:
                self.dbs_global = False

            if kwargs.has_key('depend_on'):
                self.depend_on = kwargs['depend_on']
                del kwargs['depend_on']
            else:
                self.depend_on = ['v']

            if kwargs.has_key('effect_coding'):
                self.effect_coding = kwargs['effect_coding']
                del kwargs['effect_coding']
            else:
                self.effect_coding = True

            if kwargs.has_key('on_as_1'):
                self.on_as_1 = kwargs['on_as_1']
                del kwargs['on_as_1']
            else:
                self.on_as_1 = False

            if kwargs.has_key('HL_on'):
                self.HL_on = kwargs['HL_on']
                del kwargs['HL_on']
            else:
                self.HL_on = ()

            if kwargs.has_key('theta_col'):
                self.theta_col = kwargs['theta_col']
                del kwargs['theta_col']
            else:
                self.theta_col = 'theta'

            if kwargs.has_key('single_effect'):
                self.single_effect = kwargs['single_effect']
                del kwargs['single_effect']
            else:
                self.single_effect = False
            
            # Call parent's init.
            kwargs['model_type'] = 'simple'
            super(HDDM_regress_multi, self).__init__(*args, **kwargs)

            if not self.single_effect:
                self.stims = np.unique(self.data['stim'])
            else:
                self.stims = ('',)
            self.subjs = np.unique(self.data['subj_idx'])

            self.stims_HL = ('HC', 'LC')

            if 'dbs' not in self.data.dtype.names:
                self.ignore_dbs = True
            else:
                if np.unique(self.data['dbs']).shape[0] != 1:
                    self.ignore_dbs = False
                    self.dbs = np.unique(self.data['dbs'])
                else:
                    self.ignore_dbs = True

        def _set_group_params(self):
            """Set effect and group level parameters."""
            super(HDDM_regress_multi, self)._set_group_params()

            # Set effect distributions
            for effect in self.effect_on:
                if self.single_effect: # Effect should not depend on stimuli.
                    stims = ('',)
                elif effect in self.HL_on: # Effect should depend on low and high conflict.
                    stims = self.stims_HL
                else:
                    stims = self.stims # Effect should depend on individual stimuli.
                for stim in stims:
                    self.group_params['e_theta_%s_%s'%(effect,stim)] = self._get_group_param('e', tag='theta_%s_%s'%(effect, stim))
                    if not self.ignore_dbs:
                        self.group_params['e_inter_%s_%s'%(effect,stim)] = self._get_group_param('e', tag='inter_%s_%s'%(effect, stim))
                        if not self.dbs_global:
                            self.group_params['e_dbs_%s_%s'%(effect,stim)] = self._get_group_param('e', tag='dbs_%s_%s'%(effect, stim))
                        else:
                            self.group_params['e_dbs_%s'%(effect)] = self._get_group_param('e', tag='dbs_%s_%s'%(effect, stim))


            # Set parameter distribution classes depending on data
            for depend in self.depend_on:
                if self.single_effect: # Effect should not depend on stimuli.
                    stims = ('',)
                elif depend in self.HL_on:
                    stims = self.stims_HL # param depends on high/low conflict stimuli
                else:
                    stims = self.stims # param depends on individual stimuli
                for stim in stims:
                    self.group_params['%s_%s'%(depend, stim)] = self._get_group_param(depend, tag=stim) #pm.Uniform('%s_%s'%(depend, stim), lower=-2, upper=2)

            self.param_names = self.group_params.keys()

            return self

        def _set_subj_params(self):
            """Set subject distributions that depend on group distributions."""
            super(HDDM_regress_multi, self)._set_subj_params()

            for effect in self.effect_on:
                if self.single_effect: # If effects should not depend on stimuli
                    stims = ('',)
                elif effect in self.HL_on: # If effects should depend on low and high conflict
                    stims = self.stims_HL
                else: # if effects should depend on individual stimuli
                    stims = self.stims

                for i in range(self.num_subjs):
                    param_names = []
                    for stim in stims:
                        param_names.append('e_theta_%s_%s'%(effect,stim))
                        if not self.ignore_dbs: # Create dbs effect distributions?
                            param_names.append('e_inter_%s_%s'%(effect,stim))
                            if not self.dbs_global: # Should dbs effect distributions depend on stimuli?
                                param_names.append('e_dbs_%s_%s'%(effect,stim))
                            else:
                                param_names.append('e_dbs_%s'%(effect))

                    for param_name in param_names:
                        self.subj_params[param_name][i] = self._get_subj_param(param_name, self.group_params[param_name],
                                                                               self.group_params_tau[param_name], subj_idx=i)

            
            for depend in self.depend_on:
                if self.single_effect: # Effect should not depend on stimuli.
                    stims = ('',)
                elif depend in self.HL_on:
                    stims = self.stims_HL
                else:
                    stims = self.stims

                for stim in stims:
                    param_name = '%s_%s'%(depend,stim)
                    # Reinit parameters as this can conflict with _get_subj_param().
                    self.subj_params[param_name] = np.empty(self.num_subjs, dtype=object)

                    for i in range(self.num_subjs):
                        self.subj_params[param_name][i] = self._get_subj_param(param_name, self.group_params[param_name],
                                                                                self.group_params_tau[param_name], subj_idx=i)

        def _set_model(self):
            """Generate the HDDM."""
            idx_trl = 0
            params_all = []

            # Initialize arrays
            ddm_subjs = np.empty(self.num_subjs, dtype=object)
            effect_inst = np.empty(self.num_subjs, dtype=object)
            theta_vals = np.empty(self.num_subjs, dtype=object)
            dbs_vals = np.empty(self.num_subjs, dtype=object)
            
            for i in range(self.num_subjs):
                data_subj = self.data[self.data['subj_idx'] == i]
                ddm_subjs[i] = np.empty(len(self.stims), dtype=object)
                effect_inst[i] = np.empty(len(self.stims), dtype=object)
                theta_vals[i] = np.empty(len(self.stims), dtype=object)
                dbs_vals[i] = np.empty(len(self.stims), dtype=object)

                for j,stim in enumerate(self.stims):
                    data_stim = data_subj[data_subj['stim'] == stim]

                    # Determine if stimulus is high or low conflict
                    if (stim == 'WW') | (stim == 'LL'):
                        stim_HL = 'HC'
                    else:
                        stim_HL = 'LC'

                    # Construct Theta values
                    theta = data_stim[self.theta_col]
                    theta_vals[i][j] = np.empty(theta.shape, dtype=float)                    
                    trials = data_stim.shape[0]
                    effect_inst[i][j] = {}

                    if self.effect_coding:
                        if theta.dtype != np.float:
                            theta_vals[i][j][theta == 'low'] = -1.
                            theta_vals[i][j][theta == 'high'] = 1.
                        else:
                            theta_vals[i][j] = theta
                            
                    else:
                        if theta.dtype != np.float:
                            theta_vals[i][j][theta == 'low'] = 0.
                            theta_vals[i][j][theta == 'high'] = 1.
                        else:
                            theta_vals[i][j] = theta

                    if not self.ignore_dbs:
                        # Construct DBS values
                        dbs = data_stim['dbs']
                        dbs_vals[i][j] = np.empty(dbs.shape, dtype=float)
                        if self.effect_coding:
                            if self.on_as_1:
                                dbs_vals[i][j][dbs == 'on'] = 1.
                                dbs_vals[i][j][dbs == 'off'] = -1.
                            else:
                                dbs_vals[i][j][dbs == 'on'] = -1.
                                dbs_vals[i][j][dbs == 'off'] = 1.
                                
                        else:
                            if self.on_as_1:
                                dbs_vals[i][j][dbs == 'on'] = 1.
                                dbs_vals[i][j][dbs == 'off'] = 0.
                            else:
                                dbs_vals[i][j][dbs == 'on'] = 0.
                                dbs_vals[i][j][dbs == 'off'] = 1.                                

                    # Create deterministic distributions that
                    # calculate the parameter values affected by
                    # theta/dbs and the effect sizes
                    for effect in self.effect_on:
                        if self.single_effect:
                            effect_stim = ''
                        elif effect in self.HL_on:
                            effect_stim = stim_HL
                        else:
                            effect_stim = stim
                        if effect in self.depend_on:
                            base_effect = self.subj_params['%s_%s'%(effect, effect_stim)][i]
                        else:
                            base_effect = self.subj_params[effect][i]

                        if not self.ignore_dbs:
                            if not self.dbs_global:
                                e_dbs = self.subj_params['e_dbs_%s_%s'%(effect,effect_stim)][i]
                            else:
                                e_dbs = self.subj_params['e_dbs_%s'%(effect)][i]

                            effect_inst[i][j][effect] = pm.Lambda('e_inst_%s_%s_%i'%(effect,effect_stim,i),
                                                              lambda e_base = base_effect,
                                                              theta_val = theta_vals[i][j],
                                                              dbs_val = dbs_vals[i][j],
                                                              e_dbs=e_dbs,
                                                              e_theta=self.subj_params['e_theta_%s_%s'%(effect,effect_stim)][i],
                                                              e_inter=self.subj_params['e_inter_%s_%s'%(effect,effect_stim)][i]:
                                                              e_base + theta_val*e_theta + dbs_val*e_dbs  + theta_val*dbs_val*e_inter,
                                                              trace=False)
                        else:
                            effect_inst[i][j][effect] = pm.Lambda('e_inst_%s_%s_%i'%(effect,effect_stim,i),
                                                          lambda e_base = base_effect,
                                                          theta_val = theta_vals[i][j],
                                                          e_theta=self.subj_params['e_theta_%s_%s'%(effect,effect_stim)][i]:
                                                          e_base + theta_val*e_theta,
                                                          trace=False)


                    # Construct parameter set for this ddm.
                    params = {}
                    for name, param in self.subj_params.iteritems():
                        params[name] = param[i]

                    # Overwrite depend on distribution
                    for depend in self.depend_on:
                        if depend in self.HL_on:
                            depend_stim = stim_HL
                        else:
                            depend_stim = stim
                        params[depend] = self.subj_params['%s_%s'%(depend, depend_stim)][i]

                    if 'a' in self.depend_on:
                        if 'a' in self.HL_on:
                            depend_stim = stim_HL
                        else:
                            depend_stim = stim
                        #params['z'] = None #self.subj_params['a_%s'%(depend_stim)][i]/2.

                    # Overwrite effect distribution
                    for effect in self.effect_on:
                        params[effect] = effect_inst[i][j][effect]

                    if 'a' in self.effect_on and self.no_bias:
                        #params['z'] = None #effect_inst[i][j][effect]/2.
                        multi = list(self.effect_on)+['z']
                    else:
                        #params['z'] = params['a']/2.
                        multi = self.effect_on

                    # Create the wiener likelihood distribution.
                    
                    ddm_subjs[i][j] = self._get_ddm("ddm_%i_%i"%(i,j), data_stim['rt'], params, multi)
                    

                    idx_trl+=1
                    params_all.append(params)

            # Combine all parameters
            self.model = self.group_params.values() + self.group_params_tau.values() + self.subj_params.values() + [ddm_subjs] + params_all

            return self

        def _get_ddm(self, name, data, params, multi):
            return WienerSimpleMulti(name,
                                     value=data,
                                     v=params['v'],
                                     ter=params['t'],
                                     a=params['a'],
                                     z=params['z'],
                                     multi=multi,
                                     observed=True, trace=False)


    return HDDM_regress_multi

class HDDM_regress_multi_lba(get_HDDM_regress_multi(base=HDDM_multi_lba)):
    def _get_ddm(self, name, data, params, multi):
        resps = np.double(data > 0)
        data = np.double(np.abs(data))
        return LBA_multi(name,
                         value=data,
                         resps=resps,
                         a=params['a'],
                         z=params['z'],
                         ter=params['t'],
                         v=np.array([params['v%i'%i] for i in self.resps]).flatten(),
                         sv=params['V'],
                         multi=multi,
                         observed=True)         
                        
HDDM_regress_multi = get_HDDM_regress_multi()
class HDDM_regress_multi_avg(get_HDDM_regress_multi(base=HDDM_full_avg_subj)):
    def _get_ddm(self, name, data, params, multi):
        return WienerFullAvgMulti(name,
                                  value=data,
                                  v=params['v'],
                                  ter=params['ter'], 
                                  a=params['a'],
                                  z=params['z'],
                                  ster=params['ster'],
                                  sv=params['sv'],
                                  sz=params['sz'],
                                  multi=multi,
                                  observed=True, trace=False)

    
def load_scalp_data(high_conf=False, continuous=False, remove_outliers=.4, shift_theta=False):
    subjs = range(1,15)
    if continuous:
        file_prefix = 'data/theta_continuous/m3tst3_auc_'
        dtype = np.dtype([('subj_idx', np.int), ('stim', 'S8'), ('rt', np.float), ('response', np.float), ('prer1', np.float), ('prer2', np.float), ('theta', np.float), ('cue2', np.float), ('dbs', 'S8')])

    else:
        file_prefix = 'data/theta/m3tst3_auc_'
        dtype = np.dtype([('subj_idx', np.int), ('stim', 'S8'), ('rt', np.float), ('response', np.float), ('prer1', np.float), ('prer2', np.float), ('theta', 'S8'), ('cue2', np.float), ('dbs', 'S8')])

    all_data = []
    for subj_idx, subj in enumerate(subjs):
        for onoff in ['on', 'off']:
            data = np.recfromtxt(file_prefix+str(subj)+onoff+'.txt', dtype=dtype)
            data['subj_idx'] = subj_idx

            if continuous: # Normalize theta
                mu = np.mean(data['theta'])
                sigma = np.std(data['theta'])
                data['theta'] = (data['theta']-mu)/sigma
                if shift_theta:
                    # Shift the theta values one back
                    tmp = copy(data['theta'][:-1])
                    data['theta'][1:] = tmp
                    data = data[1:]

            all_data.append(data)

    all_data = np.concatenate(all_data)

    # Remove outliers
    if remove_outliers:
        all_data = all_data[all_data['rt']>remove_outliers]
    # Set names for stim, theta and dbs
    if high_conf:
        all_data['stim'][all_data['stim'] == '1'] = 'HC'
        all_data['stim'][all_data['stim'] == '2'] = 'HC'
        all_data['stim'][all_data['stim'] == '3'] = 'LC'
    else:
        all_data['stim'][all_data['stim'] == '1'] = 'WW'
        all_data['stim'][all_data['stim'] == '2'] = 'LL'
        all_data['stim'][all_data['stim'] == '3'] = 'WL'
    if not continuous:
        all_data['theta'][all_data['theta'] == '0'] = 'low'
        all_data['theta'][all_data['theta'] == '1'] = 'high'

    all_data['dbs'][all_data['dbs'] == '0'] = 'off'
    all_data['dbs'][all_data['dbs'] == '1'] = 'on'

        
    return all_data

def load_intraop_data(high_conf=False, continuous=True):
    subjs = range(1,9)
    file_prefix = 'data/stn/inoptst3_'
    all_data = []

    dtype = np.dtype([('subj_idx', np.int), ('stim', 'S8'), ('rt', np.float), ('response', np.float), ('vent', np.float), ('mid', np.float), ('dors', np.float)])

    for subj_idx,subj in enumerate(subjs):
        data = np.recfromtxt(file_prefix+str(subj)+'.txt', dtype=dtype)
        data['subj_idx'] = subj_idx
        valid_rows = data['response'] != 10000
        data = data[valid_rows]
        if continuous: # Normalize theta
            for col in ('vent', 'mid', 'dors'):
                mu = np.mean(data[col])
                sigma = np.std(data[col])
                data[col] = (data[col]-mu)/sigma
        all_data.append(data)

    all_data = np.concatenate(all_data)
    
    # Remove outliers
    all_data = all_data[all_data['rt']>.4]
    # Set names for stim, theta and dbs
    if high_conf:
        all_data['stim'][all_data['stim'] == '3'] = 'HC'
        all_data['stim'][all_data['stim'] == '4'] = 'LC'
        all_data['stim'][all_data['stim'] == '5'] = 'HC'
    else:
        all_data['stim'][all_data['stim'] == '3'] = 'WW'
        all_data['stim'][all_data['stim'] == '4'] = 'WL'
        all_data['stim'][all_data['stim'] == '5'] = 'LL'
    
    return all_data


def analyze_final(model, range=(-.2,.2), bins=100, e_idx=0, lower=0, upper=.3, plot_prior=True, reverse=True):
    effect_on = model.effect_on[e_idx]
    if effect_on == 'a':
        effect_name = 'threshold'
    else:
        effect_name = 'ter'
        
    if len(model.HL_on) != 0:
        stims = model.stims_HL
    else:
        stims = model.stims

    if not model.dbs_global:
        effect_types = ('theta', 'dbs', 'inter')
    else:
        effect_types = ('theta', 'inter')
    # HACK    
    effect_types = ('theta', 'dbs', 'inter')
    sav_dick = {}
    x = np.linspace(range[0], range[1], bins)
    prior = uniform(x, lower, upper)
    # Get traces
    for effect_type in effect_types:
        sav_dick[effect_type] = {}
        plt.figure()
        if effect_type == 'inter':
            title = 'Interaction effect between dbs and theta on %s'%effect_name
        else:
            title = 'Effect of %s on %s'%(effect_type, effect_name)
        for i,stim in enumerate(stims):
            if effect_type == 'dbs':
                post_trace = model.group_params['e_%s_%s'%(effect_type,effect_on)].trace()
            else:
                post_trace = model.group_params['e_%s_%s_%s'%(effect_type,effect_on,stim)].trace()
            sd = savage_dickey(post_trace, range=range, bins=bins, plot=True, title=title, prior_y=prior, plot_prior=((i==0) and plot_prior), label='%s'%(stim))
            if reverse:
                sd = 1./sd
            sav_dick[effect_type][stim] = sd
            plt.xlim(range)
        plt.legend(loc=0)
        plt.savefig('plots/final_%s_%s.png'%(effect_on, effect_type))
        plt.savefig('plots/final_%s_%s.pdf'%(effect_on, effect_type))
    return sav_dick

def plot_theta_model(m, conf=None):
    if conf is None:
        conf='HC'
    effect_on = m.effect_on[0]
    a = m.params_est[effect_on]
    e_theta = m.params_est['e_theta_%s_%s'%(effect_on, conf)]
    e_dbs = m.params_est['e_theta_%s_%s'%(effect_on, conf)]
    e_inter = m.params_est['e_theta_%s_%s'%(effect_on, conf)]
    if m.effect_coding:
        if m.on_as_1:
            off = -1
            on = 1
        else:
            off = 1
            on = -1
    else:
        if m.on_as_1:
            off = 0
            on = 1
        else:
            off = 1
            on = 0
            
    plot_theta(a, e_theta, e_dbs, e_inter, off, on)
    
def plot_theta(a, e_theta, e_dbs, e_inter, off=1, on=0):
    x = np.linspace(-2, 2, 200)
    plt.plot(x, (a + x*e_theta + on*e_dbs + x*on*e_inter), label='on')
    plt.plot(x, (a + x*e_theta + off*e_dbs + x*off*e_inter), label='off')
    plt.legend()

def plot_cavanagh_model(m):
    # Find average min and max values of theta
    theta_min = np.mean([np.min(m.data['theta'][m.data['subj_idx'] == i]) for i in range(m.num_subjs)])
    theta_max = np.mean([np.max(m.data['theta'][m.data['subj_idx'] == i]) for i in range(m.num_subjs)])
    
    a = m.params_est['a']
    effect = m.params_est['e_theta_a_HC']

    a_low = a + effect*theta_min
    a_high = a + effect*theta_max

    plot_cavanagh(a_low=a_low, a_high=a_high, v=np.mean([m.params_est['v_LL'], m.params_est['v_WW']]), ter=m.params_est['ter'], tag='v_high', plot_ontop=True)
    #plot_cavanagh(a_low=a_low, a_high=a_high, v=m.params_est['v_WL'], ter=m.params_est['ter'], tag='v_low')
    
def plot_cavanagh(a_low=1.5, a_high=3, v=.5, ter=0.3, tag=None, plot_dual=True, plot_ontop=False, plot_error=True):
    import brownian
    if tag is None:
        tag = ''
    x = np.linspace(-5.,5.,1000)
    
    y_low = wfpt.pdf_array(x=x, a=a_low, z=a_low/2., v=v, ter=ter, err=.000001)
    y_high = wfpt.pdf_array(x=x, a=a_high, z=a_high/2., v=v, ter=ter, err=.000001)

    y_low_scaled = y_low
    y_high_scaled = y_high
    #y_low_scaled, y_high_scaled = brownian.scale_multi(y_low, y_high)
    # plt.figure()
    # plt.plot(x, y_low, lw=2., label='low threshold')
    # plt.plot(x, y_high, lw=2., label='high threshold')

    # plt.legend(loc=0)
    # plt.xlabel('time (s)')
    # plt.ylabel('likelihood')
    # plt.title('Drift diffusion model reaction time distribution')
    # plt.savefig('rt_dist_mirrored%s.png'%tag)

    xlim_upper = 3.5
    if plot_dual:
        plt.figure()
        plt.subplot(211)
        plt.title('Low threshold')
        plt.plot(x[x>0], y_low_scaled[x>0], 'g', label='correct', lw=2)
        plt.plot(-x[x<0], y_low_scaled[x<0], 'r', label='errors', lw=2)
        plt.legend(loc=0)
        plt.xlim(0, xlim_upper)
        plt.ylim(0, np.max(np.abs(y_low))+.0)
        plt.subplot(212)
        plt.title('High threshold')
        plt.plot(x[x>0], y_high_scaled[x>0], 'g', label='correct', lw=2)
        plt.plot(-x[x<0], y_high_scaled[x<0], 'r', label='errors', lw=2)
        plt.legend(loc=0)
        plt.xlim(0, xlim_upper)
        plt.ylim(0, np.max(np.abs(y_low))+.0)
        plt.savefig('rt_dual%s.png'%tag)
        plt.savefig('rt_dual%s.pdf'%tag)


    if plot_ontop:
        plt.figure()
        plt.title('Drift diffusion model reaction time distribution')
        plt.plot(x[x>0], y_low_scaled[x>0], 'g--', label='low threshold correct', lw=2)
        plt.plot(-x[x<0], y_low_scaled[x<0], 'r--', label='low threshold error', lw=2)
        plt.plot(x[x>0], y_high_scaled[x>0], 'g', label='high threshold correct', lw=2)
        plt.plot(-x[x<0], y_high_scaled[x<0], 'r', label='high threshold error', lw=2)
        plt.legend(loc=0)
        plt.xlabel('time (s)')
        plt.ylabel('likelihood')
        plt.savefig('rt_dist%s.png'%tag)
        plt.savefig('rt_dist%s.pdf'%tag)

    if plot_error:
        plt.figure()
        plt.subplot(211)
        plt.title('Correct')
        plt.plot(x[x>0], y_high_scaled[x>0], 'g', label='high theta/threshold', lw=2)
        plt.plot(x[x>0], y_low_scaled[x>0], 'g--', label='low theta/threshold', lw=2)
        plt.legend(loc=0)
        plt.xlim(0, xlim_upper)
        plt.ylim(0, np.max(np.abs(y_low_scaled))+.0)
        plt.subplot(212)
        plt.title('Error')
        plt.plot(-x[x<0], y_high_scaled[x<0], 'r', label='high theta/threshold', lw=2)
        plt.plot(-x[x<0], y_low_scaled[x<0], 'r--', label='low theta/threshold', lw=2)
        plt.legend(loc=0)
        plt.xlim(0, xlim_upper)
        plt.ylim(0, np.max(np.abs(y_low_scaled))+.0)
        plt.savefig('rt_dual_error%s.png'%tag)
        plt.savefig('rt_dual_error%s.pdf'%tag)
        plt.xlabel('time (secs)')
        plt.ylabel('likelihood')
