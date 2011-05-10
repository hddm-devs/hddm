from __future__ import division
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from copy import copy
import matplotlib.pyplot as plt
import numpy.lib.recfunctions as rec
import os.path
import os
from ordereddict import OrderedDict

import hddm
import kabuki

from kabuki.hierarchical import Parameter

class Theta(hddm.model.Base):
    def __init__(self, data, effect_on=('a',), effect_coding=True, on_as_1=False, theta_col=None, **kwargs):
        """Hierarchical Drift Diffusion Model analyses for Cavenagh et al, IP.

        Arguments:
        ==========
        data: structured numpy array containing columns: subj_idx, response, RT, theta, dbs

        Keyword Arguments:
        ==================
        effect_on <list>: theta and dbs effect these DDM parameters.
        depend_on <list>: separate stimulus distributions for these parameters.
        effect_coding <bool>: True: effects coding (default)
                      False: dummy coding

        Example:
        ========
        The following will create and fit a model on the dataset data, theta and dbs affect the threshold. For each stimulus,
        there are separate drift parameter, while there is a separate HighConflict and LowConflict threshold parameter. The effect coding type is dummy.

        model = Theta(data, effect_on=['a'], depend_on=['v', 'a'], effect_coding=False, HL_on=['a'])
        model.mcmc()
        """
        # Fish out keyword arguments that should not get passed on
        # to the parent.
        self.effect_on = effect_on
        self.effect_coding = effect_coding
        self.on_as_1 = on_as_1

        if self.effect_coding:
            if self.on_as_1:
                self.dbs = 'dbs_effect'
            else:
                self.dbs = 'dbs_effect_inv'
        else:
            if self.on_as_1:
                self.dbs = 'dbs'
            else:
                self.dbs = 'dbs_inv'
            
        if theta_col is None:
            self.theta_col = 'theta'
        else:
            self.theta_col = theta_col

        self.effect_id = 0

        super(self.__class__, self).__init__(data, **kwargs)

    def get_params(self):
        params = super(self.__class__, self).get_params()
        params += [Parameter('e_theta', True), 
                   Parameter('e_dbs', True), 
                   Parameter('e_inter', True), 
                   ParameterEffect('e_inst', False)]
        for effect_on in self.effect_on:
            p = Parameter('e_inst', False)
            p.effect_on = effect_on
            param.append(p)

        return param_names

    def get_rootless_child(self, param, tag, data, idx=None):
        """Generate the HDDM."""
        data_pts = len(data)
        
        if param.name.startswith('e_inst'):
            return pm.Deterministic(effect2, param.name+tag, param.name+tag,
                                    parents={'base':params[param.effect_on],
                                             'e1':params['e_theta'],
                                             'e2':params['e_dbs'],
                                             'e_inter':params['e_inter'],
                                             'data_e1':data['theta'],
                                             'data_e2':data[self.dbs]}, trace=True)

        if self.model_type == 'simple':
            model = hddm.likelihoods.WienerSimpleMulti(model_name,
                                                       value=data['rt'],
                                                       v=params['v'],
                                                       a=params['a'],
                                                       z=params['z'],
                                                       t=params['t'],
                                                       multi=params['e_inst'],
                                                       observed=True)
        elif self.model_type == 'full_mc':
            model = hddm.likelihoods.WienerFullMcMultiThresh(model_name,
                                                             value=data['rt'],
                                                             v=params_subj['v'],
                                                             V=params_subj['V'],
                                                             z=params_subj['z'],
                                                             Z=params_subj['Z'],
                                                             t=params_subj['t'],
                                                             T=params_subj['T'],
                                                             a=params_subj['a'],
                                                             observed=True)
            

        return model

def effect2(base, e1, e2, e_inter, data_e1, data_e2):
    """2-regressor effect distribution
    """
    return base + data_e1*e1 + data_e2*e2 + data_e1*data_e2*e_inter

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


@kabuki.hierarchical
class ThetaNoDBS(hddm.model.Base):
    def __init__(self, data, effect_on=('a',), theta_col=None, **kwargs):
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
        single_effect <bool>: True: effects do not depend on stimuli
                              False: only one effect distribution for all stimuli (but separate for dbs, theta and interaction) (default)

        Example:
        ========
        The following will create and fit a model on the dataset data, theta and dbs affect the threshold. For each stimulus,
        there are separate drift parameter, while there is a separate HighConflict and LowConflict threshold parameter. The effect coding type is dummy.

        model = Theta(data, effect_on=['a'], depend_on=['v', 'a'], effect_coding=False, HL_on=['a'])
        model.mcmc()
        """

        # Fish out keyword arguments that should not get passed on
        # to the parent.
        self.effect_on = effect_on
        
        if theta_col is None:
            self.theta_col = 'theta'
        else:
            self.theta_col = theta_col

        self.effect_id = 0
        super(self.__class__, self).__init__(data, **kwargs)
        
    def get_param_names(self):
        param_names = super(self.__class__, self).get_param_names()
        param_names += ('e_theta',)
        return param_names

    param_names = property(get_param_names)
    
    def get_observed(self, model_name, data, params, idx=None):
        """Generate the HDDM."""
        data = copy(data)
        params_subj = {}
        for name, param in params.iteritems():
            params_subj[name] = param[idx]
        self.effect_id += 1

        for effect in self.effect_on:
            # Create actual effect on base values, result is a matrix.
            name = 'e_inst_%s_%i_%i'%(effect,idx,self.effect_id)
            if effect == 't':
                func = effect1_nozero
            else:
                func = effect1
            params_subj[effect] = pm.Deterministic(func, name, name, parents={'base': params_subj[effect],
                                                                              'e1': params_subj['e_theta'],
                                                                              'data': data[self.theta_col]}, trace=True)

        if self.model_type == 'simple':
            model = hddm.likelihoods.WienerSimpleMulti(model_name,
                                                       value=data['rt'],
                                                       v=params_subj['v'],
                                                       a=params_subj['a'],
                                                       z=params_subj['z'],
                                                       t=params_subj['t'],
                                                       multi=self.effect_on,
                                                       observed=True)

        elif self.model_type == 'full_mc':
            model = hddm.likelihoods.WienerFullMcMultiThresh(model_name,
                                                             value=data['rt'],
                                                             v=params_subj['v'],
                                                             V=params_subj['V'],                                                             
                                                             z=params_subj['z'],
                                                             Z=params_subj['Z'],
                                                             t=params_subj['t'],
                                                             T=params_subj['T'],
                                                             a=params_subj['a'],
                                                             observed=True)

        return model

def load_scalp_data(continuous=True, remove_outliers=.4, shift_theta=False):
    import numpy.lib.recfunctions as rec
    subjs = range(1,15)
    if continuous:
        file_prefix = 'data/theta_continuous/m3tst3_auc_'
        dtype = np.dtype([('subj_idx', np.int), ('stim', 'S8'), ('rt', np.float), ('response', np.float), ('prer1', np.float), ('prer2', np.float), ('theta', np.float), ('cue2', np.float), ('dbs', np.int)])

    else:
        file_prefix = 'data/theta/m3tst3_auc_'
        dtype = np.dtype([('subj_idx', np.int), ('stim', 'S8'), ('rt', np.float), ('response', np.float), ('prer1', np.float), ('prer2', np.float), ('theta', 'S8'), ('cue2', np.float), ('dbs', np.int)])

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
    all_data = rec.append_fields(all_data,
                                 data=(all_data['stim'], all_data['dbs'], all_data['dbs'], all_data['dbs']),
                                 names=('conf', 'dbs_effect', 'dbs_effect_inv', 'dbs_inv'),
                                 usemask=False)
    
    # Remove outliers
    if remove_outliers:
        all_data = all_data[all_data['rt']>remove_outliers]
    # Set names for stim, theta and dbs
    all_data['conf'][all_data['conf'] == '1'] = 'HC'
    all_data['conf'][all_data['conf'] == '2'] = 'HC'
    all_data['conf'][all_data['conf'] == '3'] = 'LC'
    all_data['stim'][all_data['stim'] == '1'] = 'WW'
    all_data['stim'][all_data['stim'] == '2'] = 'LL'
    all_data['stim'][all_data['stim'] == '3'] = 'WL'

    #all_data['theta'][all_data['theta'] == '0'] = 'low'
    #all_data['theta'][all_data['theta'] == '1'] = 'high'

    all_data['dbs_effect'][all_data['dbs_effect'] == 0] = -1
    all_data['dbs_effect_inv'] = -all_data['dbs_effect']
    all_data['dbs_inv'] = 1-all_data['dbs']
        
    return all_data

def elderly_data(data, remove_outliers=.4):
    # Remove outliers
    if remove_outliers:
        data = data[data['rt']>remove_outliers]
    # Set names for stim, theta and dbs
    data['conf'][data['stim'] == '1'] = 'HC'
    data['conf'][data['stim'] == '2'] = 'HC'
    data['conf'][data['stim'] == '3'] = 'LC'
    data['stim'][data['stim'] == '1'] = 'WW'
    data['stim'][data['stim'] == '2'] = 'LL'
    data['stim'][data['stim'] == '3'] = 'WL'

    # Hack
    data['conf'][data['stim'] == "'1'"] = 'HC'
    data['conf'][data['stim'] == "'2'"] = 'HC'
    data['conf'][data['stim'] == "'3'"] = 'LC'
    data['stim'][data['stim'] == "'1'"] = 'WW'
    data['stim'][data['stim'] == "'2'"] = 'LL'
    data['stim'][data['stim'] == "'3'"] = 'WL'

    return data

def load_intraop_data(continuous=True):
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

    all_data = rec.append_fields(all_data, names=('conf',),
                                 data=(all_data['stim'],), dtypes=('S8', np.int, np.int), usemask=False)
    # Remove outliers
    all_data = all_data[all_data['rt']>.4]
    # Set names for stim, theta and dbs
    all_data['conf'][all_data['stim'] == '3'] = 'HC'
    all_data['conf'][all_data['stim'] == '4'] = 'LC'
    all_data['conf'][all_data['stim'] == '5'] = 'HC'
    all_data['stim'][all_data['stim'] == '3'] = 'WW'
    all_data['stim'][all_data['stim'] == '4'] = 'WL'
    all_data['stim'][all_data['stim'] == '5'] = 'LL'
    
    return all_data


def worker():
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    proc_name = MPI.Get_processor_name()
    status = MPI.Status()

    print "Worker %i on %s: ready!" % (rank, proc_name)
    # Send ready
    MPI.COMM_WORLD.send([{'rank':rank, 'name':proc_name}], dest=0, tag=10)

    # Start main data loop
    while True:
        # Get some data
        print "Worker %i on %s: waiting for data" % (rank, proc_name)
        recv = MPI.COMM_WORLD.recv(source=0, tag=MPI.ANY_TAG, status=status)
        print "Worker %i on %s: received data, tag: %i" % (rank, proc_name, status.tag)

        if status.tag == 2:
            print "Worker %i on %s: received kill signal" % (rank, proc_name)
            MPI.COMM_WORLD.send([], dest=0, tag=2)
            return

        if status.tag == 10:
            # Run emergent
            #print "Worker %i on %s: Running %s" % (rank, proc_name, recv)
            #recv['debug'] = True
            retry = 0
            while retry < 5:
                try:
                    print "Running %s:\n" % recv[0]
                    result = run_model(recv[0], recv[1])
                    break
                except pm.ZeroProbability:
                    retry +=1
            if retry == 5:
                result = None
                print "Job %s failed" % recv[0]

        print("Worker %i on %s: finished one job" % (rank, proc_name))
        MPI.COMM_WORLD.send((recv[0], result), dest=0, tag=15)

    MPI.COMM_WORLD.send([], dest=0, tag=2)
        
def run_model(name, params, load=False):
    if params.has_key('model_type'):
        model_type = params['model_type']
    else:
        model_type = 'simple'

    data = params.pop('data')

    if name.startswith('PD'):
        m = Theta(data, **params).create()
    else:
        if params.has_key('effect_on'):
            m = ThetaNoDBS(data, **params).create()
        else:
            m = hddm.model.HDDM(data, **params).create()
    
    dbname = os.path.join('/','users', 'wiecki', 'scratch', 'theta', name+'.db')

    if not load:
        try:
            os.remove(dbname)
        except OSError:
            pass
        m.mcmc(samples=30000, burn=25000, dbname=dbname)
        print "*************************************\nModel: %s\n%s" %(name, m.summary())
        return m.summary()
    else:
        print "Loading %s" %name
        m.mcmc_load_from_db(dbname=dbname)
        return m


def create_jobs_pd():
    # Load data
    data_pd = np.recfromcsv('PD_PS.csv')
    data_dbs_off = data_pd[data_pd['dbs'] == 0]
    data_dbs_on = data_pd[data_pd['dbs'] == 1]

    models = OrderedDict()
    # Create PD models
    models['PD_paper_effect_on_as_0'] = {'data': data_pd, 'effect_on':['a'], 'depends_on':{'v':['stim'], 'e_theta':['conf'], 'e_inter':['conf']}}
    models['PD_paper_dummy_on_as_0'] = {'data': data_pd, 'effect_coding':False, 'effect_on':['a'], 'depends_on':{'v':['stim'], 'e_theta':['conf'], 'e_inter':['conf']}}
    models['PD_paper_dummy_on_as_1'] = {'data': data_pd, 'effect_coding':False, 'on_as_1':True, 'effect_on':['a'], 'depends_on':{'v':['stim'], 'e_theta':['conf'], 'e_inter':['conf']}}
    models['PD_paper_effect_on_as_1'] = {'data': data_pd, 'on_as_1':True, 'effect_on':['a'], 'depends_on':{'v':['stim'], 'e_theta':['conf'], 'e_inter':['conf']}}
    models['dbs_off_PD_paper'] = {'data': data_dbs_off, 'effect_on':['a'], 'depends_on':{'v':['stim'], 'e_theta':['conf']}}
    models['dbs_on_PD_paper'] = {'data': data_dbs_on, 'effect_on':['a'], 'depends_on':{'v':['stim'], 'e_theta':['conf']}}

    return models

def create_models_nodbs(data, full=False):
    models = []
    model_types = ['simple']
    if full:
        model_types.append('full_mc')
        
    effects_on = ['a', 't']
    vs_on = ['stim', 'conf']
    e_thetas_on = ['stim', 'conf', ['stim','resp'], ['conf','resp']]
    
    for effect in effects_on:
        for v_on in vs_on:
            for e_theta_on in e_thetas_on:
                models.append({'data': data, 'effect_on':[effect], 'depends_on':{'v':[vs_on], 'e_theta':[e_theta_on]}})
                models.append({'data': data, 'effect_on':[effect], 'depends_on':{'v':[vs_on], 'e_theta':[e_theta_on], 'a':['stim']}})
                models.append({'data': data, 'effect_on':[effect, 'z'], 'depends_on':{'v':[vs_on], 'e_theta':[e_theta_on]}})
                models.append({'data': data, 'effect_on':[effect], 'depends_on':{'v':[vs_on], 'e_theta':[e_theta_on, 'rt_split']}})
                models.append({'data': data, 'effect_on':[effect], 'depends_on':{'v':[vs_on], 'e_theta':[e_theta_on, 'rt_split'], 'a':['rt_split']}})
                models.append({'data': data, 'effect_on':[effect], 'depends_on':{'v':[vs_on], 'e_theta':[e_theta_on, 'rt_split'], 'a':['stim', 'rt_split']}})

                
    models.append({'data': data, 'depends_on':{'v':['stim'], 'a':['theta_split']}})
    models.append({'data': data, 'depends_on':{'v':['conf'], 'a':['theta_split']}})
    
    return models

def load_models(pd=False, full_mc=False):
    if pd:
        jobs = create_jobs_pd()
    elif full_mc:
        jobs = create_jobs_full()
    else:
        jobs = create_jobs()

    models = OrderedDict()
    for name, params in jobs.iteritems():
        models[name] = run_model(name, params, load=True)

    return models

def controller(samples=200, burn=15, reps=5):
    process_list = range(1, MPI.COMM_WORLD.Get_size())
    rank = MPI.COMM_WORLD.Get_rank()
    proc_name = MPI.Get_processor_name()
    status = MPI.Status()

    print "Controller %i on %s: ready!" % (rank, proc_name)

    models = create_jobs_full()
    task_iter = models.iteritems()
    results = {}

    while(True):
        status = MPI.Status()
        recv = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        print "Controller: received tag %i from %s" % (status.tag, status.source)
        if status.tag == 15:
            results[recv[0]] = recv[1]

        if status.tag == 10 or status.tag == 15:
            try:
                name, task = task_iter.next()
                print "Controller: Sending task"
                MPI.COMM_WORLD.send((name, task), dest=status.source, tag=10)
            except StopIteration:
                # Task queue is empty
                print "Controller: Task queue is empty"
                print "Controller: Sending kill signal"
                MPI.COMM_WORLD.send([], dest=status.source, tag=2)

        elif status.tag == 2: # Exit
            process_list.remove(status.source)
            print 'Process %i exited' % status.source
            print 'Processes left: ' + str(process_list)
        else:
            print 'Unkown tag %i with msg %s' % (status.tag, str(data))
            
        if len(process_list) == 0:
            print "No processes left"
            break

    return results
    
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

def add_median_fields(data):
    theta_median = np.empty(data.shape, dtype=[('theta_split','S8')])
    rt_split_cor_inc = np.empty(data.shape, dtype=[('rt_split_cor_inc','S8')])
    rt_split = np.empty(data.shape, dtype=[('rt_split','S8')])

    for subj in np.unique(data['subj_idx']):
        subj_idx = data['subj_idx']==subj

        # Set median
        med = np.median(data[subj_idx]['theta'])
        theta_median[subj_idx & (data['theta'] < med)] = 'low'
        theta_median[subj_idx & (data['theta'] >= med)] = 'high'

        # Set high/low RT
        cor_idx = data['response']==1
        inc_idx = data['response']==0
        med_cor = np.median(data[subj_idx & cor_idx]['rt'])
        med_inc = np.median(data[subj_idx & inc_idx]['rt'])
        med = np.median(data[subj_idx]['rt'])

        rt_split_cor_inc[subj_idx & cor_idx & (data['rt'] < med_cor)] = 'fast'
        rt_split_cor_inc[subj_idx & cor_idx & (data['rt'] >= med_cor)] = 'slow'
        rt_split_cor_inc[subj_idx & inc_idx & (data['rt'] < med_inc)] = 'fast'
        rt_split_cor_inc[subj_idx & inc_idx & (data['rt'] >= med_inc)] = 'slow'
        rt_split[subj_idx & (data['rt'] < med)] = 'fast'
        rt_split[subj_idx & (data['rt'] >= med)] = 'slow'
        
    data = rec.append_fields(data, names=('theta_split','rt_split', 'rt_split_cor_inc'),
                             data=(theta_median,rt_split,rt_split_cor_inc), dtypes=('S8', 'S8', 'S8'), usemask=False)

    return data

def load_csv_jim(*args, **kwargs):
    data = np.recfromtxt(*args, dtype=[('subj_idx', '<i8'), ('stim', 'S8'), ('rt', '<f8'), ('response', '<i8'), ('theta', '<f8'), ('conf', 'S8')], delimiter=',', skip_header=True)

    data['stim'][data['stim'] == '1'] = 'WW'
    data['stim'][data['stim'] == '2'] = 'LL'
    data['stim'][data['stim'] == '3'] = 'WL'

    data['conf'][data['conf'] == '1'] = 'HC'
    data['conf'][data['conf'] == '2'] = 'LC'

    data = add_median_fields(data)

    return data[data['rt'] > .4]
    
def plot_cavanagh(a_low=1.5, a_high=3, v=.5, ter=0.3, tag=None, plot_dual=True, plot_ontop=False, plot_error=True):
    import brownian
    if tag is None:
        tag = ''
    x = np.linspace(-5.,5.,1000)
    
    y_low = hddm.pdf_array(x=x, a=a_low, z=a_low/2., v=v, ter=ter, err=.000001)
    y_high = hddm.pdf_array(x=x, a=a_high, z=a_high/2., v=v, ter=ter, err=.000001)

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

if __name__=='__main__':
    #import sys
    #parse_config_file(sys.argv[1])
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        results = controller()
        for name, model in results.iteritems():
            print "****************************************\n%s:\n%s\n" %(name, model)
    else:
        worker()
        
