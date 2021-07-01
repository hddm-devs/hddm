import pandas as pd
import numpy as np
from copy import deepcopy
#import re
import argparse
import sys
import pickle
from data_simulators import ddm 
from data_simulators import ddm_flexbound
from data_simulators import levy_flexbound
from data_simulators import ornstein_uhlenbeck
from data_simulators import full_ddm
from data_simulators import ddm_sdv
#from data_simulators import ddm_flexbound_pre
from data_simulators import race_model
from data_simulators import lca
from data_simulators import ddm_flexbound_seq2
from data_simulators import ddm_flexbound_par2
from data_simulators import ddm_flexbound_mic2

import data_simulators as cds
import hddm.simulators.boundary_functions as bf

import hddm.simulators

# Basic simulators and basic preprocessing

def bin_simulator_output_pointwise(out = [0, 0],
                                   bin_dt = 0.04,
                                   nbins = 0): # ['v', 'a', 'w', 't', 'angle']
    """Turns RT part of simulator output into bin-identifier by trial

    :Arguments:
        out: tuple 
            Output of the 'simulator' function
        bin_dt: float
            If nbins is 0, this determines the desired bin size which in turn automatically 
            determines the resulting number of bins.
        nbins: int
            Number of bins to bin reaction time data into. If supplied as 0, bin_dt instead determines the number of 
            bins automatically.   

    :Returns: 
        2d array. The first columns collects bin-identifiers by trial, the second column lists the corresponding choices.
    """
    
    out_copy = deepcopy(out)

    # Generate bins
    if nbins == 0:
        nbins = int(out[2]['max_t'] / bin_dt)
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, out[2]['max_t'], nbins)
        bins[nbins] = np.inf
    else:  
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, out[2]['max_t'], nbins)
        bins[nbins] = np.inf

    cnt = 0
    counts = np.zeros( (nbins, len(out[2]['possible_choices']) ) )
    
    #data_out = pd.DataFrame(np.zeros(( columns = ['rt', 'response'])
    out_copy_tmp = deepcopy(out_copy)
    for i in range(out_copy[0].shape[0]):
        for j in range(1, bins.shape[0], 1):
            if out_copy[0][i] > bins[j - 1] and out_copy[0][i] < bins[j]:
                out_copy_tmp[0][i] = j - 1
    out_copy = out_copy_tmp
    #np.array(out_copy[0] / (bins[1] - bins[0])).astype(np.int32)
    
    out_copy[1][out_copy[1] == -1] = 0
    
    return np.concatenate([out_copy[0], out_copy[1]], axis = -1).astype(np.int32)

def bin_simulator_output(out = None,
                         bin_dt = 0.04,
                         nbins = 0,
                         max_t = -1,
                         freq_cnt = False): # ['v', 'a', 'w', 't', 'angle']
    """Turns RT part of simulator output into bin-identifier by trial

    :Arguments:
        out : tuple 
            Output of the 'simulator' function
        bin_dt : float
            If nbins is 0, this determines the desired bin size which in turn automatically 
            determines the resulting number of bins.
        nbins : int
            Number of bins to bin reaction time data into. If supplied as 0, bin_dt instead determines the number of 
            bins automatically.   
        max_t : int <default=-1>
            Override the 'max_t' metadata as part of the simulator output. Sometimes useful, but usually
            default will do the job.
        freq_cnt : bool <default=False>
            Decide whether to return proportions (default) or counts in bins.

    :Returns:
        A histogram of counts or proportions.

    """

    if max_t == -1:
        max_t = out[2]['max_t']
    
    # Generate bins
    if nbins == 0:
        nbins = int(max_t / bin_dt)
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, max_t, nbins)
        bins[nbins] = np.inf
    else:  
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, max_t, nbins)
        bins[nbins] = np.inf

    cnt = 0
    counts = np.zeros( (nbins, len(out[2]['possible_choices']) ) )

    for choice in out[2]['possible_choices']:
        counts[:, cnt] = np.histogram(out[0][out[1] == choice], bins = bins)[0]
        cnt += 1

    if freq_cnt == False:
        counts = counts / out[2]['n_samples']
        
    return counts

def bin_arbitrary_fptd(out = None,
                       bin_dt = 0.04,
                       nbins = 256,
                       nchoices = 2,
                       choice_codes = [-1.0, 1.0],
                       max_t = 10.0): # ['v', 'a', 'w', 't', 'angle']

    """Takes in simulator output and returns a histogram of bin counts
    :Arguments:
        out: tuple
            Output of the 'simulator' function
        bin_dt : float
            If nbins is 0, this determines the desired bin size which in turn automatically 
            determines the resulting number of bins.
        nbins : int
            Number of bins to bin reaction time data into. If supplied as 0, bin_dt instead determines the number of 
            bins automatically.
        nchoices: int <default=2>
            Number of choices allowed by the simulator.
        choice_codes = list <default=[-1.0, 1.0]
            Choice labels to be used.
        max_t: float
            Maximum RT to consider.

    Returns:
        2d array (nbins, nchoices): A histogram of bin counts
    """

    # Generate bins
    if nbins == 0:
        nbins = int(max_t / bin_dt)
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, max_t, nbins)
        bins[nbins] = np.inf
    else:    
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, max_t, nbins)
        bins[nbins] = np.inf

    cnt = 0
    counts = np.zeros( (nbins, nchoices) ) 

    for choice in choice_codes:
        counts[:, cnt] = np.histogram(out[:, 0][out[:, 1] == choice], bins = bins)[0] 
        #print(np.histogram(out[:, 0][out[:, 1] == choice], bins = bins)[1])
        cnt += 1
    return counts

model_config = {'ddm': {'params':['v', 'a', 'z', 't'],
                        'param_bounds': [[-3.0, 0.3, 0.1, 1e-3], [3.0, 2.5, 0.9, 2.0]],
                        'param_bounds_cnn': [[-2.5, 0.5, 0.25, 1e-3], [2.5, 2.2, 0.75, 1.95]], # [-2.5, 0.5, 0.25, 0.05], [2.5, 2.2, 0.75, 1.95]]
                        'boundary': bf.constant,
                        'n_params': 4,
                        'default_params': [0.0, 1.0, 0.5, 1e-3],
                        'hddm_include': ['z']},
                'ddm_vanilla': {'params':['v', 'a', 'z', 't'],
                                'param_bounds': [[5.0, 0.1, 0.05, 0], [5.0, 5.0, 0.95, 3.0]],
                                'boundary': bf.constant,
                                'n_params': 4,
                                'default_params': [0.0, 2.0, 0.5, 0],
                                'hddm_include': ['z']},
                'angle':{'params': ['v', 'a', 'z', 't', 'theta'],
                         'param_bounds': [[-3.0, 0.3, 0.2, 1e-3, -0.1], [3.0, 2.0, 0.8, 2.0, 1.45]],
                         'param_bounds_cnn': [[-2.5, 0.2, 0.1, 0.0, 0.0], [2.5, 2.0, 0.9, 2.0, (np.pi / 2 - .2)]], # [(-2.5, 2.5), (0.2, 2.0), (0.1, 0.9), (0.0, 2.0), (0, (np.pi / 2 - .2))]
                        'boundary': bf.angle,
                        'n_params': 5,
                        'default_params': [0.0, 1.0, 0.5, 1e-3, 0.0],
                        'hddm_include':['z', 'theta']},
                'weibull':{'params': ['v', 'a', 'z', 't', 'alpha', 'beta'],
                           'param_bounds': [[-2.5, 0.3, 0.2, 1e-3, 0.31, 0.31], [2.5, 2.5, 0.8, 2.0, 4.99, 6.99]],
                           'param_bounds_cnn': [[-2.5, 0.2, 0.1, 0.0, 0.5, 0.5], [2.5, 2.0, 0.9, 2.0, 5.0, 7.0]], # [(-2.5, 2.5), (0.2, 2.0), (0.1, 0.9), (0.0, 2.0), (0.5, 5.0), (0.5, 7.0)]
                          'boundary': bf.weibull_cdf,
                          'n_params': 6,
                          'default_params': [0.0, 1.0, 0.5, 1e-3, 3.0, 3.0],
                          'hddm_include': ['z', 'alpha', 'beta']},
                'levy':{'params':['v', 'a', 'z', 'alpha', 't'],
                        'param_bounds':[[-3.0, 0.3, 0.1, 1.0, 1e-3], [3.0, 2.0, 0.9, 2.0, 2]],
                        'param_bounds_cnn':[[-2.5, 0.2, 0.1, 1.0, 0.0], [2.5, 2.0, 0.9, 2.0, 2.0]], # [(-2.5, 2.5), (0.2, 2), (0.1, 0.9), (1.0, 2.0), (0.0, 2.0)]
                        'boundary': bf.constant,
                        'n_params': 5,
                         'default_params': [0.0, 1.0, 0.5, 1.5, 1e-3],
                         'hddm_include': ['z', 'alpha']},
                'full_ddm':{'params':['v', 'a', 'z', 't', 'sz', 'sv', 'st'],
                            'param_bounds':[[-3.0, 0.3, 0.3, 0.25, 1e-3, 1e-3, 1e-3], [3.0, 2.5, 0.7, 2.25, 0.2, 2.0, 0.25]],
                            'param_bounds_cnn': [[-2.5, 0.2, 0.1, 0.25, 0.0, 0.0, 0.0], [2.5, 2.0, 0.9, 2.5, 0.4, 1.0, 0.5]], #  [(-2.5, 2.5), (0.2, 2.0), (0.1, 0.9), (0.25, 2.5), (0, 0.4), (0, 1), (0.0, 0.5)]
                            'boundary': bf.constant,
                            'n_params': 7,
                            'default_params': [0.0, 1.0, 0.5, 0.25, 1e-3, 1e-3, 1e-3],
                            'hddm_include': ['z', 'st', 'sv', 'sz']},
                'full_ddm_vanilla': {'params':['v', 'a', 'z', 't', 'sz', 'sv', 'st'],
                                    'param_bounds':[[-5.0, 0.1, 0.3, 0.25, 0, 0, 0], [5.0, 5.0, 0.7, 2.25, 0.25, 4.0, 0.25]],
                                    'boundary': bf.constant,
                                    'n_params': 7,
                                    'default_params': [0.0, 1.0, 0.5, 0.25, 0, 0, 0],
                                    'hddm_include': ['z', 'st', 'sv', 'sz']},
                'ornstein':{'params':['v', 'a', 'z', 'g', 't'],
                            'param_bounds':[[-2.0, 0.3, 0.2, -1.0, 1e-3], [2.0, 2.0, 0.8, 1.0, 2]],
                            'param_bounds_cnn': [[-2.5, 0.2, 0.1, -1.0, 0.0], [2.5, 2.0, 0.9, 1.0, 2.0]], # [(-2.5, 2.5), (0.2, 2.0), (0.1, 0.9), (-1.0, 1.0), (0.0, 2.0)]
                            'boundary': bf.constant,
                            'n_params': 5,
                            'default_params': [0.0, 1.0, 0.5, 0.0, 1e-3],
                            'hddm_include': ['z', 'g']},
                'ddm_sdv':{'params':['v', 'a', 'z', 't', 'sv'],
                           'param_bounds':[[-3.0, 0.3, 0.1, 1e-3, 1e-3],[ 3.0, 2.5, 0.9, 2.0, 2.5]],
                           'param_bounds_cnn': [[-3.0, 0.3, 0.1, 0.0, 0.0], [3.0, 2.5, 0.9, 2.0, 2.5]], # [(-3, 3), (0.3, 2.5), (0.1, 0.9), (0.0, 2.0), (0.0, 2.5)]
                           'boundary': bf.constant,
                           'n_params': 5,
                           'default_params': [0.0, 1.0, 0.5, 1e-3, 1e-3],
                           'hddm_include': ['z', 'sv']
                           },
                }

model_config['weibull_cdf'] = model_config['weibull'].copy()
model_config['full_ddm2'] = model_config['full_ddm'].copy()

def simulator(theta, 
              model = 'angle', 
              n_samples = 1000,
              delta_t = 0.001,  # n_trials
              max_t = 20,
              no_noise = False,
              bin_dim = None,
              bin_pointwise = False):
    """Basic data simulator for the models included in HDDM. 


    :Arguments:
        theta : list or numpy.array or panda DataFrame
            Parameters of the simulator. If 2d array, each row is treated as a 'trial' 
            and the function runs n_sample * n_trials simulations.
        model: str <default='angle'>
            Determines the model that will be simulated.
        n_samples: int <default=1000>
            Number of simulation runs (for each trial if supplied n_trials > 1)
        n_trials: int <default=1>
            Number of trials in a simulations run (this specifically addresses trial by trial parameterizations)
        delta_t: float
            Size fo timesteps in simulator (conceptually measured in seconds)
        max_t: float
            Maximum reaction the simulator can reach
        no_noise: bool <default=False>
            Turn noise of (useful for plotting purposes mostly)
        bin_dim: int <default=None>
            Number of bins to use (in case the simulator output is supposed to come out as a count histogram)
        bin_pointwise: bool <default=False>
            Wheter or not to bin the output data pointwise. If true the 'RT' part of the data is now specifies the
            'bin-number' of a given trial instead of the 'RT' directly. You need to specify bin_dim as some number for this to work.
    
    :Return: tuple 
        can be (rts, responses, metadata)
        or     (rt-response histogram, metadata)
        or     (rts binned pointwise, responses, metadata)

    """

    # Useful for sbi
    if type(theta) == list:
        print('theta is supplied as list --> simulator assumes n_trials = 1')
        theta = np.asarray(theta).astype(np.float32)
    elif type(theta) == np.ndarray:
        theta = theta.astype(np.float32)
    elif type(theta) == pd.core.frame.DataFrame:
        theta = theta[model_config[model]['params']].values.astype(np.float32)
    else:
        theta = theta.numpy().astype(float32)
    
    if len(theta.shape) < 2:
        theta = np.expand_dims(theta, axis = 0)
    
    # # Is this necessary ?
    # if theta.shape[0] != n_trials:
    #     print('ERROR number of trials does not match first dimension of theta array')
    #     return
    
    if theta.ndim > 1:
        n_trials = theta.shape[0]
    else:
        n_trials = 1
    
    # 2 choice models 
    if no_noise:
        s = 0.0
    else: 
        s = 1.0

    if model == 'test':
        x = ddm_flexbound(v = theta[:, 0],
                          a = theta[:, 1], 
                          z = theta[:, 2],
                          t = theta[:, 3],
                          s = s,
                          n_samples = n_samples,
                          n_trials = n_trials,
                          delta_t = delta_t,
                          boundary_params = {},
                          boundary_fun = bf.constant,
                          boundary_multiplicative = True,
                          max_t = max_t)
    
    if model == 'ddm' or model == 'ddm_elife' or model == 'ddm_analytic':
        x = ddm_flexbound(v = theta[:, 0],
                          a = theta[:, 1], 
                          z = theta[:, 2],
                          t = theta[:, 3],
                          s = s,
                          n_samples = n_samples,
                          n_trials = n_trials,
                          delta_t = delta_t,
                          boundary_params = {},
                          boundary_fun = bf.constant,
                          boundary_multiplicative = True,
                          max_t = max_t)
    
    if model == 'angle' or model == 'angle2':
        x = ddm_flexbound(v = theta[:, 0], 
                          a = theta[:, 1],
                          z = theta[:, 2], 
                          t = theta[:, 3], 
                          s = s,
                          boundary_fun = bf.angle, 
                          boundary_multiplicative = False,
                          boundary_params = {'theta': theta[:, 4]}, 
                          delta_t = delta_t,
                          n_samples = n_samples,
                          n_trials = n_trials,
                          max_t = max_t)
    
    if model == 'weibull_cdf' or model == 'weibull_cdf2' or model == 'weibull_cdf_ext' or model == 'weibull_cdf_concave' or model == 'weibull':
        x = ddm_flexbound(v = theta[:, 0], 
                          a = theta[:, 1], 
                          z = theta[:, 2], 
                          t = theta[:, 3], 
                          s = s,
                          boundary_fun = bf.weibull_cdf, 
                          boundary_multiplicative = True, 
                          boundary_params = {'alpha': theta[:, 4], 'beta': theta[:, 5]}, 
                          delta_t = delta_t,
                          n_samples = n_samples,
                          n_trials = n_trials,
                          max_t = max_t)
    
    if model == 'levy':
        x = levy_flexbound(v = theta[:, 0], 
                           a = theta[:, 1], 
                           z = theta[:, 2], 
                           alpha_diff = theta[:, 3], 
                           t = theta[:, 4],
                           s = s, 
                           boundary_fun = bf.constant, 
                           boundary_multiplicative = True, 
                           boundary_params = {},
                           delta_t = delta_t,
                           n_samples = n_samples,
                           n_trials = n_trials,
                           max_t = max_t)
    
    if model == 'full_ddm' or model == 'full_ddm2':
        x = full_ddm(v = theta[:, 0],
                     a = theta[:, 1],
                     z = theta[:, 2], 
                     t = theta[:, 3], 
                     sz = theta[:, 4], 
                     sv = theta[:, 5], 
                     st = theta[:, 6], 
                     s = s,
                     boundary_fun = bf.constant, 
                     boundary_multiplicative = True, 
                     boundary_params = {}, 
                     delta_t = delta_t,
                     n_samples = n_samples,
                     n_trials = n_trials,
                     max_t = max_t)

    if model == 'ddm_sdv':
        x = ddm_sdv(v = theta[:, 0], 
                    a = theta[:, 1], 
                    z = theta[:, 2], 
                    t = theta[:, 3],
                    sv = theta[:, 4],
                    s = s,
                    boundary_fun = bf.constant,
                    boundary_multiplicative = True, 
                    boundary_params = {},
                    delta_t = delta_t,
                    n_samples = n_samples,
                    n_trials = n_trials,
                    max_t = max_t)
        
    if model == 'ornstein' or model == 'ornstein_uhlenbeck':
        x = ornstein_uhlenbeck(v = theta[:, 0], 
                               a = theta[:, 1], 
                               z = theta[:, 2], 
                               g = theta[:, 3], 
                               t = theta[:, 4],
                               s = s,
                               boundary_fun = bf.constant,
                               boundary_multiplicative = True,
                               boundary_params = {},
                               delta_t = delta_t,
                               n_samples = n_samples,
                               n_trials = n_trials,
                               max_t = max_t)

    # 3 Choice models
    if no_noise:
        s = np.tile(np.array([0.0, 0.0, 0.0], dtype = np.float32), (n_trials, 1))
    else:
        s = np.tile(np.array([1.0, 1.0, 1.0], dtype = np.float32), (n_trials, 1))

    if model == 'race_model_3':
        x = race_model(v = theta[:, :3],
                       a = theta[:, [3]],
                       z = theta[:, 4:7],
                       t = theta[:, [7]],
                       s = s,
                       boundary_fun = bf.constant,
                       boundary_multiplicative = True,
                       boundary_params = {},
                       delta_t = delta_t,
                       n_samples = n_samples,
                       n_trials = n_trials,
                       max_t = max_t)
        
    if model == 'lca_3':
        x = lca(v = theta[:, :3],
                a = theta[:, [4]],
                z = theta[:, 4:7],
                g = theta[:, [7]],
                b = theta[:, [8]],
                t = theta[:, [9]],
                s = s,
                boundary_fun = bf.constant,
                boundary_multiplicative = True,
                boundary_params = {},
                delta_t = delta_t,
                n_samples = n_samples,
                n_trials = n_trials,
                max_t = max_t)

    # 4 Choice models
    if no_noise:
        s = np.tile(np.array([0.0, 0.0, 0.0, 0.0], dtype = np.float32), (n_trials, 1))
    else:
        s = np.tile(np.array([1.0, 1.0, 1.0, 0.0], dtype = np.float32), (n_trials, 1))

    if model == 'race_model_4':
        x = race_model(v = theta[:, :4],
                       a = theta[:, [4]],
                       z = theta[:, 5:9],
                       t = theta[:, [9]],
                       s = s,
                       boundary_fun = bf.constant,
                       boundary_multiplicative = True,
                       boundary_params = {},
                       delta_t = delta_t,
                       n_samples = n_samples,
                       n_trials = n_trials,
                       max_t = max_t)
        
    if model == 'lca_4':
        x = lca(v = theta[:, :4],
                a = theta[:, [4]],
                z = theta[:, 5:9],
                g = theta[:, [9]],
                b = theta[:, [10]],
                t = theta[:, [11]],
                s = s,
                boundary_fun = bf.constant,
                boundary_multiplicative = True,
                boundary_params = {},
                delta_t = delta_t,
                n_samples = n_samples,
                n_trials = n_trials,
                max_t = max_t)

    # Seq / Parallel models (4 choice)
    if no_noise:
        s = 0.0
    else: 
        s = 1.0

        
    if model == 'ddm_seq2':
        x = ddm_flexbound_seq2(v_h = theta[:, [0]],
                               v_l_1 = theta[:, [1]],
                               v_l_2 = theta[:, [2]],
                               a = theta[:, [3]],
                               z_h = theta[:, [4]],
                               z_l_1 = theta[:, [5]],
                               z_l_2 = theta[:, [6]],
                               t = theta[:, [7]],
                               s = s,
                               n_samples = n_samples,
                               n_trials = n_trials,
                               delta_t = delta_t,
                               max_t = max_t,
                               boundary_fun = bf.constant,
                               boundary_multiplicative = True,
                               boundary_params = {})

    if model == 'ddm_par2':
        x = ddm_flexbound_par2(v_h = theta[:, [0]],
                               v_l_1 = theta[:, [1]],
                               v_l_2 = theta[:, [2]],
                               a = theta[:, [3]],
                               z_h = theta[:, [4]],
                               z_l_1 = theta[:, [5]],
                               z_l_2 = theta[:, [6]],
                               t = theta[:, [7]],
                               s = s,
                               n_samples = n_samples,
                               n_trials = n_trials,
                               delta_t = delta_t,
                               max_t = max_t,
                               boundary_fun = bf.constant,
                               boundary_multiplicative = True,
                               boundary_params = {})

    if model == 'ddm_mic2':
        x = ddm_flexbound_mic2(v_h = theta[:, [0]],
                               v_l_1 = theta[:, [1]],
                               v_l_2 = theta[:, [2]],
                               a = theta[:, [3]],
                               z_h = theta[:, [4]],
                               z_l_1 = theta[:, [5]],
                               z_l_2 = theta[:, [6]],
                               d = theta[:, [7]],
                               t = theta[:, [8]],
                               s = s,
                               n_samples = n_samples,
                               n_trials = n_trials,
                               delta_t = delta_t,
                               max_t = max_t,
                               boundary_fun = bf.constant,
                               boundary_multiplicative = True,
                               boundary_params = {})
    
    # Output compatibility
    if n_trials == 1:
        x = (np.squeeze(x[0], axis = 1), np.squeeze(x[1], axis = 1), x[2])
    if n_trials > 1 and n_samples == 1:
        x = (np.squeeze(x[0], axis = 0), np.squeeze(x[1], axis = 0), x[2])

    x[2]['model'] = model

    if bin_dim == 0 or bin_dim == None:
        return x
    elif bin_dim > 0 and n_trials == 1 and not bin_pointwise:
        binned_out = bin_simulator_output(x, nbins = bin_dim)
        return (binned_out, x[2])
    elif bin_dim > 0 and n_trials == 1 and bin_pointwise:
        binned_out = bin_simulator_output_pointwise(x, nbins = bin_dim)
        return (np.expand_dims(binned_out[:,0], axis = 1), np.expand_dims(binned_out[:, 1], axis = 1), x[2])
    elif bin_dim > 0 and n_trials > 1 and n_samples == 1 and bin_pointwise:
        binned_out = bin_simulator_output_pointwise(x, nbins = bin_dim)
        return (np.expand_dims(binned_out[:,0], axis = 1), np.expand_dims(binned_out[:, 1], axis = 1), x[2])
    elif bin_dim > 0 and n_trials > 1 and n_samples > 1 and bin_pointwise:
        return 'currently n_trials > 1 and n_samples > 1 will not work together with bin_pointwise'
    elif bin_dim > 0 and n_trials > 1 and not bin_pointwise:
        return 'currently binned outputs not implemented for multi-trial simulators'
    elif bin_dim == -1:
        return 'invalid bin_dim'
    