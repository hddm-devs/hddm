from __future__ import division

import numpy as np
import numpy.lib.recfunctions as rec
from scipy.stats import uniform, norm
from copy import copy
import random
from numpy.random import rand, randn

import hddm

try:
    from IPython.Debugger import Tracer; debug_here = Tracer()
except:
    pass


def gen_antisaccade_rts(params, samples_pro=500, samples_anti=500, steps=5000, T=5., subj_idx=None):
    # Generate prosaccade trials
    pro_params = copy(params)
    del pro_params['t_switch']
    del pro_params['v_switch']
    
    rts = np.empty(samples_pro+samples_anti, dtype=[('response', np.float), ('rt', np.float), ('instruct', np.int), ('subj_idx', np.int)])

    pro_rts = gen_rts(pro_params, samples=samples_pro, steps=steps, T=T, subj_idx=subj_idx)
    anti_rts = gen_rts(params, samples=samples_anti, steps=steps, T=T, subj_idx=subj_idx)

    rts['instruct'][:samples_pro] = 0
    rts['instruct'][samples_pro:] = 1
    rts['response'][:samples_pro] = np.array((pro_rts > 0), float)
    rts['response'][samples_pro:] = np.array((anti_rts > 0), float)
    rts['rt'][:samples_pro] = np.abs(pro_rts)
    rts['rt'][samples_pro:] = np.abs(anti_rts)
    
    if subj_idx is not None:
        rts['subj_idx'] = subj_idx
    
    return rts
    
####################################################################
# Functions to generate RT distributions with specified parameters #
####################################################################
def _interpolate_thresh(x1, x2, t1, t2, boundary):
        # Interpolate to find crossing
        m = (y2-y1) / (x2-x1) # slope
        # y = m*x + b
        b = y1 - m*x1 # intercept
        return (boundary - b) / m
def gen_rts(params, samples=1000, dt = 1e-4, intra_sv=1., structured=False, subj_idx=None):

    if samples is None:
        samples = 1
    nn = 1000
    a = params['a']
    v = params['v']
    
    #create delay
    if params.has_key('T'):
        start_delay = (uniform.rvs(loc=params['t'], scale=params['T'], size=samples) \
                       - params['T']/2.)
    else:
        start_delay = np.ones(samples)*params['t']
    
    #create starting_points
    if params.has_key('Z'):
        starting_points = (uniform.rvs(loc=params['z'], scale=params['Z'], size=samples) \
                           - params['Z']/2.)*a
    else:
        starting_points = np.ones(samples)*params['z']*a
    
    
    rts = np.empty(samples)
    step_size = np.sqrt(dt)*intra_sv
    
    for i_sample in xrange(samples):
        crossed = False
        iter = 0
        y_0 = starting_points[i_sample]
        # drifting...
        while (not crossed):
            iter += 1
            if params.has_key('V') and params['V'] != 0:
                drift_rates = norm.rvs(v, params['V'], size=nn)
                prob_up =  0.5*(1+np.sqrt(dt)/intra_sv*drift_rates)
            else:
                prob_up =  0.5*(1+np.sqrt(dt)/intra_sv*v)
            position = ((rand(nn) < prob_up) *2 - 1) * step_size
            position[0] += y_0
            position = np.cumsum(position) 
            cross_idx = np.where((position < 0) | (position > a))[0]
            if cross_idx.shape[0]>0:
                crossed = True
            else:
                y_0 = position[-1]

        #find the boundary interception        
        y2 = position[cross_idx[0]]
        if cross_idx[0]!=0:
            y1 = position[cross_idx[0]-1]            
        else:
            y1 = y_0
        m = (y2 - y1)/dt  # slope
        # y = m*x + b
        b = y2 - m*((iter-1)*nn+cross_idx[0])*dt # intercept
        if y2 < 0:
            rt = ((0 - b) / m)
        else:
            rt = ((a - b) / m)
        rts[i_sample] = (rt + start_delay[i_sample])*np.sign(y2)
        
    if not structured:
        return rts
    else:
        if subj_idx is None:
            data = np.empty(rts.shape, dtype = ([('response', np.float), ('rt', np.float)]))
        else:
            data = np.empty(rts.shape, dtype = ([('response', np.float), ('rt', np.float), ('subj_idx', np.float)]))
            data['subj_idx'] = subj_idx
        data['response'][rts>0] = 1.
        data['response'][rts<0] = 0.
        data['rt'] = rts

        return data

def simulate_drifts(params, samples, steps, T, intra_sv=1.):
    dt = steps / T
    # Draw starting delays from uniform distribution around [ter-0.5*ster, ter+0.5*ster].
    start_delays = np.abs(uniform.rvs(loc=params['t'], scale=params['T'], size=samples) - params['T']/2.)*dt

    # Draw starting points from uniform distribution around [z-0.5*sz, z+0.5*sz]
    # Starting point is relative, so have to convert to absolute first: z*a
    starting_points = (uniform.rvs(loc=params['z'], scale=params['Z'], size=samples) - params['Z']/2.)*params['a']

    # Draw drift rates from a normal distribution
    if not params.has_key('t_switch'):
        # No change in drift rate
        if params['V'] == 0.:
            drift_rate = np.repeat(params['v'], samples)
        else:
            drift_rate = norm.rvs(loc=params['v'], scale=params['V'], size=samples)
        drift_rates = (np.tile(drift_rate, (steps, 1))/dt).T
    else:
        # Drift rate changes during the trial
        if params['V'] != 0:
            raise NotImplementedError, "Inter-trial drift variability is not supported."
        drift_rate = np.empty(steps, dtype=np.float)
        switch_pos = (params['t']+params['t_switch'])*dt
        if switch_pos > steps:
            drift_rate[:] = params['v']
        else:
            drift_rate[:int(switch_pos)] = params['v']
            drift_rate[int(switch_pos)+1:] = params['v_switch']
        drift_rates = np.tile(drift_rate, (samples,1))/dt

    # Generate samples from a normal distribution and
    # add the drift rates
    step_sizes = norm.rvs(loc=0, scale=np.sqrt(dt)*intra_sv, size=(samples, steps))/dt  + drift_rates
    # Go through every line and zero out the non-decision component
    for i in range(int(samples)):
        step_sizes[i,:start_delays[i]] = 0

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples. Add the starting points as a constant offset.
    drifts = np.cumsum(step_sizes, axis=1) + np.tile(starting_points, (steps, 1)).T

    return drifts

def find_thresholds(drifts, a, return_non_crossings=False):
    def interpolate_thresh(cross, boundary):
        # Interpolate to find crossing
        x1 = cross-1
        x2 = cross
        y1 = drift[x1]
        y2 = drift[x2]
        m = (y2-y1) / (x2-x1) # slope
        # y = m*x + b
        b = y1 - m*x1 # intercept
        return (boundary - b) / m

    steps = drifts.shape[1]
    # Find lines over the threshold
    rts_upper = []
    rts_lower = []
    non_crossings = 0
    # Go through every drift individually
    for drift in drifts:
        # Check if upper bound was reached, and where
        thresh_upper = np.where((drift > a))[0]
        upper = np.Inf
        lower = np.Inf
        if thresh_upper.shape != (0,):
            upper = interpolate_thresh(thresh_upper[0], a)

        # Check if lower bound was reached, and where
        thresh_lower = np.where((drift < 0))[0]
        if thresh_lower.shape != (0,):
            lower = interpolate_thresh(thresh_lower[0], 0)

        if upper == lower == np.Inf:
            # Threshold never crossed
            non_crossings += 1
            continue

        # Determine which one hit the threshold before
        if upper < lower:
            rts_upper.append(np.float(upper))
        else:
            rts_lower.append(np.float(lower))

    rts_upper = np.asarray(rts_upper)
    rts_lower = np.asarray(rts_lower)
    # Switch sign
    rts_lower *= -1
    rts = np.concatenate((rts_upper, rts_lower))

    if return_non_crossings:
        return rts, non_crossings
    else:
        return rts

def _gen_rts_fastdm(v=0, sv=0, z=0.5, sz=0, a=1, ter=0.3, ster=0, samples=500, fname=None, structured=True):
    """Generate simulated RTs with fixed parameters."""
    if fname is None:
        fname = 'example_DDM.txt'
    subprocess.call([sampler_exec, '-v', str(v), '-V', str(sv), '-z', str(z), '-Z', str(sz), '-a', str(a), '-t', str(ter), '-T', str(ster), '-n', str(samples), '-o', fname])
    data = np.loadtxt(fname)
    if structured:
        data.dtype = np.dtype([('response', np.float), ('rt', np.float)])

    return data


def add_contaminate_data(data, params):
    t_max = max(data['rt'])+1.5;
    pi = params['pi']
    gamma = params['gamma']
    n_cont = int(len(data)*pi)
    n_unif = int(n_cont*gamma)
    n_other = n_cont - n_unif
    l_data = range(len(data))
    cont_idx = random.sample(l_data,n_cont)
    unif_idx = cont_idx[:n_unif]
    other_idx = cont_idx[n_unif:]
    data[unif_idx]['rt']  = uniform.rvs(0,t_max,n_unif)
    data[other_idx]['rt']  = uniform.rvs(0,t_max,n_other)
    return data



def gen_rand_data(samples=500, params=None, no_var=False):
    """Generate simulated RTs with random parameters."""
    #z = np.random.normal(loc=1, scale=2)
    #ster = np.random.uniform(loc=0, scale=.5)
    #params_true = {'v': np.random.normal(loc=-2, scale=4), 'V': np.random.normal(loc=0, scale=.5), 'z': z, 'Z': np.random.normal(loc=0, scale=.5), 't': np.random.normal(loc=ster/2., scale=ster/2.), 'T': ster, 'a': z+np.random.normal(loc=.5, scale=3)}
    if params is None:
        if not no_var:
            params = {'v': .5, 'V': 0.5, 'z': .5, 'Z': 0.5, 't': .3, 'T': 0.3, 'a': 2}
        else:
            params = {'v': .5, 'V': 0., 'z': .5, 'Z': 0., 't': .3, 'T': 0., 'a': 2}

    # Create RT data
    data = gen_rts(params, samples=samples, structured=True)
    if params.has_key('pi'):
        add_contaminate_data(data, params)
    
    return (data, params)

def gen_rand_cond_data(params_set=None, samples_per_cond=100):
    """Generate simulated RTs with random parameters."""
    # Create RT data
    if params_set is None:
        params_set = [{'v': .5, 'V': 0., 'z': .5, 'Z': 0., 't': .3, 'T': 0., 'a': 2},
                      {'v': 1., 'V': 0., 'z': .5, 'Z': 0., 't': .3, 'T': 0., 'a': 2}]
    n_conds = len(params_set)
    n = samples_per_cond
    data = np.empty(n*n_conds, dtype = ([('response', np.float), ('rt', np.float), ('cond', np.int), ('cond2', np.int)]))
    counter = 0
    for i in range(n_conds):
        i_data = gen_rts(params_set[i], samples=n, structured=True, strict_size=True)
        data[counter:counter+len(i_data)]['response'] = np.sign(i_data['response'])
        data[counter:counter+len(i_data)]['rt'] = np.abs(i_data['rt'])
        data[counter:counter+len(i_data)]['cond'] = i
        data[counter:counter+len(i_data):2]['cond2'] = i+1
        data[counter+1:counter+len(i_data):2]['cond2'] = i+2
        counter += len(i_data)
    if params_set[0].has_key('pi'):
        add_contaminate_data(data, params_set[0])


    return data, params_set


def gen_rand_correlation_data(v=.5, corr=.1):
    params = {'v': v,
              'V': .001,
              'z': .5,
              't': .3,
              'T': 0.,
              'Z':0}

    all_data = []
    a_offset = 2
    for i in np.linspace(-1,1,10):
        params['a'] = a_offset + i*corr
        data = gen_rand_subj_data(num_subjs=1, params=params, samples=20, noise=.1)[0]
        theta = np.ones(data.shape) * i
        theta.dtype = dtype=np.dtype([('theta', np.float)])
        stim = np.tile('test', data.shape)
        stim.dtype = np.dtype([('stim', 'S4')])
        
        data = rec.append_fields(data, names=['theta', 'stim'],
                                 data=[theta, stim],
                                 usemask=False)
        all_data.append(data)

    return np.concatenate(all_data)
    
def _add_noise(params_subj, noise=.1):
    """Add individual noise to each parameter."""
    params_subj = copy(params_subj)
    for param, value in params_subj.iteritems():
        if param != 'z' and param.islower():
            params_subj[param] = np.random.normal(loc=value, scale=noise)

    return params_subj

def gen_rand_subj_data(num_subjs=10, params=None, samples=100, noise=0.1, tag=None):
    """Generate simulated RTs of multiple subjects with fixed parameters."""
    # Set global parameters
    #z = rnd(loc=1, scale=2)
    #ster = rnd(loc=0, scale=.5)
    #self.params_true = {'v': rnd(loc=-2, scale=4), 'V': rnd(loc=0, scale=.5), 'z': z, 'Z': rnd(loc=0, scale=.5), 't': rnd(loc=ster/2., scale=ster/2.), 'T': ster, 'a': z+rnd(loc=.5, scale=3)}
    if params is None:
        params = {'v': .5, 'V': 0.1, 'z': .5, 'Z': 0.1, 't': .3, 'T': 0.1, 'a': 2}

    params_subjs = []
    #data = np.empty((samples*num_subjs, 3), dtype=np.float)
    resps = []
    rts = []
    subj_idx = []
    data_gens = []
    # Derive individual parameters
    for i in range(num_subjs):
        params_subj = copy(params)
        params_subj = _add_noise(params_subj, noise=noise)

        params_subjs.append(params_subj)

        # Create RT data
        data_gen = gen_rts(params_subj, samples=samples, structured=True, subj_idx=i)
        data_gens.append(data_gen)
    
    return (rec.stack_arrays(data_gens, usemask=False), params)

def run_param_combo((V, Z, T, correlation)):
    repeat = 5
    m = []
    params = {'v': .5, 'V': V, 'z': .5, 'Z': Z, 't': .3, 'T': T, 'a': 2, 'e': correlation}
    # Generate data
    for r in range(repeat):
        if correlation is None:
            data, params = gen_rand_subj_data(num_subjs=15, samples=100, params=params)
            m.append(hddm.model.HDDM(data).mcmc())
        else:
            data, params = gen_correlated_rts(num_subjs=15, samples=100, params=params)
            m.append(hddm.model.HDDMOneRegressor(data, e_data='cov').mcmc())

    result = {}
    for param in ['a','v','z','t']:
        result[param] = np.array([model.params_est[param] for model in m])

    # Compute sum squared error
    return result

def run_all_var_individual(num_subjs=15, samples=100, correlation=None, jobs=2, interval=3, multiprocessing=True):
    """Experiment to test how well the simple DDM can recover
    parameters generated by the full DDM. Runs all possible
    combinations of inter-trial variabilities and generates data which
    it tries to recover using the simple DDM.
    """

    # Generate parameter combinations
    Vs = np.r_[0:1.:complex(0,interval)]
    Zs = np.r_[0:.4:complex(0,interval)]
    Ts = np.r_[0:.3:complex(0,interval)]

    # Run param combinations
    import multiprocessing
    p = multiprocessing.Pool(jobs)
    zeros = np.zeros_like(Vs)
    cors = [correlation for i in range(len(zeros))]

    results_V = p.map(run_param_combo, zip(Vs, zeros, zeros, cors))
    results_Z = p.map(run_param_combo, zip(zeros, Zs, zeros, cors))
    results_T = p.map(run_param_combo, zip(zeros, zeros, Ts, cors))
    
    return {'V':(results_V, Vs), 'Z': (results_Z, Zs), 'T':(results_T, Ts)}

def plot_combs_individual(results, correlation=False):
    import matplotlib.pyplot as plt
    params = {'v': .5, 'z': .5, 't': .3, 'a': 2, 'e_cov': .1}
    plot_params = ['a', 'v', 't']
    if correlation:
        plot_params.append('e_cov')

    for i, (variability, (est, variability_vals)) in enumerate(results.iteritems()):
        plt.subplot(3,1,i+1)
        for name in plot_params:
            size = len(est)
            plt.errorbar(variability_vals, 
                         [np.mean(est[i][name])-params[name] for i in range(size)],
                         yerr=[np.std(est[i][name])/np.sqrt(5) for i in range(size)],
                         label=name)
        plt.xlabel(variability)
    plt.legend()


def run_all_var_combs(num_subjs=15, samples=100, correlation=None, jobs=2, interval=3, multiprocessing=True):
    """Experiment to test how well the simple DDM can recover
    parameters generated by the full DDM. Runs all possible
    combinations of inter-trial variabilities and generates data which
    it tries to recover using the simple DDM.
    """
    # Generate parameter combinations
    Vs, Zs, Ts = np.mgrid[0:.2:complex(0,interval), 0:.25:complex(0,interval), 0:.2:complex(0,interval)]

    # Run param combinations
    import multiprocessing
    p = multiprocessing.Pool(jobs)
    results = p.map(run_param_combo, zip(Vs.flatten(), Zs.flatten(), Ts.flatten()))
    
    results = np.array(results).reshape((interval, interval, interval))

    return results, (Vs, Zs, Ts)

        

def gen_correlated_rts(num_subjs=10, params=None, samples=100, correlation=.1, cor_param=None, subj_noise=.01):
    """Generate RT data where cor_param is linearly influenced by another variable."""
    
    if params is None:
        params = {'v': .3, 'V': 0, 'z': .5, 'Z': 0., 't': .3, 'T': 0., 'a': 2, 'e': correlation}

    if cor_param is None:
        cor_param = 'a'

    data = np.empty(samples*num_subjs, dtype = ([('response', np.float), ('rt', np.float), ('subj_idx', np.float), ('cov', np.float)]))
    params_subjs = []
    # Generate RTs with given parameters (influenced by covariate)
    trial = 0
    for subj_idx in range(num_subjs):
        params_subj = copy(params)
        params_subj = _add_noise(params_subj, subj_noise)
        params_subjs.append(params_subj)
        
        for i in range(samples):
            params_trial = copy(params_subj)
            data['subj_idx'][trial] = subj_idx

            # Calculate resulting cor_param values
            data['cov'][trial] = np.random.randn()

            # Calculate new param values based on covariate
            params_trial[cor_param] = data['cov'][trial] * params_subj['e'] + params_subj[cor_param]

            # Generate RT (resample if it didn't reach threshold)
            rt = np.array([])
            while rt.shape == (0,):
                rt = gen_rts(params_trial, samples=1, structured=False)

            data['rt'][trial] = rt
            
            trial+=1

    data['response'][data['rt']>0] = 1.
    data['response'][data['rt']<0] = 0.
    data['rt'] = np.abs(data['rt'])

    return data, params_subjs
    

        
