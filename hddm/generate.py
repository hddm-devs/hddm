from __future__ import division

import numpy as np
import numpy.lib.recfunctions as rec
from scipy.stats import uniform, norm
from copy import copy

try:
    from IPython.Debugger import Tracer; debug_here = Tracer()
except:
    pass


####################################################################
# Functions to generate RT distributions with specified parameters #
####################################################################
def gen_rts(self, params, samples=1000, steps=1000, T=5, structured=False, subj_idx=None):
    dt = steps / T
    drifts = simulate_drifts(params, samples, steps, T)
    rts = find_thresholds(drifts, a)/dt

    if structured:
        if subj_idx is None:
            data = np.empty(rts.shape, dtype = ([('response', np.float), ('rt', np.float)]))
        else:
            data = np.empty(rts.shape, dtype = ([('response', np.float), ('rt', np.float), ('subj_idx', np.float)]))
            data['subj_idx'] = subj_idx
        data['response'][rts>0] = 1.
        data['response'][rts<0] = 0.
        data['rt'] = rts

        return data
    else:
        return rts

    
def simulate_drifts(params, samples, steps, T):
    dt = steps / T
    # Draw starting delays from uniform distribution around [ter-0.5*ster, ter+0.5*ster].
    start_delays = np.abs(uniform.rvs(size=(samples), scale=(params['ster'])) - params['ster']/2. + params['ter'])*dt

    # Draw starting points from uniform distribution around [z-0.5*sz, z+0.5*sz]
    starting_points = uniform.rvs(size=samples, scale=params['sz']) - params['sz']/2. + params['z']

    # Draw drift rates from a normal distribution
    drift_rates = norm.rvs(loc=params['v'], scale=params['sv'], size=samples)/dt

    # Generate samples from a normal distribution and
    # add the drift rates
    step_sizes = norm.rvs(loc=0, scale=np.sqrt(dt), size=(samples, steps))/dt  + np.tile(drift_rates, (steps, 1)).T
    # Go through every line and zero out the non-decision component
    for i in range(samples):
        step_sizes[i,:start_delays[i]] = 0

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples. Add the starting points as a constant offset.
    drifts = np.cumsum(step_sizes, axis=1) + np.tile(starting_points, (steps, 1)).T

    return drifts

def find_thresholds(drifts, a):
    steps = drifts.shape[1]
    # Find lines over the threshold
    rts_upper = []
    rts_lower = []
    # Go through every drift individually
    for drift in drifts:
        # Check if upper bound was reached, and where
        thresh_upper = np.where((drift > a))[0]
        upper = np.Inf
        lower = np.Inf
        if thresh_upper.shape != (0,):
            # This provides the RT, append
            upper = thresh_upper[0]

        # Check if lower bound was reached, and where
        thresh_lower = np.where((drift < 0))[0]
        if thresh_lower.shape != (0,):
            # Lower RT
            lower = thresh_lower[0]

        if upper == lower == np.Inf:
            # Threshold not crossed
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

    return rts

def gen_mask(drifts, a):
    steps = drifts.shape[1]
    mask = np.ones(drifts.shape, dtype='bool')
    for drift,mask_drift in zip(drifts, mask):
        # For plotting, we want to cut off the diffusion processes
        # after they reached the threshold. So get all thresholds,
        # and create a mask with Trues if below threshold. Mask is used
        # for displaying only.
        thresh = np.where((drift > a) | (drift < 0))[0]

        if thresh.shape != (0,):
            mask_drift[thresh[0]:] = np.zeros((steps-thresh[0]), dtype='bool')
    return mask

def _gen_rts_fastdm(v=0, sv=0, z=0.5, sz=0, a=1, ter=0.3, ster=0, samples=500, fname=None, structured=True):
    """Generate simulated RTs with fixed parameters."""
    if fname is None:
        fname = 'example_DDM.txt'
    subprocess.call([sampler_exec, '-v', str(v), '-V', str(sv), '-z', str(z), '-Z', str(sz), '-a', str(a), '-t', str(ter), '-T', str(ster), '-n', str(samples), '-o', fname])
    data = np.loadtxt(fname)
    if structured:
        data.dtype = np.dtype([('response', np.float), ('rt', np.float)])

    return data

def gen_rand_data(samples=500, params=None, gen_data=True, no_var=False, tag=None):
    """Generate simulated RTs with random parameters."""
    #z = np.random.normal(loc=1, scale=2)
    #ster = np.random.uniform(loc=0, scale=.5)
    #params_true = {'v': np.random.normal(loc=-2, scale=4), 'sv': np.random.normal(loc=0, scale=.5), 'z': z, 'sz': np.random.normal(loc=0, scale=.5), 'ter': np.random.normal(loc=ster/2., scale=ster/2.), 'ster': ster, 'a': z+np.random.normal(loc=.5, scale=3)}
    if params is None:
        if not no_var:
            params = {'v': .5, 'sv': 0.3, 'z': 1., 'sz': 0.25, 'ter': .3, 'ster': 0.1, 'a': 2}
        else:
            params = {'v': .5, 'sv': 0., 'z': 1., 'sz': 0., 'ter': .3, 'ster': 0., 'a': 2}

    if gen_data:
        # Create RT data
        data = gen_rts(params, samples=samples, structured=True)

    if tag is None:
        tag = ''
    if gen_data:
        np.save('data_%s'%tag, data)
    else:
        data = np.load('data_%s.npy'%tag)  

    return (data, params)

def gen_rand_correlation_data(v=.5, corr=.1):
    params = {'v': v,
              'sv': .001,
              'ter': .3,
              'ster': 0.,
              'sz':0}

    all_data = []
    a_offset = 2
    for i in np.linspace(-1,1,10):
        params['a'] = a_offset + i*corr
        params['z'] = (a_offset + i*corr)/2.
        data = gen_rand_subj_data(num_subjs=1, params=params, samples=20, add_noise=False)[0]
        theta = np.ones(data.shape) * i
        theta.dtype = dtype=np.dtype([('theta', np.float)])
        stim = np.tile('test', data.shape)
        stim.dtype = np.dtype([('stim', 'S4')])
        
        data = rec.append_fields(data, names=['theta', 'stim'],
                                 data=[theta, stim],
                                 usemask=False)
        all_data.append(data)

    return np.concatenate(all_data)
    
def gen_rand_subj_data(num_subjs=10, params=None, samples=100, gen_data=True, add_noise=True, tag=None):
    """Generate simulated RTs of multiple subjects with fixed parameters."""
    # Set global parameters
    #z = rnd(loc=1, scale=2)
    #ster = rnd(loc=0, scale=.5)
    #self.params_true = {'v': rnd(loc=-2, scale=4), 'sv': rnd(loc=0, scale=.5), 'z': z, 'sz': rnd(loc=0, scale=.5), 'ter': rnd(loc=ster/2., scale=ster/2.), 'ster': ster, 'a': z+rnd(loc=.5, scale=3)}
    if params is None:
        params = {'v': .5, 'sv': 0.1, 'z': 1., 'sz': 0.1, 'ter': 1., 'ster': 0.1, 'a': 2}

    params_subjs = []
    #data = np.empty((samples*num_subjs, 3), dtype=np.float)
    resps = []
    rts = []
    subj_idx = []
    data_gens = []
    # Derive individual parameters
    for i in range(num_subjs):
        params_subj = copy(params)
        # Add noise to all values
        if add_noise:
            for param, value in params_subj.iteritems():
                if param == 'ter' or param == 'ster' or param == 'z':
                    continue
                elif param[0] == 's':
                    params_subj[param] = np.abs(value + np.random.randn()*.01)
                else:
                    params_subj[param] = np.abs(value + np.random.randn()*.05)
        params_subj['z'] = params_subj['a']/2.
        params_subjs.append(params_subj)

        if gen_data:
            # Create RT data
            data_gen = gen_rts(params_subj, samples=samples, structured=True, subj_idx=i)
            data_gens.append(data_gen)
    
    if tag is None:
        tag = ''
    if gen_data:
        data = np.concatenate(data_gens)
        np.save('data_%s'%tag, data)
    else:
        data = np.load('data_%s.npy'%tag)

    return (data, params)
