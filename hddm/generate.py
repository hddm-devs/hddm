
from __future__ import division

import numpy as np
import numpy.lib.recfunctions as rec
from scipy.stats import uniform, norm
from copy import copy, deepcopy
import random
import kabuki

import hddm

def gen_rand_params(include=()):
    """Returns a dict of DDM parameters with random values.

        :Optional:
            include : tuple
                Which optional parameters include. Can be
                any combination of:

                * 'z' (bias, default=0.5)
                * 'V' (inter-trial drift variability)
                * 'Z' (inter-trial bias variability)
                * 'T' (inter-trial non-decision time variability)

                Special arguments are:
                * 'all': include all of the above
                * 'all_inter': include all of the above except 'z'

    """
    params = {}
    if include == 'all':
        include = ['z','V','Z','T']
    elif include == 'all_inter':
        include = ['V','Z','T']

    from numpy.random import rand

    params['V'] = rand() if 'V' in include else 0
    params['Z'] = rand()* 0.3 if 'Z' in include else 0
    params['T'] = rand()* 0.2 if 'T' in include else 0
    params['z'] = .4+rand()*0.2 if 'z' in include else 0.5

    # Simple parameters
    params['v'] = (rand()-.5)*4
    params['t'] = 0.2+rand()*0.3+(params['T']/2)
    params['a'] = 1.0+rand()


    if 'pi' in include or 'gamma' in include:
        params['pi'] = max(rand()*0.1,0.01)
#        params['gamma'] = rand()

    return params


def gen_antisaccade_rts(params=None, samples_pro=500, samples_anti=500, dt=1e-4, subj_idx=0):
    if params is None:
        params = {'v':-2.,
                  'v_switch': 2.,
                  'V_switch': .1,
                  'a': 2.5,
                  't': .3,
                  't_switch': .3,
                  'z':.5,
                  'T': 0, 'Z':0, 'V':0}
    # Generate prosaccade trials
    pro_params = copy(params)
    del pro_params['t_switch']
    del pro_params['v_switch']

    rts = np.empty(samples_pro+samples_anti, dtype=[('response', np.float), ('rt', np.float), ('instruct', int), ('subj_idx', int)])

    pro_rts = gen_rts(pro_params, samples=samples_pro, dt=dt, subj_idx=subj_idx)
    anti_rts = gen_rts(params, samples=samples_anti, dt=dt, subj_idx=subj_idx, method='drift')

    rts['instruct'][:samples_pro] = 0
    rts['instruct'][samples_pro:] = 1
    rts['response'][:samples_pro] = np.array((pro_rts > 0), float)
    rts['response'][samples_pro:] = np.array((anti_rts > 0), float)
    rts['rt'][:samples_pro] = np.abs(pro_rts)
    rts['rt'][samples_pro:] = np.abs(anti_rts)
    rts['subj_idx'] = subj_idx

    return rts, params

####################################################################
# Functions to generate RT distributions with specified parameters #
####################################################################

def gen_rts(params, samples=1000, range_=(-6, 6), dt=1e-3, intra_sv=1., structured=False, subj_idx=None, method='cdf'):
    """
    Returns a numpy.array of randomly simulated RTs from the DDM.

    :Arguments:
        params : dict
            Parameter names and values to use for simulation.

    :Optional:
        samples : int
            Number of RTs to simulate.
        range_ : tuple
            Minimum (negative) and maximum (positve) RTs.
        dt : float
            Number of steps/sec.
        intra_sv : float
            Intra-trial variability.
        structured : bool
            Return a structured array with fields 'RT'
            and 'response'.
        subj_idx : int
            If set, append column 'subj_idx' with value subj_idx.
        method : str
            Which method to use to simulate the RTs:
                * 'cdf': fast, uses the inverse of cumulative density function to sample, dt can be 1e-2.
                * 'drift': slow, simulates each complete drift process, dt should be 1e-4.

    """
    if params.has_key('v_switch') and method != 'drift':
        print "Warning: Only drift method supports changes in drift-rate. v_switch will be ignored."

    # Set optional default values if they are not provided
    for var_param in ('V', 'Z', 'T'):
        if var_param not in params:
            params[var_param] = 0
    if 'z' not in params:
        params['z'] = .5


    if 'V' not in params:
        params['V'] = 0
    if 'Z' not in params:
        params['Z'] = 0
    if method=='cdf_py':
        rts = _gen_rts_from_cdf(params, samples, range_, dt)
    elif method=='drift':
        rts = _gen_rts_from_simulated_drift(params, samples, dt, intra_sv)[0]
    elif method=='cdf':
        rts = hddm.wfpt.gen_rts_from_cdf(params['v'],params['V'],params['a'],params['z'],
                                         params['Z'],params['t'],params['T'],
                                         samples, range_[0], range_[1], dt)
    else:
        raise TypeError, "Sampling method %s not found." % method
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
        data['rt'] = np.abs(rts)

        return data

def kabuki_data_to_hddm_data(kabuki_data):
    "transform data generate by kabuki.generate.gen_rand_data to hddm data"

    kd = kabuki_data
    dtype = ([('response', np.int), ('rt', np.float), ('subj_idx', np.int32), ('condition', 'S20')])
    data = np.empty(kd.shape, dtype = dtype)

    rts = kd['data']
    data['response'][rts>0] = 1.
    data['response'][rts<0] = 0.
    data['rt'] = np.abs(rts)
    data['condition'] = kd['condition']
    data['subj_idx'] = kd['subj_idx']
    return data


def _gen_rts_from_simulated_drift(params, samples=1000, dt = 1e-4, intra_sv=1.):
    """Returns simulated RTs from simulating the whole drift-process.

        :Arguments:
            params : dict
                Parameter names and values.

        :Optional:
            samlpes : int
                How many samples to generate.
            dt : float
                How many steps/sec.
            intra_sv : float
                Intra-trial variability.

        :SeeAlso:
            gen_rts
    """

    from numpy.random import rand

    if samples is None:
        samples = 1
    nn = 1000
    a = params['a']
    v = params['v']

    if params.has_key('v_switch'):
        switch = True
        t_switch = params['t_switch']/dt
        # Hack so that we will always step into a switch
        nn = int(round(t_switch))
    else:
        switch = False

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
    drifts = []

    for i_sample in xrange(samples):
        drift = np.array([])
        crossed = False
        iter = 0
        y_0 = starting_points[i_sample]
        # drifting...
        if params.has_key('V') and params['V'] != 0:
            drift_rate = norm.rvs(v, params['V'])
        else:
            drift_rate = v

        if params.has_key('v_switch'):
            if params.has_key('V_switch') and params['V_switch'] != 0:
                drift_rate_switch = norm.rvs(params['v_switch'], params['V_switch'])
            else:
                drift_rate_switch = params['v_switch']

        prob_up = 0.5*(1+np.sqrt(dt)/intra_sv*drift_rate)

        while (not crossed):
            # Generate nn steps
            iter += 1
            if iter == 2 and switch:
                prob_up = 0.5*(1+np.sqrt(dt)/intra_sv*drift_rate_switch)
            position = ((rand(nn) < prob_up)*2 - 1) * step_size
            position[0] += y_0
            position = np.cumsum(position)
            # Find boundary crossings
            cross_idx = np.where((position < 0) | (position > a))[0]
            drift = np.concatenate((drift, position))
            if cross_idx.shape[0]>0:
                crossed = True
            else:
                # If not crossed, set last position as starting point
                # for next nn steps to continue drift
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

        delay = start_delay[i_sample]/dt
        drifts.append(np.concatenate((np.ones(int(delay))*starting_points[i_sample], drift[:int(abs(rt)/dt)])))

    return rts, drifts

def pdf_with_params(rt, params):
    """Helper function that calls full_pdf and gets the parameters
    from the dict params.

    """
    v = params['v']; V= params['V']; z = params['z']; Z = params['Z']; t = params['t'];
    T = params['T']; a = params['a']
    return hddm.wfpt.full_pdf(rt,v=v,V=V,a=a,z=z,Z=Z,t=t,
                        T=T,err=1e-4, nT=2, nZ=2, use_adaptive=1, simps_err=1e-3)

def _gen_rts_from_cdf(params, samples=1000):
    """Returns simulated RTs sampled from the inverse of the CDF.

       :Arguments:
            params : dict
                Parameter names and values.

        :Optional:
            samlpes : int
                How many samples to generate.

        :SeeAlso:
            gen_rts

    """
    v = params['v']; V = params['V']; z = params['z']; Z = params['Z']; t = params['t'];
    T = params['T']; a = params['a']
    return hddm.likelihoods.wfpt.ppf(np.random.rand(samples), args=(v, V, a, z, Z, t, T))

def add_contaminate_data(data, params):
    """ add contaminated data"""
    if not params.has_key('gamma'):
        gamma = 1
    else:
        gamma = params['gamma']
    t_min = max(np.abs(data['rt']))+0.5
    t_max = t_min+3;
    pi = params['pi']
    n_cont = max(int(len(data)*pi),2)
    n_unif = max(int(n_cont*gamma),2)
    n_other = n_cont - n_unif
    l_data = range(len(data))
    cont_idx = random.sample(l_data,n_cont)
    unif_idx = cont_idx[:n_unif]
    other_idx = cont_idx[n_unif:]

    # create guesses
    response = np.round(uniform.rvs(0,1,size=n_unif))
    data[unif_idx]['rt']  = uniform.rvs(0,t_max,size=n_unif) * response
    data[unif_idx]['rt']  = response
    data[unif_idx[0]]['rt'] = min(abs(data['rt']))/2.
    data[unif_idx[1]]['rt'] = max(abs(data['rt'])) + 0.8

    #create late responses
    response = (np.sign(gen_rts(params, n_other))+1) / 2
    data[other_idx]['rt']  = uniform.rvs(t_min,t_max,size=n_other) * response
    return data

def gen_rand_data(params, method='cdf', **kwargs):
    """Generate simulated RTs with random parameters.

       :Optional:
            params : dict
                Parameter names and values. If not
                supplied, takes random values.
            method : string
                method to generate samples
            the rest of the arguments are forward to kabuki.generate.gen_rand_data

       :Returns:
            data array with RTs
            parameter values

    """

    wfpt = deepcopy(hddm.likelihoods.wfpt_like)
    wfpt.method = method


    # Create RT data
    data, subj_params = kabuki.generate.gen_rand_data(wfpt, params, **kwargs)
    data = kabuki_data_to_hddm_data(data)

    return data, subj_params



def gen_correlated_rts(num_subjs=10, params=None, samples=100, correlation=.1, cor_param=None, subj_noise=.1):
    """Generate RT data where cor_param is linearly influenced by
    another variable.

    """

    if params is None:
        params = {'v': .5, 'V': 0, 'z': .5, 'Z': 0., 't': .3, 'T': 0., 'a': 2, 'e': correlation}

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
            data['cov'][trial] = np.random.rand()-.5

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



