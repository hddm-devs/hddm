

import kabuki
import hddm
import numpy as np
import pandas as pd

from numpy.random import rand
from scipy.stats import uniform, norm
from copy import copy

def gen_single_params_set(include=()):
    """Returns a dict of DDM parameters with random values for a singel conditin
    the function is used by gen_rand_params.

        :Optional:
            include : tuple
                Which optional parameters include. Can be
                any combination of:

                * 'z' (bias, default=0.5)
                * 'sv' (inter-trial drift variability)
                * 'sz' (inter-trial bias variability)
                * 'st' (inter-trial non-decision time variability)

                Special arguments are:
                * 'all': include all of the above
                * 'all_inter': include all of the above except 'z'

    """
    params = {}
    if include == 'all':
        include = ['z','sv','sz','st']
    elif include == 'all_inter':
        include = ['sv','sz','st']

    params['sv'] = 2.5*rand() if 'sv' in include else 0
    params['sz'] = rand()* 0.4 if 'sz' in include else 0
    params['st'] = rand()* 0.35 if 'st' in include else 0
    params['z'] = .4+rand()*0.2 if 'z' in include else 0.5

    # Simple parameters
    params['v'] = (rand()-.5)*8
    params['t'] = 0.2+rand()*0.3
    params['a'] = 0.5+rand()*1.5


    if 'pi' in include or 'gamma' in include:
        params['pi'] = max(rand()*0.1,0.01)
#        params['gamma'] = rand()

    assert hddm.utils.check_params_valid(**params)

    return params


def gen_rand_params(include=(), cond_dict=None, seed=None):
    """Returns a dict of DDM parameters with random values.

        :Optional:
            include : tuple
                Which optional parameters include. Can be
                any combination of:

                * 'z' (bias, default=0.5)
                * 'sv' (inter-trial drift variability)
                * 'sz' (inter-trial bias variability)
                * 'st' (inter-trial non-decision time variability)

                Special arguments are:
                * 'all': include all of the above
                * 'all_inter': include all of the above except 'z'

            cond_dict : dictionary
                cond_dict is used when multiple conditions are desired.
                the dictionary has the form of {param1: [value_1, ... , value_n], param2: [value_1, ... , value_n]}
                and the function will output n sets of parameters. each set with values from the
                appropriate place in the dictionary
                for instance if cond_dict={'v': [0, 0.5, 1]} then 3 parameters set will be created.
                the first with v=0 the second with v=0.5 and the third with v=1.

            seed: float
                random seed

            Output:
            if conditions is None:
                params: dictionary
                    a dictionary holding the parameters values
            else:
                cond_params: a dictionary holding the parameters for each one of the conditions,
                    that has the form {'c1': params1, 'c2': params2, ...}
                    it can be used directly as an argument in gen_rand_data.
                merged_params:
                     a dictionary of parameters that can be used to validate the optimization
                     and learning algorithms.
    """


    #set seed
    if seed is not None:
        np.random.seed(seed)

    #if there is only a single condition then we can use gen_single_params_set
    if cond_dict is None:
       return gen_single_params_set(include=include)

    #generate original parameter set
    org_params = gen_single_params_set(include)

    #create a merged set
    merged_params = org_params.copy()
    for name in cond_dict.keys():
        del merged_params[name]

    cond_params = {};
    n_conds = len(list(cond_dict.values())[0])
    for i in range(n_conds):
        #create a set of parameters for condition i
        #put them in i_params, and in cond_params[c#i]
        i_params = org_params.copy()
        for name in cond_dict.keys():
            i_params[name] = cond_dict[name][i]
            cond_params['c%d' %i] = i_params

            #update merged_params
            merged_params['%s(c%d)' % (name, i)] = cond_dict[name][i]

    return cond_params, merged_params


####################################################################
# Functions to generate RT distributions with specified parameters #
####################################################################

def gen_rts(size=1000, range_=(-6, 6), dt=1e-3,
            intra_sv=1., structured=True, subj_idx=None,
            method='cdf', **params):
    """
    A private function used by gen_rand_data
    Returns a DataFrame of randomly simulated RTs from the DDM.

    :Arguments:
        params : dict
            Parameter names and values to use for simulation.

    :Optional:
        size : int
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
    if 'v_switch' in params and method != 'drift':
        print("Warning: Only drift method supports changes in drift-rate. v_switch will be ignored.")

    # Set optional default values if they are not provided
    for var_param in ('sv', 'sz', 'st'):
        if var_param not in params:
            params[var_param] = 0
    if 'z' not in params:
        params['z'] = .5
    if 'sv' not in params:
        params['sv'] = 0
    if 'sz' not in params:
        params['sz'] = 0

    #check sample
    if isinstance(size, tuple): #this line is because pymc stochastic use tuple for sample size
        if size == ():
            size = 1
        else:
            size = size[0]

    if method=='cdf_py':
        rts = _gen_rts_from_cdf(params, size, range_, dt)
    elif method=='drift':
        rts = _gen_rts_from_simulated_drift(params, size, dt, intra_sv)[0]
    elif method=='cdf':
        rts = hddm.wfpt.gen_rts_from_cdf(params['v'],params['sv'],params['a'],params['z'],
                                         params['sz'],params['t'],params['st'],
                                         size, range_[0], range_[1], dt)
    else:
        raise TypeError("Sampling method %s not found." % method)
    if not structured:
        return rts
    else:
        data = pd.DataFrame(rts, columns=['rt'])
        data['response'] = 1.
        data['response'][data['rt']<0] = 0.
        data['rt'] = np.abs(data['rt'])

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

    if 'v_switch' in params:
        switch = True
        t_switch = params['t_switch']/dt
        # Hack so that we will always step into a switch
        nn = int(round(t_switch))
    else:
        switch = False

    #create delay
    if 'st' in params:
        start_delay = (uniform.rvs(loc=params['t'], scale=params['st'], size=samples) \
                       - params['st']/2.)
    else:
        start_delay = np.ones(samples)*params['t']

    #create starting_points
    if 'sz' in params:
        starting_points = (uniform.rvs(loc=params['z'], scale=params['sz'], size=samples) \
                           - params['sz']/2.)*a
    else:
        starting_points = np.ones(samples)*params['z']*a

    rts = np.empty(samples)
    step_size = np.sqrt(dt)*intra_sv
    drifts = []

    for i_sample in range(samples):
        drift = np.array([])
        crossed = False
        iter = 0
        y_0 = starting_points[i_sample]
        # drifting...
        if 'sv' in params and params['sv'] != 0:
            drift_rate = norm.rvs(v, params['sv'])
        else:
            drift_rate = v

        if 'v_switch' in params:
            if 'V_switch' in params and params['V_switch'] != 0:
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
    v = params['v']; V= params['sv']; z = params['z']; Z = params['sz']; t = params['t'];
    T = params['st']; a = params['a']
    return hddm.wfpt.full_pdf(rt,v=v,V=V,a=a,z=z,Z=Z,t=t,
                        T=T,err=1e-4, n_st=2, n_sz=2, use_adaptive=1, simps_err=1e-3)

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
    v = params['v']; V = params['sv']; z = params['z']; Z = params['sz']; t = params['t'];
    T = params['st']; a = params['a']
    return hddm.likelihoods.wfpt.ppf(np.random.rand(samples), args=(v, V, a, z, Z, t, T))

def gen_rand_data(params=None, n_fast_outliers=0, n_slow_outliers=0, **kwargs):
    """Generate simulated RTs with random parameters.

       :Optional:
            params : dict <default=generate randomly>
                Either dictionary mapping param names to values.

                Or dictionary mapping condition name to parameter
                dictionary (see example below).

                If not supplied, takes random values.

            n_fast_outliers : int <default=0>
                How many fast outliers to add (outlier_RT < ter)

            n_slow_outliers : int <default=0>
                How many late outliers to add.

            The rest of the arguments are forwarded to kabuki.generate.gen_rand_data

       :Returns:
            data array with RTs
            parameter values

       :Example:
            # Generate random data set
            >>> data, params = hddm.generate.gen_rand_data({'v':0, 'a':2, 't':.3},
                                                           size=100, subjs=5)

            # Generate 2 conditions
            >>> data, params = hddm.generate.gen_rand_data({'cond1': {'v':0, 'a':2, 't':.3},
                                                            'cond2': {'v':1, 'a':2, 't':.3}})

       :Notes:
            Wrapper function for kabuki.generate.gen_rand_data. See
            the help doc of that function for more options.

    """

    if params is None:
        params = gen_rand_params()

    from numpy import inf

    # set valid param ranges
    bounds = {'a': (0, inf),
              'z': (0, 1),
              't': (0, inf),
              'st': (0, inf),
              'sv': (0, inf),
              'sz': (0, 1)
    }

    if 'share_noise' not in kwargs:
        kwargs['share_noise'] = set(['a','v','t','st','sz','sv','z'])

    # Create RT data
    data, subj_params = kabuki.generate.gen_rand_data(gen_rts, params,
                                                      check_valid_func=hddm.utils.check_params_valid,
                                                      bounds=bounds, **kwargs)
    #add outliers
    seed = kwargs.get('seed', None)
    data = add_outliers(data, n_fast=n_fast_outliers, n_slow=n_slow_outliers, seed=seed)

    return data, subj_params

def add_outliers(data, n_fast, n_slow, seed=None):
    """add outliers to data. outliers are distrbuted randomly across condition.
    Input:
        data - data
        n_fast/n_slow - numberprobability of fast/slow outliers
    """
    data = pd.DataFrame(data)
    n_outliers = n_fast + n_slow
    if n_outliers == 0:
        return data

    if seed is not None:
        np.random.seed(seed)

    #init outliers DataFrame
    idx = np.random.permutation(len(data))[:n_outliers]
    outliers = data.ix[idx].copy()

    #fast outliers
    outliers[:n_fast]['rt'] = np.random.rand(n_fast) * (min(abs(data['rt'])) - 0.1001)  + 0.1001

    #slow outliers
    outliers[n_fast:]['rt'] = np.random.rand(n_slow) * 2 + max(abs(data['rt']))
    outliers['response'] = np.random.randint(0,2,n_outliers)

    #combine data with outliers
    data = pd.concat((data, outliers), ignore_index=True)
    return data
