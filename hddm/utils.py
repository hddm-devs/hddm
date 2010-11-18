from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

import hddm

def flip_errors(data):
    """Flip sign for lower boundary responses."""
    # Check if data is already flipped
    if np.any(data['rt'] < 0):
        return data
    
    # Copy data
    data = np.array(data)
    # Flip sign for lower boundary responses
    idx = data['response'] == 0
    data['rt'][idx] = -data['rt'][idx]
    
    return data

def scale(x, max=None, min=None):
    x = x.astype(np.float)
    if max is None:
        max = np.max(x)
    if min is None:
        min = np.min(x)
    return (x-min)/(max-min)

def scale_multi(a1, a2):
    """Scale two arrays to be in range [0,1].
    """
    # Scale appropriately
    if np.max(a1) > np.max(a2):
        a1_scaled = scale(a1)
        a2_scaled = scale(a2, max=np.max(a1), min=np.min(a1))
    else:
        a2_scaled = scale(a2)
        a1_scaled = scale(a1, max=np.max(a2), min=np.min(a2))

    return (a1_scaled, a2_scaled)

def difference_prior(delta):
    # See Wagenmakers et al 2010, equation 14
    if type(delta) is int:
        if delta<=0:
            return 1+delta
        else:
            return 1-delta
    else:
        out = copy(delta)
        out[delta <= 0] = 1+delta[delta <= 0]
        out[delta > 0] = 1-delta[delta > 0]
        return out

def interpolate_trace(x, trace, range=(-1,1), bins=100):
    import scipy.interpolate

    x_histo = np.linspace(range[0], range[1], bins)
    histo = np.histogram(trace, bins=bins, range=range, normed=True)[0]
    interp = scipy.interpolate.InterpolatedUnivariateSpline(x_histo, histo)(x)

    return interp

def uniform(x, lower, upper):
    y = np.ones(x.shape, dtype=np.float)/(upper-lower)
    #y[x<lower] = 0.
    #y[x>upper] = 0.

    return y

def savage_dickey(post_trace, range=(-1,1), bins=100, plot=False, title=None, savefig=None, prior_trace=None, prior_y=None, plot_prior=True, label=None):
    # Calculate Savage-Dickey density ratio test, see Wagenmakers et al 2010
    # Estimate density of posterior
    # Calculate normalized histogram (density)
    x = np.linspace(range[0], range[1], bins)
    if prior_trace is not None:
        prior0 = interpolate_trace(0, prior_trace, range=range, bins=bins)
        prior = interpolate_trace(x, prior_trace, range=range, bins=bins)
    elif prior_y is not None:
        import scipy.interpolate
        prior0 = scipy.interpolate.InterpolatedUnivariateSpline(x, prior_y)(0)
        prior = prior_y
    else:
        assert ValueError, "Supply either prior_trace or prior_y keyword arguments"
        
    posterior0 = interpolate_trace(0, post_trace, range=range, bins=bins)
    posterior = interpolate_trace(x, post_trace, range=range, bins=bins)

    # Calculate Savage-Dickey density ratio at x=0
    sav_dick = posterior0 / prior0

    if plot:
        if label is None:
            label='posterior'
        plt.plot(x, posterior, label=label, lw=2.)
        if plot_prior:
            plt.plot(x, prior, label='prior', lw=2.)
        plt.axvline(x=0, lw=1., color='k')
        plt.ylim(ymin=0)
        if title:
            plt.title(title)
        if savefig:
            plt.savefig('plots/'+savefig+'.png')

    return sav_dick #, prior, posterior, prior0, posterior0
    

def call_mcmc((model_class, data, dbname, rnd, kwargs)):
    # Randomize seed
    np.random.seed(int(rnd))

    model = model_class(data, **kwargs)
    model.mcmc(dbname=dbname)
    model.mcmc_model.db.close()

def create_tag_names(tag, chains=None):
    import multiprocessing
    if chains is None:
        chains = multiprocessing.cpu_count()
    tag_names = []
    # Create copies of data and the corresponding db names
    for chain in range(chains):
        tag_names.append("db/mcmc%s%i.pickle"% (tag,chain))

    return tag_names

def load_parallel_chains(model_class, data, tag, kwargs, chains=None, test_convergance=True, combine=True):
    tag_names = create_tag_names(tag, chains=chains)
    models = []
    for tag_name in tag_names:
        model = model_class(data, **kwargs)
        model.mcmc_load_from_db(tag_name)
        models.append(model)

    if test_convergance:
        Rhat = test_chain_convergance(models)
        print Rhat
    
    if combine:
        m = combine_chains(models, model_class, data, kwargs)
        return m
    
    return models

def combine_chains(models, model_class, data, kwargs):
    """Combine multiple model runs into one final model (make sure that chains converged)."""
    # Create model that will contain the other traces
    m = copy(models[0])

    # Loop through models and combine chains
    for model in models[1:]:
        m._set_traces(m.group_params, mcmc_model=model.mcmc_model, add=True)
        m._set_traces(m.group_params_tau, mcmc_model=model.mcmc_model, add=True)
        m._set_traces(m.subj_params, mcmc_model=model.mcmc_model, add=True)

    return m
        
def run_parallel_chains(model_class, data, tag, load=False, cpus=None, chains=None, **kwargs):
    import multiprocessing
    if cpus is None:
        cpus = multiprocessing.cpu_count()

    tag_names = create_tag_names(tag, chains=chains)
    # Parallel call
    if not load:
        rnds = np.random.rand(len(tag_names))*10000
        pool = multiprocessing.Pool(processes=cpus)
        pool.map(call_mcmc, [(model_class, data, tag_name, rnd, kwargs) for tag_name,rnd in zip(tag_names, rnds)])

    models = load_parallel_chains(model_class, data, tag_names, kwargs)

    return models

def R_hat(samples):
    n, num_chains = samples.shape # n=num_samples
    chain_means = np.mean(samples, axis=1)
    # Calculate between-sequence variance
    between_var = n * np.var(chain_means, ddof=1)

    chain_var = np.var(samples, axis=1, ddof=1)
    within_var = np.mean(chain_var)

    marg_post_var = ((n-1.)/n) * within_var + (1./n) * between_var # 11.2
    R_hat_sqrt = np.sqrt(marg_post_var/within_var)

    return R_hat_sqrt

def test_chain_convergance(models):
    # Calculate R statistic to check for chain convergance (Gelman at al 2004, 11.4)
    params = models[0].group_params
    R_hat_param = {}
    for param_name in params.iterkeys():
        # Calculate mean for each chain
        num_samples = models[0].group_params[param_name].trace().shape[0] # samples
        num_chains = len(models)
        samples = np.empty((num_chains, num_samples))
        for i,model in enumerate(models):
            samples[i,:] = model.group_params[param_name].trace()

        R_hat_param[param_name] = R_hat(samples)

    return R_hat_param

def load_gene_data(exclude_missing=None, exclude_inst_stims=True):
    import numpy.lib.recfunctions as rec
    pos_stims = ('A', 'C', 'E')
    neg_stims = ('B', 'D', 'F')
    fname = 'Gene_tst1_RT.csv'
    data = np.recfromcsv(fname)
    data['subj_idx'] = data['subj_idx']-1 # Offset subj_idx to start at 0

    data['rt'] = data['rt']/1000. # Time in seconds
    # Remove outliers
    data = data[data['rt'] > .25]

    # Exclude subjects for which there is no particular gene.
    if exclude_missing is not None:
        if isinstance(exclude_missing, (tuple, list)):
            for exclude in exclude_missing:
                data = data[data[exclude] != '']
        else:
            data = data[data[exclude_missing] != '']

    # Add convenience columns
    # First stim and second stim
    cond1 = np.empty(data.shape, dtype=np.dtype([('cond1','S1')]))
    cond2 = np.empty(data.shape, dtype=np.dtype([('cond2','S1')]))
    cond_class = np.empty(data.shape, dtype=np.dtype([('conf','S2')]))
    contains_A = np.empty(data.shape, dtype=np.dtype([('contains_A',np.bool)]))
    contains_B = np.empty(data.shape, dtype=np.dtype([('contains_B',np.bool)]))
    include = np.empty(data.shape, dtype=np.bool)
    
    for i,cond in enumerate(data['cond']):
        cond1[i] = cond[0]
        cond2[i] = cond[1]
        # Condition contains A or B, with AB trials excluded
        contains_A[i] = (((cond[0] == 'A') or (cond[1] == 'A')) and not ((cond[0] == 'B') or (cond[1] == 'B')),)
        contains_B[i] = (((cond[0] == 'B') or (cond[1] == 'B')) and not ((cond[0] == 'A') or (cond[1] == 'A')),)

        if cond[0] in pos_stims and cond[1] in pos_stims:
            cond_class[i] = 'WW'
        elif cond[0] in neg_stims and cond[1] in neg_stims:
            cond_class[i] = 'LL'
        else:
            cond_class[i] = 'WL'

        # Exclude instructed stims
        include[i] = cond.find(data['instructed1'][i])
        
    # Append rows to data
    data = rec.append_fields(data, names=['cond1','cond2','conf', 'contains_A', 'contains_B'],
                             data=(cond1,cond2,cond_class,contains_A,contains_B), dtypes=['S1','S1','S2','B1','B1'], usemask=False)

    return data[include]

def rec2csv(data, fname, sep=None):
    """Save record array to fname as csv.
    """
    if sep is None:
        sep = ','
    with open(fname, 'w') as fd:
        # Write header
        fd.write(sep.join(data.dtype.names))
        fd.write('\n')
        # Write data
        for line in data:
            line_str = [str(i) for i in line]
            fd.write(sep.join(line_str))
            fd.write('\n')

def csv2rec(fname):
    return np.lib.io.recfromcsv(fname)

def parse_config_file(fname, mcmc=False, load=False):
    import os.path
    if not os.path.isfile(fname):
        raise ValueError("%s could not be found."%fname)

    import ConfigParser
    
    config = ConfigParser.ConfigParser()
    config.read(fname)
    
    #####################################################
    # Parse config file
    data_fname = config.get('data', 'load')
    if not os.path.exists(data_fname):
        raise IOError, "Data file %s not found."%data_fname

    try:
        save = config.get('data', 'save')
    except ConfigParser.NoOptionError:
        save = False

    data = np.recfromcsv(data_fname)
    
    try:
        model_type = config.get('model', 'type')
    except ConfigParser.NoOptionError:
        model_type = 'simple'

    try:
        is_subj_model = config.getboolean('model', 'is_subj_model')
    except ConfigParser.NoOptionError:
        is_subj_model = True

    try:
        no_bias = config.getboolean('model', 'no_bias')
    except ConfigParser.NoOptionError:
        no_bias = True

    try:
        debug = config.getboolean('model', 'debug')
    except ConfigParser.NoOptionError:
        debug = False

    try:
        dbname = config.get('mcmc', 'dbname')
    except ConfigParser.NoOptionError:
        dbname = None

    if model_type == 'simple' or model_type == 'simple_gpu':
        group_param_names = ['a', 'v', 'z', 't']
    elif model_type == 'full_avg' or model_type == 'full':
        group_param_names = ['a', 'v', 'V', 'z', 'Z', 't', 'T']
    elif model_type == 'lba':
        group_param_names = ['a', 'v', 'z', 't', 'V']
    else:
        raise NotImplementedError('Model type %s not implemented'%model_type)

    # Get depends
    depends = {}
    for param_name in group_param_names:
        try:
            # Multiple depends can be listed (separated by a comma)
            depends[param_name] = config.get('depends', param_name).split(',')
        except ConfigParser.NoOptionError:
            pass

    # MCMC values
    try:
        samples = config.getint('mcmc', 'samples')
    except ConfigParser.NoOptionError:
        samples = 10000
    try:
        burn = config.getint('mcmc', 'burn')
    except ConfigParser.NoOptionError:
        burn = 5000
    try:
        thin = config.getint('mcmc', 'thin')
    except ConfigParser.NoOptionError:
        thin = 3
    try:
        verbose = config.getint('mcmc', 'verbose')
    except ConfigParser.NoOptionError:
        verbose = 0

    print "Creating model..."
    m = hddm.models.Multi(data, model_type=model_type, is_subj_model=is_subj_model, no_bias=no_bias, depends_on=depends, debug=debug)

    if mcmc:
        if not load:
            print "Sampling... (this can take some time)"
            m.mcmc(samples=samples, burn=burn, thin=thin, verbose=verbose, dbname=dbname)
        else:
            m.mcmc_load_from_db(dbname=dbname)

    if save:
        m.save_stats(save)
    else:
        print m.summary()
        
    return m

def posterior_predictive_check(model, data):
    params = copy(model.params_est)
    if model.model_type.startswith('simple'):
        params['sv'] = 0
        params['sz'] = 0
        params['ster'] = 0
    if model.no_bias:
        params['z'] = params['a']/2.
        
    data_sampled = _gen_rts_params(params)

    # Check
    return pm.discrepancy(data_sampled, data, .5)
    
    
def check_geweke(model, assert_=True):
    # Test for convergence using geweke method
    for param in model.group_params.itervalues():
        geweke = np.array(pm.geweke(param))
        if assert_:
            assert (np.any(np.abs(geweke[:,1]) < 2)), 'Chain of %s not properly converged'%param
            return False
        else:
            if np.any(np.abs(geweke[:,1]) > 2):
                print "Chain of %s not properly converged" % param
                return False

    return True

def EZ_subjs(data):
    params = {}
    
    # Estimate EZ group parameters
    v, a, t = EZ_data(data)
    params['v'] = v
    params['a'] = a
    params['t'] = t-.2 if t-.2>0 else .1 # Causes problems otherwise
    params['z'] = a/2.

    
    # Estimate EZ parameters for each subject
    try:
        for subj in np.unique(data['subj_idx']):
            try:
                v, a, t = EZ_data(data[data['subj_idx'] == subj])
                params['v_%i'%subj] = v
                params['a_%i'%subj] = a
                params['t_%i'%subj] = 0 #t-.2 if t-.2>0 else .1
                params['z_%i'%subj] = a/2.
            except ValueError:
                # Subject either had 0%, 50%, or 100% correct, which does not work
                # with easy. But we can deal with that by just not initializing the
                # parameters for that one model.
                params['v_%i'%subj] = None
                params['a_%i'%subj] = None
                params['t_%i'%subj] = None
                params['z_%i'%subj] = None
                
    except ValueError:
        # Data array has no subj_idx -> ignore.
        pass
        
    return params
        
def EZ_param_ranges(data, range_=1.):
    v, a, t = EZ_data(data)
    z = a/2.
    param_ranges = {'a_lower': a-range_,
                    'a_upper': a+range_,
                    'z_lower': z-range_,
                    'z_upper': z+range_,
                    't_lower': t-range_ if (t-range_)>0 else 0.,
                    't_upper': t+range_,
                    'v_lower': v-range_,
                    'v_upper': v+range_}

    return param_ranges

def EZ_data(data, s=1):
    """
    Calculate Wagenmaker's EZ-diffusion statistics on data.

       :Parameters:
       - data : numpy.array
           Data array with reaction time data. Correct RTs
           are positive, incorrect RTs are negative.
       - s : float
           Scaling parameter (default=1)

      :Returns:
      - (v, a, ter) : tuple
          drift-rate, threshold and non-decision time

    :SeeAlso: EZ
    """

    try:
        rt = data['rt']
    except ValueError:
        rt = data

    # Compute statistics over data
    idx_correct = rt > 0
    idx_error = rt < 0
    mrt = np.mean(rt[idx_correct])
    vrt = np.var(rt[idx_correct])
    pc = np.sum(idx_correct) / np.float(rt.shape[0])

    # Calculate EZ estimates.
    return EZ(pc, vrt, mrt, s)

def EZ(pc, vrt, mrt, s=1):
    """
    Calculate Wagenmaker's EZ-diffusion statistics.
    
      :Parameters:
      - pc : float
          probability correct.
      - vrt : float
          variance of response time for correct decisions (only!).
      - mrt : float
          mean response time for correct decisions (only!).
      - s : float
          scaling parameter. Default s=1.

      :Returns:
      - (v, a, ter) : tuple
          drift-rate, threshold and non-decision time
          
    The error RT distribution is assumed identical to the correct RT distrib.

    Edge corrections are required for cases with Pc=0 or Pc=1. (Pc=.5 is OK)

    Assumptions of the EZ-diffusion model:
    * The error RT distribution is identical to the correct RT distrib.
    * z=a/2 -- starting point is equidistant from the response boundaries
    * sv=0 -- across-trial variability in drift rate is negligible
    * sz=0  -- across-trial variability in starting point is negligible
    * st=0  -- across-trial range in nondecision time is negligible

    Reference:
    Wagenmakers, E.-J., van der Maas, H. Li. J., & Grasman, R. (2007).
    An EZ-diffusion model for response time and accuracy.
    Psychonomic Bulletin & Review, 14 (1), 3-22.

    :Example from Wagenmakers et al. (2007):
    >>> EZ(.802, .112, .723, s=.1)
    (0.099938526231301769, 0.13997020267583737, 0.30002997230248141)

    :SeeAlso: EZ_data
    """
    if (pc == 0 or pc == .5 or pc == 1):
        raise ValueError('pc is either 0%, 50% or 100%')
    
    s2 = s**2
    logit_p = np.log(pc/(1-pc))

    # Eq. 7
    x = ((logit_p*(pc**2 * logit_p - pc * logit_p + pc - .5)) / vrt)
    v = np.sign(pc - .5) * s * x**.25
    # Eq 5
    a = (s2 * logit_p) / v

    y = (-v*a)/s2
    # Eq 9
    mdt = (a/(2*v)) * ((1-np.exp(y)) / (1+np.exp(y)))

    # Eq 8
    ter = mrt-mdt

    return (v, a, ter)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
