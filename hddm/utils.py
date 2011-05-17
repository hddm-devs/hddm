from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm

import hddm
try:
    from IPython.Debugger import Tracer; debug_here = Tracer()
except:
    pass


def flip_errors(data):
    """Flip sign for lower boundary responses."""
    # Check if data is already flipped
    if np.any(data['rt'] < 0):
        return data
    
    # Copy data
    data = np.array(data)
    # Flip sign for lower boundary response
    idx = data['response'] == 0
    data['rt'][idx] = -data['rt'][idx]
    
    return data

def effect(base, effects):
    first_order = np.sum(effects)

def return_fixed(value=.5):
    return value


def set_proposal_sd(mc, tau=.1):
    for var in mc.variables:
        if var.__name__.endswith('tau'):
            # Change proposal SD
            mc.use_step_method(pm.Metropolis, var, proposal_sd = tau)

    return
    
def histogram(a, bins=10, range=None, normed=False, weights=None, density=None):
    """
    Compute the histogram of a set of data.

    Parameters
    ----------
    a : array_like
        Input data. The histogram is computed over the flattened array.
    bins : int or sequence of scalars, optional
        If `bins` is an int, it defines the number of equal-width
        bins in the given range (10, by default). If `bins` is a sequence,
        it defines the bin edges, including the rightmost edge, allowing
        for non-uniform bin widths.
    range : (float, float), optional
        The lower and upper range of the bins.  If not provided, range
        is simply ``(a.min(), a.max())``.  Values outside the range are
        ignored.
    normed : bool, optional
        This keyword is deprecated in Numpy 1.6 due to confusing/buggy
        behavior. It will be removed in Numpy 2.0. Use the density keyword
        instead.
        If False, the result will contain the number of samples
        in each bin.  If True, the result is the value of the
        probability *density* function at the bin, normalized such that
        the *integral* over the range is 1. Note that this latter behavior is
        known to be buggy with unequal bin widths; use `density` instead.
    weights : array_like, optional
        An array of weights, of the same shape as `a`.  Each value in `a`
        only contributes its associated weight towards the bin count
        (instead of 1).  If `normed` is True, the weights are normalized,
        so that the integral of the density over the range remains 1
    density : bool, optional
        If False, the result will contain the number of samples
        in each bin.  If True, the result is the value of the
        probability *density* function at the bin, normalized such that
        the *integral* over the range is 1. Note that the sum of the
        histogram values will not be equal to 1 unless bins of unity
        width are chosen; it is not a probability *mass* function.
        Overrides the `normed` keyword if given.

    Returns
    -------
    hist : array
        The values of the histogram. See `normed` and `weights` for a
        description of the possible semantics.
    bin_edges : array of dtype float
        Return the bin edges ``(length(hist)+1)``.


    See Also
    --------
    histogramdd, bincount, searchsorted, digitize

    Notes
    -----
    All but the last (righthand-most) bin is half-open.  In other words, if
    `bins` is::

      [1, 2, 3, 4]

    then the first bin is ``[1, 2)`` (including 1, but excluding 2) and the
    second ``[2, 3)``.  The last bin, however, is ``[3, 4]``, which *includes*
    4.

    Examples
    --------
    >>> np.histogram([1, 2, 1], bins=[0, 1, 2, 3])
    (array([0, 2, 1]), array([0, 1, 2, 3]))
    >>> np.histogram(np.arange(4), bins=np.arange(5), density=True)
    (array([ 0.25,  0.25,  0.25,  0.25]), array([0, 1, 2, 3, 4]))
    >>> np.histogram([[1, 2, 1], [1, 0, 1]], bins=[0,1,2,3])
    (array([1, 4, 1]), array([0, 1, 2, 3]))

    >>> a = np.arange(5)
    >>> hist, bin_edges = np.histogram(a, density=True)
    >>> hist
    array([ 0.5,  0. ,  0.5,  0. ,  0. ,  0.5,  0. ,  0.5,  0. ,  0.5])
    >>> hist.sum()
    2.4999999999999996
    >>> np.sum(hist*np.diff(bin_edges))
    1.0

    """

    a = np.asarray(a)
    if weights is not None:
        weights = asarray(weights)
        if np.any(weights.shape != a.shape):
            raise ValueError(
                    'weights should have the same shape as a.')
        weights = weights.ravel()
    a =  a.ravel()

    if (range is not None):
        mn, mx = range
        if (mn > mx):
            raise AttributeError(
                'max must be larger than min in range parameter.')

    if not np.iterable(bins):
        if np.isscalar(bins) and bins < 1:
            raise ValueError("`bins` should be a positive integer.")
        if range is None:
            if a.size == 0:
                # handle empty arrays. Can't determine range, so use 0-1.
                range = (0, 1)
            else:
                range = (a.min(), a.max())
        mn, mx = [mi+0.0 for mi in range]
        if mn == mx:
            mn -= 0.5
            mx += 0.5
        bins = np.linspace(mn, mx, bins+1, endpoint=True)
    else:
        bins = np.asarray(bins)
        if (np.diff(bins) < 0).any():
            raise AttributeError(
                    'bins must increase monotonically.')

    # Histogram is an integer or a float array depending on the weights.
    if weights is None:
        ntype = int
    else:
        ntype = weights.dtype
    n = np.zeros(bins.shape, ntype)

    block = 65536
    if weights is None:
        for i in np.arange(0, len(a), block):
            sa = np.sort(a[i:i+block])
            n += np.r_[sa.searchsorted(bins[:-1], 'left'), \
                sa.searchsorted(bins[-1], 'right')]
    else:
        zero = np.array(0, dtype=ntype)
        for i in np.arange(0, len(a), block):
            tmp_a = a[i:i+block]
            tmp_w = weights[i:i+block]
            sorting_index = np.argsort(tmp_a)
            sa = tmp_a[sorting_index]
            sw = tmp_w[sorting_index]
            cw = np.concatenate(([zero,], sw.cumsum()))
            bin_index = np.r_[sa.searchsorted(bins[:-1], 'left'), \
                sa.searchsorted(bins[-1], 'right')]
            n += cw[bin_index]

    n = np.diff(n)

    if density is not None:
        if density:
            db = np.array(np.diff(bins), float)
            return n/db/n.sum(), bins
        else:
            return n, bins
    else:
        # deprecated, buggy behavior. Remove for Numpy 2.0
        if normed:
            db = np.array(np.diff(bins), float)
            return n/(n*db).sum(), bins
        else:
            return n, bins


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

def uniform(x, lower, upper):
    y = np.ones(x.shape, dtype=np.float)/(upper-lower)
    #y[x<lower] = 0.
    #y[x>upper] = 0.

    return y


def interpolate_trace(x, trace, range=(-1,1), bins=100):
    """Create a histogram over a trace and interpolate to get a smoothed distribution."""
    import scipy.interpolate

    x_histo = np.linspace(range[0], range[1], bins)
    histo = np.histogram(trace, bins=bins, range=range, normed=True)[0]
    interp = scipy.interpolate.InterpolatedUnivariateSpline(x_histo, histo)(x)

    return interp

def savage_dickey(pos, post_trace, range=(-1,1), bins=100, prior_trace=None, prior_y=None):
    """Calculate Savage-Dickey density ratio test, see Wagenmakers et
    al. 2010 at http://dx.doi.org/10.1016/j.cogpsych.2009.12.001

    Arguments:
    **********
    pos<float>: position at which to calculate the savage dickey ratio at (i.e. the specific hypothesis you want to test)
    post_trace<numpy.array>: trace of the posterior distribution

    Keyword arguments:
    ******************
    prior_trace<numpy.array>: trace of the prior distribution
    prior_y<numpy.array>: prior density at each point (must match range and bins)
    range<(int,int)>: Range over which to interpolate and plot
    bins<int>: Over how many bins to compute the histogram over

    IMPORTANT: Supply either prior_trace or prior_y.
    """
    x = np.linspace(range[0], range[1], bins)

    if prior_trace is not None:
        # Prior is provided as a trace -> histogram + interpolate
        prior_pos = interpolate_trace(pos, prior_trace, range=range, bins=bins)

    elif prior_y is not None:
        # Prior is provided as a density for each point -> interpolate to retrieve positional density
        import scipy.interpolate
        prior_pos = scipy.interpolate.InterpolatedUnivariateSpline(x, prior_y)(pos)
    else:
        assert ValueError, "Supply either prior_trace or prior_y keyword arguments"

    # Histogram and interpolate posterior trace at SD position
    posterior_pos = interpolate_trace(pos, post_trace, range=range, bins=bins)

    # Calculate Savage-Dickey density ratio at pos
    sav_dick = posterior_pos / prior_pos

    return sav_dick

def plot_savage_dickey(range=(-1,1), bins=100):
    x = np.linspace(range[0], range[1], bins)
    label='posterior'
            
    # Histogram and interpolate posterior trace
    posterior = interpolate_trace(x, post_trace, range=range, bins=bins)

    plt.plot(x, posterior, label=label, lw=2.)
    if plot_prior:
        if prior_trace is not None:
            # Histogram and interpolate prior trace
            prior_y = interpolate_trace(x, prior_trace, range=range, bins=bins)
        plt.plot(x, prior_y, label='prior', lw=2.)
    plt.axvline(x=0, lw=1., color='k')
    plt.ylim(ymin=0)

def call_mcmc((model_class, data, dbname, rnd, kwargs)):
    # Randomize seed
    np.random.seed(int(rnd))

    model = model_class(data, **kwargs)
    model.mcmc(dbname=dbname)
    model.mcmc_model.db.close()


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
                             data=(cond1,cond2,cond_class,contains_A,contains_B),
                             dtypes=['S1','S1','S2','B1','B1'], usemask=False)

    return data[include]

def save_csv(data, fname, sep=None):
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

def load_csv(fname):
    return np.recfromcsv(fname)

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
    elif model_type == 'full_mc' or model_type == 'full':
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

    try:
        plot_rt_fit = config.getboolean('stats', 'plot_rt_fit')
    except ConfigParser.NoOptionError, ConfigParser.NoSectionError:
        plot_rt_fit = False
        
    try:
        plot_posteriors = config.getboolean('stats', 'plot_posteriors')
    except ConfigParser.NoOptionError, ConfigParser.NoSectionError:
        plot_posteriors = False

    
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

    if plot_rt_fit:
        m.plot_rt_fit()
        
    if plot_posteriors:
        m.plot_posteriors
        
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
    params['z'] = .5

    
    # Estimate EZ parameters for each subject
    try:
        for subj in np.unique(data['subj_idx']):
            try:
                v, a, t = EZ_data(data[data['subj_idx'] == subj])
                params['v_%i'%subj] = v
                params['a_%i'%subj] = a
                params['t_%i'%subj] = 0 #t-.2 if t-.2>0 else .1
                params['z_%i'%subj] = .5
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
    z = .5
    param_ranges = {'a_lower': a-range_,
                    'a_upper': a+range_,
                    'z_lower': np.min(z-range_, 0),
                    'z_upper': np.max(z+range_, 1),
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
    * z=.5 -- starting point is equidistant from the response boundaries
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

def post_pred_simple(v, a, t, z=None, x=None):
    """Posterior predictive likelihood."""
    trace_len = len(a)

    if z is None:
        z = np.ones(a.shape)*.5

    if x is None:
        x = np.arange(-5,5,0.01)

    p = np.zeros(len(x), dtype=np.float)

    for i in range(trace_len):
        p[:] += hddm.wfpt.pdf_array(x, v[i], a[i], z[i], t[i], 1e-4)
    
    return p/trace_len

def plot_post_pred(nodes, bins=50, range=(-5.,5.)):
    x = np.arange(range[0],range[1],0.01)
    
    for name, node in nodes.iteritems():
        # Find wfpt node
        if not name.startswith('wfpt'):
            continue 

        plt.figure()
        if type(node) is np.ndarray: # Group model
            data = np.concatenate([subj_node.value for subj_node in node])
            # Walk through nodes up to the root node
            a = node[0].parents['a'].parents['mu'].trace()
            v = node[0].parents['v'].parents['mu'].trace()
            t = node[0].parents['t'].parents['mu'].trace()
            if node[0].parents['z'] != .5: # bias model
                z = node[0].parents['z'].parents['mu'].trace()
            else:
                z = None
        else:
            data = node.value
            # Walk through nodes up to the root node
            a = node.parents['a'].trace()
            v = node.parents['v'].trace()
            t = node.parents['t'].trace()
            if node.parents['z'] != .5: # bias model
                z = node.parents['z'].trace()
            else:
                z = None
            
            
        # Plot data
        x_data = np.linspace(range[0], range[1], bins)
        empirical_dens = histogram(data, bins=bins, range=range, density=True)[0]
        plt.plot(x_data, empirical_dens, color='b', lw=2., label='data')
        
        # Plot analytical
        analytical_dens = post_pred_simple(v, a, t, z=z, x=x)

        plt.plot(x, analytical_dens, '--', color='g', label='estimate', lw=2.)

        plt.xlim(range)
        plt.title("%s (n=%d)" %(name, len(data)))
        plt.legend()

    plt.show()

def plot_posteriors(model):
    """Generate posterior plots for each parameter.

    This is a wrapper for pymc.Matplot.plot()
    """
    pm.Matplot.plot(model.mcmc_model)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
