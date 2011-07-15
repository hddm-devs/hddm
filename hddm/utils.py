from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
from scipy.stats import scoreatpercentile
import sys

import hddm
try:
    from IPython.Debugger import Tracer; debug_here = Tracer()
except:
    pass


def flip_errors(data):
    """Flip sign for lower boundary responses.

        :Arguments:
            data : numpy.recarray
                Input array with at least one column named 'RT' and one named 'response'
        :Returns:
            data : numpy.recarray
                Input array with RTs sign flipped where 'response' == 0

    """
    # Check if data is already flipped
    if np.any(data['rt'] < 0):
        return data
    
    # Copy data
    data = np.array(data)
    # Flip sign for lower boundary response
    idx = data['response'] == 0
    data['rt'][idx] = -data['rt'][idx]
    
    return data

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
    """Create a histogram over a trace and interpolate to get a
    smoothed distribution.

    """
    import scipy.interpolate

    x_histo = np.linspace(range[0], range[1], bins)
    histo = histogram(trace, bins=bins, range=range, density=True)[0]
    interp = scipy.interpolate.InterpolatedUnivariateSpline(x_histo, histo)(x)

    return interp

def savage_dickey(pos, post_trace, range=(-.3,.3), bins=40, prior_trace=None, prior_y=None):
    """Calculate Savage-Dickey density ratio test, see Wagenmakers et
    al. 2010 at http://dx.doi.org/10.1016/j.cogpsych.2009.12.001

    :Arguments:
        pos : float
            position at which to calculate the savage dickey ratio at (i.e. the spec hypothesis you want to test)
        post_trace : numpy.array
            trace of the posterior distribution
    
    :Optional:
         prior_trace : numpy.array
             trace of the prior distribution
         prior_y : numpy.array
             prior density pos
         range : (int,int)
             Range over which to interpolate and plot
         bins : int
             Over how many bins to compute the histogram over
    
    :Note: Supply either prior_trace or prior_y.

    """
    
    x = np.linspace(range[0], range[1], bins)

    if prior_trace is not None:
        # Prior is provided as a trace -> histogram + interpolate
        prior_pos = interpolate_trace(pos, prior_trace, range=range, bins=bins)

    elif prior_y is not None:
        # Prior is provided as a density for each point -> interpolate to retrieve positional density
        import scipy.interpolate
        prior_pos = prior_y #scipy.interpolate.InterpolatedUnivariateSpline(x, prior_y)(pos)
    else:
        assert ValueError, "Supply either prior_trace or prior_y keyword arguments"

    # Histogram and interpolate posterior trace at SD position
    posterior_pos = interpolate_trace(pos, post_trace, range=range, bins=bins)

    # Calculate Savage-Dickey density ratio at pos
    sav_dick = prior_pos / posterior_pos

    return sav_dick

def gen_stats(traces, alpha=0.05, batches=100):
    """Useful helper function to generate stats() on a loaded database
    object.  Pass the db._traces list.

    """
    
    from pymc.utils import hpd, quantiles
    from pymc import batchsd

    stats = {}
    for name, trace_obj in traces.iteritems():
        trace = np.squeeze(np.array(trace_obj(), float))
        stats[name] = {'standard deviation': trace.std(0),
                       'mean': trace.mean(0),
                       '%s%s HPD interval' % (int(100*(1-alpha)),'%'): hpd(trace, alpha),
                       'mc error': batchsd(trace, batches),
                       'quantiles': quantiles(trace)}

    return stats


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

def parse_config_file(fname, mcmc=False, data=None):
    import kabuki
    import os.path
    if not os.path.isfile(fname):
        raise ValueError("%s could not be found."%fname)

    import ConfigParser
    
    config = ConfigParser.ConfigParser()
    config.optionxform = str
    config.read(fname)
    
    #####################################################
    # Parse config file
    if data is not None:
        data_fname = data
    else:
        try:
            data_fname = config.get('model', 'data')
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            print "ERROR: No data file specified. Either provide data file as an argument to hddmfit or in the model specification"
            sys.exit(-1)
    if not os.path.exists(data_fname):
        raise IOError, "Data file %s not found."%data_fname
    
    data = np.recfromcsv(data_fname)
    
    try:
        include = config.get('model', 'include')
    except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
        include = ()

    try:
        is_group_model = config.getboolean('model', 'is_group_model')
    except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
        is_group_model = None

    try:
        bias = config.getboolean('model', 'bias')
    except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
        bias = False

    try:
        db = config.get('mcmc', 'db')
    except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
        db = 'ram'

    try:
        dbname = config.get('mcmc', 'dbname')
    except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
        dbname = None

    # MCMC values
    try:
        samples = config.getint('mcmc', 'samples')
    except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
        samples = 10000
    try:
        burn = config.getint('mcmc', 'burn')
    except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
        burn = 5000
    try:
        thin = config.getint('mcmc', 'thin')
    except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
        thin = 2
    try:
        verbose = config.getint('mcmc', 'verbose')
    except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
        verbose = 0

    try:
        plot_rt_fit = config.getboolean('stats', 'plot_rt_fit')
    except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
        plot_rt_fit = True
        
    try:
        plot_posteriors = config.getboolean('stats', 'plot_posteriors')
    except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
        plot_posteriors = True

    group_params = ['v', 'V', 'a', 'z', 'Z', 't', 'T']
    
    # Get depends
    depends = {}
    for param_name in group_params:
        try:
            # Multiple depends can be listed (separated by a comma)
            depends[param_name] = config.get('depends', param_name).split(',')
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            pass

    print "Creating model..."
    m = hddm.HDDM(data, include=include, bias=bias, is_group_model=is_group_model, depends_on=depends)

    m.mcmc().sample(samples, burn=burn, thin=thin, verbose=verbose)

    print kabuki.analyze.print_stats(m.mc.stats())

    print "logp: %f" % m.mc.logp
    print "DIC: %f" % m.mc.dic

    if plot_rt_fit:
        plot_post_pred(m.nodes)
        
    if plot_posteriors:
        hddm.plot_posteriors(m)
        
    return m

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

    :Arguments:
       data : numpy.array
           Data array with reaction time data. Correct RTs
           are positive, incorrect RTs are negative.
       s : float
           Scaling parameter (default=1)

    :Returns:
      (v, a, ter) : tuple
          drift-rate, threshold and non-decision time

    :See Also: EZ

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
        pc : float
            probability correct.
        vrt : float
            variance of response time for correct decisions (only!).
        mrt : float
            mean response time for correct decisions (only!).
        s : float
            scaling parameter. Default s=1.
    :Returns:
        (v, a, ter) : tuple
             drift-rate, threshold and non-decision time
          
    The error RT distribution is assumed identical to the correct RT distrib.

    Edge corrections are required for cases with Pc=0 or Pc=1. (Pc=.5 is OK)

    :Assumptions:
        * The error RT distribution is identical to the correct RT distrib.
        * z=.5 -- starting point is equidistant from the response boundaries
        * sv=0 -- across-trial variability in drift rate is negligible
        * sz=0  -- across-trial variability in starting point is negligible
        * st=0  -- across-trial range in nondecision time is negligible

    :Reference:
        Wagenmakers, E.-J., van der Maas, H. Li. J., & Grasman, R. (2007).

        An EZ-diffusion model for response time and accuracy.
        Psychonomic Bulletin & Review, 14 (1), 3-22.

    :Example:
        >>> EZ(.802, .112, .723, s=.1)
        (0.099938526231301769, 0.13997020267583737, 0.30002997230248141)

    :See Also: EZ_data
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

def pdf_of_post_pred(traces, pdf=None, args=None, x=None, interval=10):
    """Calculate posterior predictive probability density function.

    :Arguments:
        traces : dict
            A dictionary of traces (e.g. MCMC._dict_container).
        pdf : func
            A pdf to generate the posterior predictive from [default=wfpt].
        args : tuple
            Tuple of arguments to be supplied to the pdf 
            [default=('v', 'V', 'a','z','Z', 't','T')].

    """
    if pdf is None:
        pdf = hddm.likelihoods.wfpt.pdf
        args = ('v', 'V', 'a','z','Z', 't','T')
        
    if x is None:
        x = np.arange(-5,5,0.01)


    trace_len = len(traces['a'])
    p = np.zeros(len(x), dtype=np.float)

    # Add default traces if needed parameter is excluded.
    if not traces.has_key('V'):
        traces['V'] = np.zeros(trace_len)
    if not traces.has_key('T'):
        traces['T'] = np.zeros(trace_len)
    if not traces.has_key('Z'):
        traces['Z'] = np.zeros(trace_len)
    if not traces.has_key('z'):
        traces['z'] = np.ones(trace_len)*.5

    samples = 0
    for i in np.arange(0, trace_len, interval):
        samples += 1
        valued_args = []
        # Construct arguments to be passed to pdf
        for arg in args:
            valued_args.append(traces[arg][i])
        pdf_full = lambda x: pdf(x, *valued_args)
        
        p[:] += map(pdf_full, x)
            
    return p/samples
    
def plot_post_pred(nodes, bins=50, range=(-5.,5.), interval=10, fname=None):
    if type(nodes) is pm.MCMC:
        nodes = nodes._dict_container
        
    x = np.arange(range[0],range[1],0.05)
    # Plot data
    x_data = np.linspace(range[0], range[1], bins)
    
    figure_idx = 0
    for name, node in nodes.iteritems():
        # Find wfpt node
        if not name.startswith('wfpt'):
            continue 

        plt.figure()
        figure_idx += 1
        if type(node) is np.ndarray or type(node) is pm.ArrayContainer: # Group model
            for i, subj_node in enumerate(node):
                data = subj_node.value
                # Walk through nodes and collect traces
                traces = {}
                for parent_name, parent_node in subj_node.parents.iteritems():
                    if type(parent_node) is int or type(parent_node) is float or type(parent_node) is list or type(parent_node) is pm.ListContainer:
                        continue
                    traces[parent_name] = parent_node.trace()

                # Plot that shit ;)
                plt.subplot(3, int(np.ceil(len(node)/3.)), i+1)

                empirical_dens = histogram(data, bins=bins, range=range, density=True)[0]
                plt.plot(x_data, empirical_dens, color='b', lw=2., label='data')
                
                # Plot analytical
                analytical_dens = pdf_of_post_pred(traces, x=x, interval=interval)

                plt.plot(x, analytical_dens, '--', color='g', label='estimate', lw=2.)

                plt.xlim(range)
                plt.title("%s (n=%d). idx: %i" %(name, len(data), i))

        else:
            data = node.value
            # Walk through nodes and collect traces
            traces = {}
            for parent_name, parent_node in node.parents.iteritems():
                if np.isscalar(parent_node) or type(parent_node) is list or type(parent_node) is pm.ListContainer:
                        continue
                traces[parent_name] = parent_node.trace()
            
            empirical_dens = histogram(data, bins=bins, range=range, density=True)[0]
            plt.plot(x_data, empirical_dens, color='b', lw=2., label='data')

            # Plot analytical
            analytical_dens = pdf_of_post_pred(traces, x=x, interval=interval)

            plt.plot(x, analytical_dens, '--', color='g', label='estimate', lw=2.)

            plt.xlim(range)
            plt.title("%s (n=%d)" %(name, len(data)))
            plt.legend()
            
        if fname is not None:
            plt.savefig('%s%i.png'(fname, figure_idx))

    plt.show()

def remove_outliers(nodes, depends_on=None, cutoff_prob=.4):
    raise NotImplemented, "Coming in v0.2."
    if depends_on is None:
        depends_on = []

    if type(nodes) is pm.MCMC:
        nodes = nodes._dict_container
    
    for name, node in nodes.iteritems():
        # Select x nodes that code for contaminant
        if not name.startswith('x'):
            continue
        
        # Select appropriate wfpt node
        wfpt_node = nodes[name.replace('x', 'wfpt')]

        if type(node) is np.ndarray or type(node) is pm.ArrayContainer: # Group model
            for cont, wfpt in zip(node, wfpt_node):
                data = wfpt.value
                contaminant_prob = np.mean(subj_node.trace(), axis=0)
        else:
            raise NotImplemented, "TODO, use group model."
        
def hddm_parents_trace(node,idx):
    """Return the parents' value of an wfpt node in index 'idx' (the
    function is used by ppd_test)

    """
    params = {}
    for name in ['a','v','t']:
        params[name] = node.parents[name].trace()[idx]

    if node.parents['z'] != .5: # bias model
        params['z'] = node.parents['z'].trace()[idx]
    else:
        params['z'] = 0.5
    
    for name in ['V','Z','T']:
        if node.parents.has_key(name):
            if node.parents[name] != 0:
                params[name] = node.parents[name].trace()[idx]
            else:
                params[name] = 0
        else:
            params[name] = 0

    return params
            
def _gen_statistics():
    """generate different statistical tests from ppd_test."""
    statistics = []
    
    ##accuracy test
    test = {}
    test['name'] = 'acc'
    test['func'] = lambda rts: sum(rts > 0)/len(rts)
    statistics.append(test)
    
    ##quantile statistics of absolute response time
    quantiles = [10, 30, 50, 70, 90]
    for q in quantiles:
        test = {}
        test['name'] = 'q%d' % q
        test['func'] = lambda rts,q=q: scoreatpercentile(np.abs(rts), q)
        statistics.append(test)
        
    return statistics

def ppd_test(nodes, n_times = 1000, confidence = 95, stats = None, plot_all = False, verbose = 1):
    """
    Test statistics over the posterior predictive distibution.

    :Arguments:
        nodes : set or MCMC object
            set of nodes / the mc model
        n_times : int 
            number of samples to take out of the trace
        confidence : int
            confidence interval
        stats : set
            a set of statistics to check over the sampled data. if stats is None thedefault set of statistics is created
        plot_all : bool
            should all result be ploted
    """
    if type(nodes) is pm.MCMC:
        nodes = nodes._dict_container
    
    #get statistics    
    if stats == None:
        stats  = _gen_statistics()
    else:
        stats  = _gen_statistics() + stats

    conf_lb = ((100 - confidence)/ 2.)
    conf_ub = (100 - conf_lb)
    
    # get statistics from simulated data
    for name, node in nodes.iteritems():

        # Find wfpt node
        if not name.startswith('wfpt'):
            continue

        if verbose>0:
            print "computing stats for %s" % name
        
        len_trace = len(node.parents['a'].trace())
        thin = max(int(len_trace // n_times), 1)
        n_times = int(len_trace // thin)
        res = np.zeros((len(stats),n_times))
        obs = np.zeros(len(stats))
        #simulate data and compute stats
        for i in xrange(n_times):
            idx = i*thin
            if verbose > 1 and ((i+1) % 100)==0:
                print "created samples for %d params" % (i+1)
            sys.stdout.flush()
            params = hddm_parents_trace(node, idx)
            samples = hddm.generate.gen_rts(params, len(node.value), dt=1e-3,method='cdf')
            for i_stat in xrange(len(stats)):
                res[i_stat][i] = stats[i_stat]['func'](samples)
        

        #compute stats of oberved data
        for i_stat in range(len(stats)):
            obs[i_stat] = stats[i_stat]['func'](node.value)
        
        #compute quantile statistic and plot it if needed
        for i_stat in range(len(stats)):
            plot_this = False
            p = sum(res[i_stat] < obs[i_stat])*1./len(res[i_stat]) * 100
            if (p < conf_lb) or (p>conf_ub):
                print "*!*!* %s :: %s %.1f" % (name,stats[i_stat]['name'], p )
                plot_this = True 
            #plot that shit
            if plot_this or plot_all:
                pm.Matplot.gof_plot(res[i_stat], obs[i_stat], nbins=30, name=name, verbose=0)
                plt.title('%s : %.1f' % (stats[i_stat]['name'], p))
        
        plt.show()                        

def cont_report(nodes, cont_threshold = 0.5, plot= True):
    """create conaminate report."""
    if type(nodes) is pm.MCMC:
        nodes = nodes._dict_container
        
    cont_keys = [z for z in nodes.keys() if z.startswith('x')]
    
    # loop over cont nodes
    n_cont = 0
    rts = np.empty(0)
    for key in cont_keys:
        print "*********************"
        print "looking at %s" % key
        node = nodes[key]
        m = np.mean(node.trace(),0)
        #look for outliers with high probabilty
        idx = np.where(m > cont_threshold)[0]
        n_cont += len(idx)
        if idx.size > 0:
            print "found %d outliers in %s" % (len(idx), key)            
            wfpt = [z for z in nodes[key].children if z.__name__.startswith('wfpt')][0]
            for i_cont in range(len(idx)):
                print "rt: %8.5f prob: %.2f" % (wfpt.value[idx[i_cont]], m[idx[i_cont]])
            rts = np.concatenate((rts, wfpt.value[idx]))
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
    if plot:
        plt.show()
        
    print "!!!!!**** There were %d outliers in the data ****!!!!!" % n_cont
    return rts

def plot_posteriors(model):                 
    """Generate posterior plots for each parameter.

    This is a wrapper for pymc.Matplot.plot()
    """
    pm.Matplot.plot(model.mc)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
