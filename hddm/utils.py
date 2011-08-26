from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
from scipy.stats import scoreatpercentile
import sys
import table_print
try:
    from termcolor import colored
except ImportError:
    colored = lambda x:x
from numpy import array, zeros, empty, ones


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

def parse_config_file(fname, map=True, mcmc=False, data=None):
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

    model_name = os.path.splitext(data_fname)[0]

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

    if map:
        print "Finding good initial values..."
        m.map()
        
    m.mcmc().sample(samples, burn=burn, thin=thin, verbose=verbose)

    print kabuki.analyze.print_stats(m.mc.stats())

    print "logp: %f" % m.mc.logp
    print "DIC: %f" % m.mc.dic

    if plot_rt_fit:
        print "Plotting posterior predictive to %s..." % (model_name+'.png')
        plot_post_pred(m, fname=model_name, show=False)
        
    if plot_posteriors:
        hddm.plot_posteriors(m)
        
    return m

def EZ_subjs(data):
    params = {}
    
    # Estimate EZ group parameters
    v, a, t = EZ_data(data)
    params['v'] = v
    params['a'] = a
    params['t'] = t
    params['z'] = .5
    
    # Estimate EZ parameters for each subject
    try:
        for subj in np.unique(data['subj_idx']):
            try:
                v, a, t = EZ_data(data[data['subj_idx'] == subj])
                params['v_%i'%subj] = v
                params['a_%i'%subj] = a
                params['t_%i'%subj] = t #t-.2 if t-.2>0 else .1
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
        raise ValueError('Probability correct is either 0%, 50% or 100%')
    
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

def pdf_of_post_pred(traces, pdf=None, args=None, x=None, samples=30, use_mean=False):
    """Calculate posterior predictive probability density function.

    :Arguments:
        traces : dict
            A dictionary of traces (e.g. MCMC._dict_container).
        pdf : func
            A pdf to generate the posterior predictive from [default=wfpt].
        args : tuple
            Tuple of arguments to be supplied to the pdf 
            [default=('v', 'V', 'a','z','Z', 't','T')].
        use_mean : bool
            Whether to use the mean of or samples from the trace.

    """
    if pdf is None:
        pdf = hddm.likelihoods.wfpt.pdf
        args = ('v', 'V', 'a','z','Z', 't','T')
        
    if x is None:
        x = np.arange(-5,5,0.1)

    trace_len = len(traces['a'])
    p = np.zeros(len(x), dtype=np.float)

    # Add default traces if needed parameter is excluded.
    # This makes it easier to always make the same call to pdf below
    if not traces.has_key('V'):
        traces['V'] = np.zeros(trace_len)
    if not traces.has_key('T'):
        traces['T'] = np.zeros(trace_len)
    if not traces.has_key('Z'):
        traces['Z'] = np.zeros(trace_len)
    if not traces.has_key('z'):
        traces['z'] = np.ones(trace_len)*.5

    if use_mean:
        valued_args = []
        # Construct arguments from traces to be passed to pdf
        for arg in args:
            valued_args.append(np.mean(traces[arg][:]))
        dens = pdf(x, *valued_args)

    else:
        for i in np.round(np.linspace(0, trace_len-1, samples)):
            valued_args = []
            # Construct arguments from traces to be passed to pdf
            for arg in args:
                valued_args.append(traces[arg][i])
            pdf_full = lambda x: pdf(x, *valued_args)
        
            p[:] += map(pdf_full, x)
        dens = p/samples
        
    return dens

def plot_post_pred(model, bins=50, interval=(-5.,5.), n_rows = 3, samples=20, fname=None, show=True, use_mean=True):
    """
    plot posterior predective distribution
    
    :Arguments:
        model : HDDM object
             hddm model

    :Optional:
        bins : int
             number of bins in the histogram of the data
        interval : (int, int)
             a tuple for the time interval which will be presented
        n_rows : int
             number of rows in each figure 
        fname : str
             the file name which the images will be saved to 
        use_mean : bool
            Whether to use the mean of or samples from the trace.
        show : bool
             show the plots
    """
        
    x = np.arange(interval[0],interval[1],0.05)
    # Plot data
    x_data = np.linspace(interval[0], interval[1], bins)
    
    figure_idx = 0
    wfpt = model.params_dict['wfpt'].subj_nodes
    for (cond, nodes) in wfpt.iteritems():
        plt.figure()
        figure_idx += 1
        #group model
        if model.is_group_model:
            n_subjs = model._num_subjs
            for i, subj_node in enumerate(nodes):
                data = subj_node.value
                # Walk through nodes and collect traces
                traces = {}
                for parent_name, parent_node in subj_node.parents.iteritems():
                    if np.isscalar(parent_node) or type(parent_node) is list or type(parent_node) is pm.ListContainer:
                        continue
                    traces[parent_name] = parent_node.trace()

                # Plot that shit ;)
                plt.subplot(n_rows, int(np.ceil(n_subjs/n_rows)), i+1)

                empirical_dens = histogram(data, bins=bins, range=interval, density=True)[0]
                plt.plot(x_data, empirical_dens, color='b', lw=2., label='data')
                
                # Plot analytical
                analytical_dens = pdf_of_post_pred(traces, x=x, samples=samples, use_mean=use_mean)

                plt.plot(x, analytical_dens, '--', color='g', label='estimate', lw=2.)
 
                plt.xlim(interval)
                plt.title("subj %i. (n=%d)" %(model._subjs[i], len(data)))
            
            plt.suptitle(cond)

        else:
            node = nodes
            data = node.value
            # Walk through nodes and collect traces
            traces = {}
            for parent_name, parent_node in node.parents.iteritems():
                if np.isscalar(parent_node) or type(parent_node) is list or type(parent_node) is pm.ListContainer:
                    continue
                traces[parent_name] = parent_node.trace()
            
            empirical_dens = histogram(data, bins=bins, range=interval, density=True)[0]
            plt.plot(x_data, empirical_dens, color='b', lw=2., label='data')

            # Plot analytical
            analytical_dens = pdf_of_post_pred(traces, x=x, samples=samples)

            plt.plot(x, analytical_dens, '--', color='g', label='estimate', lw=2.)

            plt.xlim(interval)
            plt.title("%s (n=%d)" %(cond, len(data)))
            plt.legend()
            
        if fname is not None:
            plt.savefig('%s%i.png'%(fname, figure_idx))

    if show:
        plt.show()
        
def hddm_parents_trace(model, obs_node, idx):
    """Return the parents' value of an wfpt node in index 'idx' (the
    function is used by ppd_test)
    """
    model.params_include.keys()
    params = {'a':0, 'v': 0, 't':0, 'z': 0.5, 'Z': 0 , 'T': 0 , 'V': 0}
    if not np.isscalar(idx):
        for (key, value) in params.iteritems():
            params[key] = np.ones(len(idx))*value
    #example for local_name:  a,v,t,z....
    #example for parent_full_name: v(['cond1'])3
    for local_name in model.params_include.keys():
        if local_name == 'wfpt':
            continue
        
        parent_full_name = obs_node.parents[local_name].__name__
        params[local_name] = model.mc.db.trace(parent_full_name)[idx]

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

def ppd_test(hm, n_samples = 1000, confidence = 95, plot_verbose = 0, verbose = 1,
             table_width = 10):
    """
    Test statistics over the posterior predictive distibution.

    :Arguments:
        hm : HDDM model
        n_samples : int 
            number of samples to use for the ppd test
        confidence : int
            confidence interval
        stats : set
            a set of statistics to check over the sampled data. if stats is None thedefault set of statistics is created
        plot_verbose : int
            0 - no plots
            1 - plot only the statistics that fall outside of the confidencde interval (default)
            2 - plot everything          
        verbose : verbosity of output. 
    """
    
    #if input is tuple than hm is a group model and we are in a recursion
    if type(hm) == type(()):
        #get conds, nodes, and hm 
        conds = [x[0] for x in hm[1]]
        nodes = [x[1] for x in hm[1]]
        model = hm[0]
    else:
        #break group model to subjects models
        if hm.is_group_model:
            group_res = [None]*hm._num_subjs
            for i in range(hm._num_subjs):
                print "--- Results for subj %d ---" % (hm._subjs[i])
                nodes_tuple = [(item[0], item[1][i]) for item in hm.params_include['wfpt'].subj_nodes.items()]
                group_res[i] = ppd_test((hm, nodes_tuple), n_samples, confidence, plot_verbose, verbose)
            
            print_ppd_test_result_for_group(group_res, table_width)
            return group_res
        
        #run subject model
        else:
            model = hm
            (conds, nodes) = model.params_include['wfpt'].subj_nodes.items()
    
    #get statistics    
    stats  = _gen_statistics()
    
    conf_lb = ((100 - confidence)/ 2.)
    conf_ub = (100 - conf_lb)
    
    subj_res = {}
    
    # get statistics from simulated data
    for i_node in xrange(len(nodes)):
        node = nodes[i_node]
        cond = conds[i_node]
        subj_res[cond] = {}
        name = node.__name__

        if verbose>=2:
            print "computing stats for %s" % cond
        
        #when loading from the db, the trace is not assign to the variables
        #so I need to change the way I read from the trace. beheichs 
        len_trace = len(model.mc.db.trace('a')[:])
        thin = max(int(len_trace // n_samples), 1)
        n_samples = int(len_trace // thin)
        res = np.zeros((len(stats),n_samples))
        obs = np.zeros(len(stats))
        all_params = hddm_parents_trace(model, node, np.arange(0, len_trace, thin))
        #simulate data and compute stats
        params = {}
        for i in xrange(n_samples):
            if verbose >= 4 and ((i+1) % 100)==0:
                print "created samples for %d params" % (i+1)
            sys.stdout.flush()
            for key in all_params.iterkeys():
                params[key] = all_params[key][i]
            samples = hddm.generate.gen_rts(params, len(node.value), dt=1e-3,method='cdf')
            for i_stat in xrange(len(stats)):
                res[i_stat][i] = stats[i_stat]['func'](samples)
        

        #compute stats of oberved data
        for i_stat in range(len(stats)):
            obs[i_stat] = stats[i_stat]['func'](node.value)
        
        #compute quantile statistic and plot it if needed
        for i_stat in range(len(stats)):
            stats_name = stats[i_stat]['name']            
            out_conf = False
            p = sum(res[i_stat] < obs[i_stat])*1./len(res[i_stat]) * 100
            if (p < conf_lb) or (p>conf_ub):
                if verbose >= 3:
                    print "*!*!* %s :: %s %.1f" % (name,stats[i_stat]['name'], p )
                out_conf = True 
            #plot that shit
            if (plot_verbose==2) or (out_conf and plot_verbose >= 1):
                pm.Matplot.gof_plot(res[i_stat], obs[i_stat], nbins=30, name=name, verbose=0)
                plt.title('%s : %.1f' % (stats[i_stat]['name'], p))
            #save results to subj_res
            subj_res[cond][stats_name] = {}
            subj_res[cond][stats_name]['obs'] = obs[i_stat]
            subj_res[cond][stats_name]['mean'] = np.mean(res[i_stat])
            subj_res[cond][stats_name]['median'] = np.median(res[i_stat])
            subj_res[cond][stats_name]['lb'] = scoreatpercentile(res[i_stat], conf_lb)
            subj_res[cond][stats_name]['ub'] = scoreatpercentile(res[i_stat], conf_ub)
            subj_res[cond][stats_name]['p'] = p
            subj_res[cond][stats_name]['out_conf'] = out_conf

    plt.show()
    if verbose >= 1:
        print_ppd_test_result_for_subject(subj_res)                
    return subj_res

def print_ppd_test_result_for_subject(subj_res, width = 10):
    """
    print results of ppd_test for single subject
    Input:
        subj_res - the results from ppd_test of a single subject or 
                an element from the output of ppd_test for a group model 
    """
    
    conds = sorted(subj_res.keys())
    stats = sorted(subj_res[conds[0]].keys())
    n_conds = len(conds)
    
    #create table for print
    table = [None] * (len(stats)+1)
    table[0] = [''] + conds + ['sum']
    total_conf = np.zeros(n_conds, dtype=np.int)
    for i in range(len(stats)):
        p = [str(subj_res[x][stats[i]]['p']) for x in conds]
        out_conf = [subj_res[x][stats[i]]['out_conf'] for x in conds]
        total_conf += out_conf
        
        table[i+1] = [stats[i]] + [None]*(n_conds+1)
        for j in range(n_conds):
            if out_conf[j]:
                table[i+1][j+1] = p[j]
            else:
                table[i+1][j+1] = '-'
        table[i+1][-1] = str(sum(out_conf))
    
    table.append(['sum'] + map(str, list(total_conf)))

    #print
    print "\n"
    print "----------------  stats v.s. conds ----------------\n" 
    print "(quantiles of the statistics that fell out side of the confidence interval)\n"
    print table_print.indent(table, hasHeader=True,
                             wrapfunc=lambda x:table_print.wrap_onspace_strict(x,width))
    
def print_ppd_test_result_for_group(group_res, width = 10):
    """
    print results of ppd_test for group model
    Input:
        group_res - the results from ppd_test of a group model
    """
    
    conds = sorted(group_res[0].keys())
    stats = sorted(group_res[0][conds[0]].keys())
    n_conds = len(conds)
    n_stats = len(stats)
    n_subjs = len(group_res)
    
    #compute summary tables
    sum_stats = np.zeros((n_stats, n_conds), dtype=np.int)
    sum_subjs =  np.zeros((n_subjs, n_conds), dtype=np.int)
    for i_subj in range(n_subjs):
        for i_stat in range(n_stats):
            out_conf = [group_res[i_subj][x][stats[i_stat]]['out_conf'] for x in conds]
            sum_stats[i_stat] += array(out_conf) 
            sum_subjs[i_subj] += array(out_conf)


    #print stats table
    stats_table = [None] * (n_stats+1)
    stats_table[0] = [''] + conds + ['sum']    
    for i_stat in range(n_stats):
        stats_table[i_stat+1] = [stats[i_stat]] + [None]*(n_conds+1)
        stats_table[i_stat+1][1:-1] = [str(x).replace('0','-') for x in sum_stats[i_stat]]
        stats_table[i_stat+1][-1] = str(sum(sum_stats[i_stat]))
        
    stats_table.append(['sum'] + map(str, sum_stats.sum(0)))
    
    print "\n"
    print "----------------  stats v.s. conds ----------------"
    print "(number of statistics that fell out side of the confidence interval)\n"
    print table_print.indent(stats_table, hasHeader=True,
                             wrapfunc=lambda x:table_print.wrap_onspace_strict(x,width))
    
    #print subjs table
    subjs_table = [None] * (n_subjs+1)
    subjs_table[0] = [''] + conds + ['sum']
    for i_subj in range(n_subjs):
        subjs_table[i_subj+1] = [str(i_subj)] + [None]*(n_conds+1)
        subjs_table[i_subj+1][1:-1] = [str(x).replace('0','-') for x in sum_subjs[i_subj]]
        subjs_table[i_subj+1][-1] = str(sum(sum_subjs[i_subj]))
        
    subjs_table.append(['sum'] + map(str, sum_subjs.sum(0)))
   
    print "\n"
    print "----------------  subjects v.s. conds ----------------"
    print "(number of statistics that fell out side of the confidence interval)\n"
    print table_print.indent(subjs_table, hasHeader=True,
                             wrapfunc=lambda x:table_print.wrap_onspace_strict(x,width))
    
    


def plot_posteriors(model):                 
    """Generate posterior plots for each parameter.

    This is a wrapper for pymc.Matplot.plot()
    """
    pm.Matplot.plot(model.mc)

def data_plot(data, nbins=50):
    data = hddm.utils.flip_errors(data)
    plt.figure()
    plt.hist(data['rt'], nbins)
    plt.show()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
