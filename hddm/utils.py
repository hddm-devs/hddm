from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import hddm
import sys
import kabuki
import pandas as pd
import string
from kabuki.analyze import post_pred_gen, post_pred_compare_stats

from scipy.stats import scoreatpercentile
from scipy.stats.mstats import mquantiles

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
    data = data.copy()

    # Flip sign for lower boundary response
    idx = data['response'] == 0
    data['rt'][idx] = -data['rt'][idx]

    return data

def check_params_valid(**params):
    a = params.get('a')
    v = params.get('v')
    z = params.get('z', .5)
    t = params.get('t')
    sv = params.get('sv', 0)
    st = params.get('st', 0)
    sz = params.get('sz', 0)

    if (sv < 0) or (a <= 0) or (z < 0) or (z > 1) or (sz < 0) or (sz > 1) or (z+sz/2. > 1) or \
    (z-sz/2. < 0) or (t-st/2. < 0) or (t < 0) or (st < 0):
        return False
    else:
        return True

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
        weights = np.asarray(weights)
        if np.any(weights.shape != a.shape):
            raise ValueError(
                    'weights should have the same shape as a.')
        weights = weights.ravel()
    a = a.ravel()

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
        mn, mx = [mi + 0.0 for mi in range]
        if mn == mx:
            mn -= 0.5
            mx += 0.5
        bins = np.linspace(mn, mx, bins + 1, endpoint=True)
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
            sa = np.sort(a[i:i + block])
            n += np.r_[sa.searchsorted(bins[:-1], 'left'), \
                sa.searchsorted(bins[-1], 'right')]
    else:
        zero = np.array(0, dtype=ntype)
        for i in np.arange(0, len(a), block):
            tmp_a = a[i:i + block]
            tmp_w = weights[i:i + block]
            sorting_index = np.argsort(tmp_a)
            sa = tmp_a[sorting_index]
            sw = tmp_w[sorting_index]
            cw = np.concatenate(([zero, ], sw.cumsum()))
            bin_index = np.r_[sa.searchsorted(bins[:-1], 'left'), \
                sa.searchsorted(bins[-1], 'right')]
            n += cw[bin_index]

    n = np.diff(n)

    if density is not None:
        if density:
            db = np.array(np.diff(bins), float)
            return n / db / n.sum(), bins
        else:
            return n, bins
    else:
        # deprecated, buggy behavior. Remove for Numpy 2.0
        if normed:
            db = np.array(np.diff(bins), float)
            return n / (n * db).sum(), bins
        else:
            return n, bins


def parse_config_file(fname, map=True, mcmc=False, data=None, samples=None, burn=None, thin=None, only_group_stats=False, plot=True, verbose=False):
    import kabuki
    import os.path

    if not os.path.isfile(fname):
        raise ValueError("%s could not be found." % fname)

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
        raise IOError("Data file %s not found." % data_fname)

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
    if samples is None:
        try:
            samples = config.getint('mcmc', 'samples')
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            samples = 10000

    if burn is None:
        try:
            burn = config.getint('mcmc', 'burn')
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            burn = 5000

    if thin is None:
        try:
            thin = config.getint('mcmc', 'thin')
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            thin = 2

    group_params = ['v', 'sv', 'a', 'z', 'sz', 't', 'st']

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

    m.print_stats()
    m.print_stats(fname='stats.txt')
    print "logp: %f" % m.mc.logp
    print "DIC: %f" % m.mc.dic

    if plot:
        m.plot_posteriors(save=True)
        print "Plotting posterior predictive..."
        m.plot_posterior_predictive(save=True)

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
                params['v_%i' % subj] = v
                params['a_%i' % subj] = a
                params['t_%i' % subj] = t  # t-.2 if t-.2>0 else .1
                params['z_%i' % subj] = .5
            except ValueError:
                # Subject either had 0%, 50%, or 100% correct, which does not work
                # with easy. But we can deal with that by just not initializing the
                # parameters for that one model.
                params['v_%i' % subj] = None
                params['a_%i' % subj] = None
                params['t_%i' % subj] = None
                params['z_%i' % subj] = None

    except ValueError:
        # Data array has no subj_idx -> ignore.
        pass

    return params


def EZ_param_ranges(data, range_=1.):
    v, a, t = EZ_data(data)
    z = .5
    param_ranges = {'a_lower': a - range_,
                    'a_upper': a + range_,
                    'z_lower': np.min(z - range_, 0),
                    'z_upper': np.max(z + range_, 1),
                    't_lower': t - range_ if (t - range_) > 0 else 0.,
                    't_upper': t + range_,
                    'v_lower': v - range_,
                    'v_upper': v + range_}

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

    s2 = s ** 2
    logit_p = np.log(pc / (1 - pc))

    # Eq. 7
    x = ((logit_p * (pc**2 * logit_p - pc * logit_p + pc - .5)) / vrt)
    v = np.sign(pc - .5) * s * x**.25
    # Eq 5
    a = (s2 * logit_p) / v

    y = (-v * a) / s2
    # Eq 9
    mdt = (a / (2 * v)) * ((1 - np.exp(y)) / (1 + np.exp(y)))

    # Eq 8
    ter = mrt - mdt

    return (v, a, ter)

def hddm_parents_trace(model, obs_node, idx):
    """Return the parents' value of an wfpt node in index 'idx' (the
    function is used by ppd_test)
    """
    model.params_include.keys()
    params = {'a': 0, 'v': 0, 't':0, 'z': 0.5, 'sz': 0, 'st': 0, 'sv': 0}
    if not np.isscalar(idx):
        for (key, value) in params.iteritems():
            params[key] = np.ones(len(idx)) * value
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
    test['func'] = lambda rts: sum(rts > 0) / len(rts)
    statistics.append(test)

    ##quantile statistics of absolute response time
    quantiles = [10, 30, 50, 70, 90]
    for q in quantiles:
        test = {}
        test['name'] = 'q%d' % q
        test['func'] = lambda rts, q=q: scoreatpercentile(np.abs(rts), q)
        statistics.append(test)

    return statistics

def gen_ppc_stats(quantiles=(10, 30, 50, 70, 90)):
    """Generate default statistics for posterior predictive check on
    RT data.

    :Returns:
        OrderedDict mapping statistic name -> function
    """
    from collections import OrderedDict

    stats = OrderedDict()
    stats['accuracy'] = lambda x: np.mean(x>0)

    #upper bound stats
    stats['mean_ub'] = lambda x: np.mean(x[x>0])
    stats['std_ub'] = lambda x: np.std(x[x>0])
    for q in quantiles:
        key = str(q) + 'q'
        stats[key+'_ub'] = lambda x, q=q: scoreatpercentile(x[x>0], q) if np.any(x>0) else np.nan

    #lower bound stats
    stats['mean_lb'] = lambda x: np.mean(x[x<0])
    stats['std_lb'] = lambda x: np.std(x[x<0])
    for q in quantiles:
        key = str(q) + 'q'
        stats[key+'_lb'] = lambda x, q=q: scoreatpercentile(np.abs(x[x<0]), q) if np.any(x<0) else np.nan

    return stats


def post_pred_stats(data, sim_datasets, **kwargs):
    """Calculate a set of summary statistics over posterior predictives.

    :Arguments:
        data : pandas.DataFrame

        sim_data : pandas.DataFrame

    :Optional:
        bins : int
            How many bins to use for computing the histogram.
        evals : dict
            User-defined evaluations of the statistics (by default 95 percentile and SEM).
            :Example: {'percentile': scoreatpercentile}
        plot : bool
            Whether to plot the posterior predictive distributions.
        progress_bar : bool
            Display progress bar while sampling.

    :Returns:
        Hierarchical pandas.DataFrame with the different statistics.

    """
    data = flip_errors(data)
    sim_datasets = flip_errors(sim_datasets)

    if 'stats' not in kwargs:
        kwargs['stats'] = gen_ppc_stats()

    return kabuki.analyze.post_pred_stats(data['rt'], sim_datasets['rt'], **kwargs)

def plot_posteriors(model, **kwargs):
    """Generate posterior plots for each parameter.

    This is a wrapper for pymc.Matplot.plot()
    """
    pm.Matplot.plot(model.mc, **kwargs)


def data_plot(model, bins=50, nrows=3):
    nplots = len(model.get_observeds())
    if nplots < nrows:
        nrows = nplots
    ncols = int(np.ceil(nplots / nrows))

    bin_edges = np.linspace(0, np.abs(model.data.rt).max(), bins+1)


    fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    axs = axs.flatten()
    for (i_plt, (name, node_row)) in enumerate(model.iter_observeds()):
        node = node_row['node']
        ax = axs[i_plt]
        for i in range(2):
            if i == 0:
                value = node.value[node.value > 0]
            else:
                value = np.abs(node.value[node.value < 0])

            counts, bin_edges = np.histogram(value, bins=bin_edges)
            dx = (bin_edges[1] - bin_edges[0]) / 2.
            ax.plot(bin_edges[:-1] + dx, counts, lw=2.)
        ax.set_title(pretty_tag(node_row['tag']))

    plt.show()


def _plot_posterior_quantiles_node(node, axis, quantiles=(.1, .3, .5, .7, .9),
                                   samples=100, alpha=.75, hexbin=True,
                                   value_range=(0, 5),
                                   data_plot_kwargs=None, predictive_plot_kwargs=None):
    """Plot posterior quantiles for a single node.

    :Arguments:

        node : pymc.Node
            Must be observable.

        axis : matplotlib.axis handle
            Axis to plot into.

    :Optional:

        value_range : numpy.ndarray
            Range over which to evaluate the CDF.

        samples : int (default=10)
            Number of posterior samples to use.

        alpha : float (default=.75)
           Alpha (transparency) of posterior quantiles.

        hexbin : bool (default=False)
           Whether to plot posterior quantile density
           using hexbin.

        data_plot_kwargs : dict (default=None)
           Forwarded to data plotting function call.

        predictive_plot_kwargs : dict (default=None)
           Forwareded to predictive plotting function call.

    """

    quantiles = np.asarray(quantiles)

    axis.set_xlim(value_range)
    axis.set_ylim((0, 1))

    sq_lower = np.empty((len(quantiles), samples))
    sq_upper = sq_lower.copy()
    sp_upper = np.empty(samples)
    for i_sample in range(samples):
        kabuki.analyze._parents_to_random_posterior_sample(node)
        sample_values = node.random()
        sq_lower[:, i_sample], sq_upper[:, i_sample], sp_upper[i_sample] = data_quantiles(sample_values)

    y_lower = np.dot(np.atleast_2d(quantiles).T, np.atleast_2d(1 - sp_upper))
    y_upper = np.dot(np.atleast_2d(quantiles).T, np.atleast_2d(sp_upper))
    if hexbin:
        if predictive_plot_kwargs is None:
            predictive_plot_kwargs = {'gridsize': 75, 'bins': 'log', 'extent': (value_range[0], value_range[1], 0, 1)}
        x = np.concatenate((sq_lower, sq_upper))
        y = np.concatenate((y_lower, y_upper))
        axis.hexbin(x.flatten(), y.flatten(), label='post pred lb', **predictive_plot_kwargs)
    else:
        if predictive_plot_kwargs is None:
            predictive_plot_kwargs = {'alpha': .75}
        axis.plot(sq_lower, y_lower, 'o', label='post pred lb', color='b', **predictive_plot_kwargs)
        axis.plot(sq_upper, y_upper, 'o', label='post pred ub', color='r', **predictive_plot_kwargs)

    # Plot data
    data = node.value
    color = 'w' if hexbin else 'k'
    if data_plot_kwargs is None:
        data_plot_kwargs = {'color': color, 'lw': 2., 'marker': 'o', 'markersize': 7}

    if len(data) != 0:
        q_lower, q_upper, p_upper = data_quantiles(data)

        axis.plot(q_lower, quantiles*(1-p_upper), **data_plot_kwargs)
        axis.plot(q_upper, quantiles*p_upper, **data_plot_kwargs)

    axis.set_xlabel('RT')
    axis.set_ylabel('Prob respond')
    axis.set_ylim(bottom=0) # Likelihood and histogram can only be positive


def plot_posterior_quantiles(model, **kwargs):
    """Plot posterior predictive quantiles.

    :Arguments:

        model : HDDM model

    :Optional:

        value_range : numpy.ndarray
            Range over which to evaluate the CDF.

        samples : int (default=10)
            Number of posterior samples to use.

        alpha : float (default=.75)
           Alpha (transparency) of posterior quantiles.

        hexbin : bool (default=False)
           Whether to plot posterior quantile density
           using hexbin.

        data_plot_kwargs : dict (default=None)
           Forwarded to data plotting function call.

        predictive_plot_kwargs : dict (default=None)
           Forwareded to predictive plotting function call.

        columns : int (default=3)
            How many columns to use for plotting the subjects.

        save : bool (default=False)
            Whether to save the figure to a file.

        path : str (default=None)
            Save figure into directory prefix

    """

    if 'value_range' not in kwargs:
        rt = np.abs(model.data['rt'])
        kwargs['value_range'] = (np.min(rt.min()-.2, 0), rt.max())

    kabuki.analyze.plot_posterior_predictive(model,
                                             plot_func=_plot_posterior_quantiles_node,
                                             **kwargs)


def create_test_model(samples=5000, burn=1000, subjs=1, size=100):
    data, params = hddm.generate.gen_rand_data(subjs=subjs, size=size)
    m = hddm.HDDM(data)
    m.sample(samples, burn=burn)

    return m

def pretty_tag(tag):
    return tag[0] if len(tag) == 1 else string.join(tag, ', ')

def qp_plot(model, quantiles=(0.1, 0.3, 0.5, 0.7, 0.9), ncols=None):
    """
    qp plot
    Input:
        model : HDDM model

        quantiles : sequence
            sequence of quantiles

        ncols : int
            number of columns in output figure
    """

    #if model is a group model then we create an average model and plot it
    if model.is_group_model:
        avg_model = model.get_average_model()
        qp_plot(avg_model, quantiles=quantiles)

    #create axes
    n_subjs = model.num_subjs
    if ncols is None:
        ncols = min(4, n_subjs)
    nrows = int(np.ceil(n_subjs / ncols))
    fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=False)

    #plot single subject model
    obs = model.get_observeds()
    if not model.is_group_model:
        axs = np.array([axs])
        _qp_plot_of_nodes_db(obs, quantiles=quantiles, ax=axs[0])

    #plot group model
    else:
        for i, (subj_idx, t_nodes_db) in enumerate(obs.groupby('subj_idx')):
            _qp_plot_of_nodes_db(t_nodes_db, quantiles=quantiles, ax=axs.flat[i])

    #add legend
    leg = axs.flat[0].legend(loc='best', fancybox=True)
    if leg is not None:
        leg.get_frame().set_alpha(0.5)

def _qp_plot_of_nodes_db(nodes_db, quantiles, ax):
    """
    function used by qp_plot
    Input
        nodes_db : DataFrame
            DataFram of observeds nodes (e.g. model.get_observeds())

        quantiles : sequence
            sequence of quantiles

        ax : axes
            axes to plot on
    """

    # ax.set_xlim(0,1)
    nq = len(quantiles)
    nodes_db = nodes_db.sort_index()

    #loop over nodes
    for name, node_row in nodes_db.iterrows():

        #get quantiles
        q_lower, q_upper, p_upper = node_row['node'].empirical_quantiles(quantiles)

        #plot two lines for each node
        tag = node_row['tag']
        tag = pretty_tag(tag)
        line = ax.plot(np.ones(nq)*p_upper, q_upper, '-x', label=tag)[0]
        ax.plot(np.ones(nq)*(1-p_upper), q_lower, '-x', c=line.get_color())[0]


def data_quantiles(data, quantiles=(0.1, 0.3, 0.5, 0.7, 0.9)):
    """
    compute the quantiles of 2AFC data

    Output:
        q_lower - lower boundary quantiles
        q_upper - upper_boundary_quantiles
        p_upper - probability of hitting the upper boundary
    """
    if isinstance(data, pd.DataFrame):
        if 'response' in data.columns:
            data = flip_errors(data).rt.values
        else:
            data = data.rt.values

    quantiles = np.asarray(quantiles)
    p_upper = float(np.mean(data>0))
    q_lower = mquantiles(-data[data<0], quantiles)
    q_upper = mquantiles(data[data>0], quantiles)

    return q_lower, q_upper, p_upper

if __name__ == "__main__":
    import doctest
    doctest.testmod()
