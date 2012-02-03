from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import hddm
import sys
import table_print

from scipy.stats import scoreatpercentile
from numpy import array, zeros, ones, empty
from copy import deepcopy
from time import time

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

    if only_group_stats:
        print kabuki.analyze.print_group_stats(m.mc.stats())
    else:
        print kabuki.analyze.print_stats(m.mc.stats())

    print "logp: %f" % m.mc.logp
    print "DIC: %f" % m.mc.dic

    if plot:
        hddm.plot_posteriors(m)
        print "Plotting posterior predictive..."
        kabuki.analyze.plot_posterior_predictive(m, value_range=np.linspace(-3, 3, 100), savefig=True)

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
    params = {'a': 0, 'v': 0, 't':0, 'z': 0.5, 'Z': 0, 'T': 0, 'V': 0}
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


def plot_posteriors(model, **kwargs):
    """Generate posterior plots for each parameter.

    This is a wrapper for pymc.Matplot.plot()
    """
    pm.Matplot.plot(model.mc, **kwargs)


def data_plot(data, nbins=50):
    data = hddm.utils.flip_errors(data)
    plt.figure()
    plt.hist(data['rt'], nbins)
    plt.show()


def quantiles_summary(hm, method, n_samples,
                      quantiles = (10, 30, 50, 70, 90), sorted_idx = None,
                      cdf_range  = (-5, 5)):
    """
    generate quantiles summary of data or samples
    Input:
        hm - hddm model
        method - which parameters to use for the computation of quantiles
            'observed' - quantiles from the observed data (not using any parameters)
            'deviance' - quantiles of data sampled using the parameters that have maximum deviance
            'samples'  - quantiles of data generated from samples from the posterior distrbution
        qunaitles - the qunatiles to be generated
        sorted_idx - the condtions will be sort according to sorted_idx.
            if sorted_idx == None then they will be sorted according to the
                accuracy achieved in each condition.
        cdf_range (advanced) - the range of the cdf used to generate the estimated
            quantiles.

    Ouput:
        group_dict - a dictionary with group statistics
        subj_dict - a list of dictionaries wih subjects statistcs
        conds - the list of conditions sorted associated with the statistics
        sorted_idx - the indices of the conditions such that:
            if we set: keys = hm.params_dict['wfpt'].subj_nodes.keys()
            than: keys[sorted_idx] == conds

    """

    #check if group model
    if hm.is_group_model:
        is_group = True
        n_subj = hm._num_subjs
    else:
        n_subj = 1
        is_group = False

    #check method
    if method == 'observed':
        n_samples = 1
        len_trace = 1
    elif method == 'samples':
        db =hm.mc.db
        name = list(hm.mc.stochastics)[0].__name__
        len_trace = db.trace(name).length()
        params = hddm.generate.gen_rand_params()
    elif method == 'deviance':
        db =hm.mc.db
        params = hddm.generate.gen_rand_params()
    else:
        raise ValueError, "unkown method"

    #Init
    n_q = len(quantiles)
    wfpt_dict = hm.params_dict['wfpt'].subj_nodes
    conds = wfpt_dict.keys()
    n_conds = len(conds)
    temp_dict = {}
    group_dict = {}; subj_dict = [None]*n_subj;
    count = zeros((n_conds * 2, n_samples))
    temp_dict['q'] = zeros((n_conds * 2, n_samples, n_q))
    temp_dict['prob'] = zeros((n_conds * 2, n_samples))
    temp_dict['n'] = zeros((n_conds*2, n_samples))
    group_dict = deepcopy(temp_dict)
    for i_subj in range(n_subj):
        subj_dict[i_subj] = deepcopy(temp_dict)

    #loop over subjs
    for i_subj in range(n_subj):
        #get wfpt nodes
        if is_group:
            wfpt = [x[i_subj] for x in hm.params_dict['wfpt'].subj_nodes.values()]
        else:
            wfpt = hm.params_dict['wfpt'].subj_nodes.values()
        #loop over conds
        for i_cond in range(n_conds):
            wfpt_params = dict(wfpt[i_cond].parents)
            data_len = len(wfpt[i_cond].value)
            for i_sample in xrange(n_samples):
                #get rt
                if method == 'observed':
                    rt = wfpt[i_cond].value
                else:
                    #get sample_idx
                    if method == 'deviance':
                        sample_idx  = np.argmin(db.trace('deviance')[:])
                    else:
                        sample_idx = ((len_trace - 1) // (n_samples - 1)) * i_sample
                    #get params
                    for key in params.iterkeys():
                        if isinstance(wfpt_params[key], pm.Node):
                            node = wfpt_params[key]
                            name = node.__name__
                            params[key] = db.trace(name)[sample_idx]
                    #generate rt
                    rt = hddm.generate.gen_rts(params, samples=data_len, range_=cdf_range,
                                               dt=1e-3, intra_sv=1., structured=False,
                                               subj_idx=None, method='cdf')
                #loop over responses
                for i_resp in range(2):
                    cond_ind = i_cond * 2 + i_resp
                    if i_resp==0:
                        t_rt = rt[rt > 0]
                    else:
                        t_rt = rt[rt < 0]
                    #get n_q
                    if len(t_rt)>=n_q:
                        t_quan = [scoreatpercentile(abs(t_rt),x) for x in quantiles]
                        subj_dict[i_subj]['q'][cond_ind, i_sample, :] = t_quan
                        group_dict['q'][cond_ind, i_sample, :] += t_quan
                        count[cond_ind, i_sample] += 1
                    else:
                        subj_dict[i_subj]['q'][cond_ind, i_sample, :] = np.NaN
                    t_prob = 1.*len(t_rt) / len(rt)
                    subj_dict[i_subj]['prob'][cond_ind, i_sample] = t_prob
                    group_dict['prob'][cond_ind, i_sample] += t_prob

                    subj_dict[i_subj]['n'][cond_ind, i_sample] = len(t_rt)
                    group_dict['n'][cond_ind, i_sample] = len(t_rt)



    #compute group values
    group_dict['prob'] /= n_subj
    for i_c in range(n_conds * 2):
        for i_sample in xrange(n_samples):
            if count[i_c, i_sample] > 0:
                group_dict['q'][i_c, i_sample,:] /= count[i_c, i_sample]
            else:
                group_dict['q'][i_c, i_sample,:] = np.NaN
    #sort
    if sorted_idx == None:
        sorted_idx = np.argsort(group_dict['prob'][:,0]).flatten()
    group_dict['prob'] = group_dict['prob'][sorted_idx,:]
    group_dict['q'] = group_dict['q'][sorted_idx,:,:]
    group_dict['n'] = group_dict['n'][sorted_idx,:]
    for i_subj in range(n_subj):
        subj_dict[i_subj]['prob'] = subj_dict[i_subj]['prob'][sorted_idx,:]
        subj_dict[i_subj]['q'] = subj_dict[i_subj]['q'][sorted_idx,:, :]
        subj_dict[i_subj]['n'] = subj_dict[i_subj]['n'][sorted_idx, :]

    conds = array(conds)[(sorted_idx // 2)]
    return group_dict, subj_dict, conds, sorted_idx



def _draw_qp_plot(dict, conds, conds_to_plot, title_str, samples_summary,
                  marker, handle = None):
    """
    draw QP plot
    Input:
        dict - subject/group dictionary from qunatiles_summary
        conds - conditions from quantiles_summary
        conds_to_plot - a list of conds to plot
        title_str - the title of the plot
        samples_summary - sould we plot only samples sammury and not the samples themselves
        maarker - marker used in plot
        handle - axes's handle
    """

    colors  = ['b', 'g', 'c', 'm', 'r', 'y', 'k']
    n_q = dict['q'].shape[2]
    n_conds = len(conds)//2
    n_samples = dict['prob'].shape[1]
    #group plot
    if handle == None:
        f = plt.figure()
        handle = f.add_subplot(111)
    color_counter = -1
    for i_c in range(n_conds):
        #check if we need to plot this condition
        if i_c not in conds_to_plot:
            continue
        color_counter += 1
        #loop over correct and incorrect reponses
        for i_resp in range(2):
            if i_resp == 0:
                idx = i_c
            else:
                idx = (n_conds * 2) - 1 - i_c
            #get color of marker
            cc = colors[color_counter % len(colors)]
            #plot it
            if samples_summary:
                ok_idx = ~np.isnan(dict['prob'][idx,:])
                if sum(ok_idx) == 0:
                    continue
                mean_prob = np.mean(dict['prob'][idx,ok_idx])
                xerr = np.empty((2,n_q))
                xerr[0,:] = mean_prob - scoreatpercentile(dict['prob'][idx,ok_idx], 2.5)
                xerr[1,:] = scoreatpercentile(dict['prob'][idx,ok_idx], 97.5) - mean_prob

                mean_q = np.mean(dict['q'][idx,ok_idx,:],0)
                yerr = np.empty((2,n_q))
                yerr[0,:] = array([scoreatpercentile(x, 2.5) for x in dict['q'][idx,ok_idx,:].T]) - mean_q
                yerr[1,:] = mean_q - array([scoreatpercentile(x, 97.5) for x in dict['q'][idx,ok_idx,:].T])
                handle.errorbar(ones(n_q)*mean_prob, mean_q, yerr=yerr, xerr=xerr,
                                marker=marker, fmt=cc)

            else:
                for i_sample in range(n_samples):
                    if np.isnan(dict['prob'][idx, i_sample]):
                        continue
                    handle.plot(ones(n_q)*dict['prob'][idx,i_sample],
                                      dict['q'][idx,i_sample,:],'%s%s' % (cc, marker))

    #add title and labels
    if title_str != None:
        plt.gcf().canvas.set_window_title(title_str)
        plt.title(title_str)
        plt.xlabel('probability')
        plt.ylabel('RT')
        plt.xlim([-0.05, 1.05])

    handle.get_figure().canvas.draw()
    return handle


def qp_plot(hm, quantiles = (10, 30, 50, 70, 90), plot_subj=True,
            split_func=lambda x:0, method = None, samples_summary=True,
            n_samples=50, cdf_range=(-5, 5)):
    """
    generate a quantile-probability plot
    Input:
        hm - hddm model
        qunatiles - the quantiles that are going to be displyed
        plot_subj - generate a QP plot for each subject as well
        split_func - split the plot into several plots according to the output of
            the function
        method - which values will be used to compute the estimated quantiles.
            may be one of the followings:
            -'none'   : display only observed quantiles
            -'deviance': use the point of minimum deviance
            -'samples': samples n_samples and plot them all (not recommended)
            -'samples_summary': samples n_samples and plot the mean and 95 credible set of them
        n_samples - see method
        cdf_range (advanced) - the range of the cdf used to generate the estimated
            quantiles.

        TODO:
            there should be an option to use the average value of the parameters to create samples.
            it make much more sense than 'deviance'
    """

    #### compute empirical quantiles
    #Init
    n_q = len(quantiles)
    if hm.is_group_model:
        is_group = True
        n_subj = hm._num_subjs
    else:
        n_subj = 1
        is_group = False

    #get group and subj dicts
    obs_group_d, obs_subj_d, conds, sorted_idx = \
    quantiles_summary(hm, 'observed', n_samples = 1, quantiles = quantiles)

    #get splits
    keys = [split_func(x) for x in conds]
    dic  = {}
    for i_s in range(len(keys)):
        if dic.has_key(keys[i_s]):
            dic[keys[i_s]].append(i_s)
        else:
            dic[keys[i_s]] = [i_s]
    splits = dic.values()
    splits_keys = dic.keys()

    #plot
    g_handles = [None]*len(splits)
    s_handles = [[None]*n_subj for x in range(len(splits))]
    for i_s in range(len(splits)):
        g_handles[i_s] = _draw_qp_plot(obs_group_d, conds, splits[i_s],
                                title_str = "QP group (%s)" % splits_keys[i_s],
                                samples_summary=False, marker = 'd')
        if not plot_subj:
            continue
        for i_subj in range(n_subj):
            s_handles[i_s][i_subj] = _draw_qp_plot(obs_subj_d[i_subj], conds, splits[i_s],
                                           title_str = "QP %d (%s)" % (i_subj, splits_keys[i_s]),
                                           samples_summary=False, marker = 'd')

    if (method == None) or (method == 'none'):
        return
    elif (method == 'deviance'):
        quantiles_method = 'deviance'
    elif method == 'samples':
        quantiles_method = 'samples'
    else:
        raise ValueError, "unknown method"

    #### compute estimated quantiles

    #get group and subj dicts
    print "getting quantiles summary of samples"
    i_t = time();
    sim_group_d, sim_subj_d, conds, sorted_idx = \
    quantiles_summary(hm, method=method, n_samples=n_samples,
                      quantiles=quantiles, sorted_idx=sorted_idx)
    print "took %d seconds to prepare quantiles" % (time() - i_t)
    sys.stdout.flush()

    #plot
    for i_s in range(len(splits)):
        _draw_qp_plot(sim_group_d, conds, splits[i_s],
                      title_str=None, samples_summary=samples_summary,
                      marker='o', handle=g_handles[i_s])
        if not plot_subj:
            continue
        for i_subj in range(n_subj):
            _draw_qp_plot(sim_subj_d[i_subj], conds, splits[i_s],
                          title_str = None, samples_summary=samples_summary,
                          marker = 'o', handle = s_handles[i_s][i_subj])


if __name__ == "__main__":
    import doctest
    doctest.testmod()

