.. index:: Demo
.. _chap_demo:

Demo
----

In the following we will show an example session of using HDDM to
analyze a real-world dataset. The main purpose is to provide an overview
of some of the funcionality and interface. By no means, however, is it a
complete overview of all the functionality in HDDM. For more
information, including on how to use HDDM as a command-line utility, 
online tutorial and a refernce manual
see http://ski.clps.brown.edu/hddm_docs.

First, we have to load the modules we are going to use.

.. code:: python

    import hddm
    import matplotlib.pyplot as plt

We import HDDM(0.5dev) and matplotlib, a module for generating graphs and


Loading data
````````````

Next, we will load in a data set. The easiest way to get your data into
HDDM is by storing it in a csv (comma-separated-value, see below) file.
In this example we will be using data collected in a reward-based
decision making experiment in our lab (:cite:CavanaghWieckiCohenEtAl11). In brief, at each trial subjects choose
between two symbols. The trials were divided into win-win trials (WW),
in which the two symbols were associated with high winning chances;
lose-lose trials (LL), in which the symbols were associated with low
winning chances, and win-lose trials (WL), which are the easiest because
only one symbol was associated with high winning chances. Thus WW and LL
decisions together comprise high conflict
(HC) trials (although there are other differences between them, we
don't focus on those here), whereas WL decisions are low conflict (LC).
The main hypothesis of the study was that high conflict trials induce an
increase in the decision threshold, and that the mechanism for this
threshold modulation depends on communication between mediofrontal
cortex (which exhibits increased activity under conditions of choice
uncertainty or conflict) and the subthalamic nucleus (STN) of the basal
ganglia (which provides a temporary brake on response selection by
increasing the decision threshold). The details of this mechanism are
described in other modeling papers (e.g., Ratcliff & Frank 2012).
Cavanagh et al 2011 tested this theory by measuring EEG activity over
mid-frontal cortex, focusing on the theta band, given prior associations
with conflict, and testing whether trial-to-trial variations in frontal
theta were related to adjustments in decision threshold during high
conflict trials. They tested the STN component of the theory by
administering the same experiment to patients who had deep brain
stimulation (dbs) of the STN, which interferes with normal processing.


The first ten lines of the data file look as follows:

.. code:: python

    !head hddm_demo.csv

.. parsed-literal::

    subj_idx,stim,rt,response,theta,dbs,conf
    0,LL,1.21,1.0,0.65627512226100004,1,HC
    0,WL,1.6299999999999999,1.0,-0.32788867166199998,1,LC
    0,WW,1.03,1.0,-0.480284512399,1,HC
    0,WL,2.77,1.0,1.9274273452399999,1,LC
    0,WW,1.1399999999999999,0.0,-0.21323572605999999,1,HC
    0,WL,1.1499999999999999,1.0,-0.43620365940099998,1,LC
    0,LL,2.0,1.0,-0.27447891439400002,1,HC
    0,WL,1.04,0.0,0.66695707371400004,1,LC
    0,WW,0.85699999999999998,1.0,0.11861689909799999,1,HC


We use the ``hddm.load_csv()`` function to load this file.

.. code:: python

    data = hddm.load_csv('hddm_demo.csv')

.. code:: python

    data.head(10)

.. parsed-literal::

       subj_idx stim     rt  response     theta  dbs conf
    0         0   LL  1.210         1  0.656275    1   HC
    1         0   WL  1.630         1 -0.327889    1   LC
    2         0   WW  1.030         1 -0.480285    1   HC
    3         0   WL  2.770         1  1.927427    1   LC
    4         0   WW  1.140         0 -0.213236    1   HC
    5         0   WL  1.150         1 -0.436204    1   LC
    6         0   LL  2.000         1 -0.274479    1   HC
    7         0   WL  1.040         0  0.666957    1   LC
    8         0   WW  0.857         1  0.118617    1   HC
    9         0   WL  1.500         0  0.823626    1   LC

Lets look at the RT distributions of each individual subject using
pandas' ``groupby()`` functionality. Because there are two possible
responses (here we are using accuracy coding where 1 means the more
rewarding symbol was chosen, and 0 the less rewarding) we flip error RTs
to be negative.

.. code:: python

    data = hddm.utils.flip_errors(data)

    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel='RT', ylabel='count', title='RT distributions')
    for i, subj_data in data.groupby('subj_idx'):
        ax.hist(subj_data.rt, bins=20, histtype='step')

.. image:: hddm_demo_files/hddm_demo_fig_00.png

Fitting a hierarchical model
````````````````````````````

Lets fit a hierarchical DDM to this data set, starting off first with
the simplest model that does not allow parameters to vary by condition.

.. code:: python

    # Instantiate model object passing it our data (no need to call flip_errors() before passing it).
    # This will tailor an individual hierarchical DDM around your dataset.
    m = hddm.HDDM(data)
    # find a good starting point which helps with the convergence.
    m.find_starting_values()
    # start drawing 2000 samples and discarding 20 as burn-in
    m.sample(2000, burn=20)

We now want to analyze our estimated model. ``m.print_stats()`` will
print a table of summary statistics for each parameters' posterior.
Because that is quite long we only print a subset of the parameters
using pandas selection functionality.

.. code:: python

    stats = m.gen_stats()
    stats[stats.index.isin(['a', 'a_var', 'a_subj.0', 'a_subj.1'])]

.. parsed-literal::

                  mean       std      2.5q       25q       50q       75q     97.5q

    a         2.058015  0.102570  1.862412  1.988854  2.055198  2.123046  2.261410
    a_var     0.379303  0.089571  0.244837  0.316507  0.367191  0.426531  0.591643
    a_subj.0  2.384066  0.059244  2.274352  2.340795  2.384700  2.423012  2.500647
    a_subj.1  2.127582  0.061901  2.003605  2.086776  2.126963  2.166261  2.254350

                mc err
    a         0.002539
    a_var     0.002973
    a_subj.0  0.001727
    a_subj.1  0.002113

As you can see, the model estimated the group mean parameter for
threshold ``a``, group variability ``a_var`` and individual subject
parameters ``a_subj.0``. Other parameters are not shown here.

The inference algorithm, MCMC, requires the chains of the model to have
properly converged. While there is no way to guarantee convergence for a
finite set of samples in MCMC, there are many heuristics that allow you
identify problems of convergence. One main analysis to look at is the
trace, the autocorrelation, and the marginal posterior. You can plot
these using the ``plot_posteriors()`` function. For the sake of brevity
we only plot three here. In practice, however, you will always want to
examine all of them.

.. code:: python

    m.plot_posteriors(['a', 't', 'v', 'a_var'])

.. image:: hddm_demo_files/hddm_demo_fig_01.png

.. image:: hddm_demo_files/hddm_demo_fig_02.png

.. image:: hddm_demo_files/hddm_demo_fig_03.png

.. image:: hddm_demo_files/hddm_demo_fig_04.png

As you can see, there are no drifts or large jumps in the trace. The
autocorrelation is also very low.

We might also be interested in how well the model fits the data. To
inspect this visually you can call ``plot_posterior_predictive()`` to
plot individual subject RT distributions in red on top of the predictive
likelihood in blue.

.. code:: python

    m.plot_posterior_predictive(figsize=(14, 10))

.. image:: hddm_demo_files/hddm_demo_fig_05.png

While visually the fit looks decent, we also have prior knowledge about
our experiment which could be leveraged to improve the model. For
example, we would expect that because LL and WW trials are harder than
WL trials, drift rate would be higher in WL, which has lower uncertainty
about the correct choice. (One could also develop a posterior predictive
check statistic that would evaluate whether accuracy and mean RT are
different in the different conditions. Since the parameters of the model
were estimated to be the same across conditions, the posterior
predictive distributions for these conditions would not look different
from each other, whereas those in the data do. A formal posterior
predictive check would thus show that the data violates the simple
assumptions of the model. This is not evident above because we simply
plotted the distributions collapsed across conditions).

In any case, we can create a new model quite easily which estimates
separate drift-rate ``v`` for those different conditions by using the
``depends_on`` keyword argument. This argument expects a Python ``dict``
which maps the parameter to be split to the column name containing the
conditions we want to split by.

.. code:: python

    m_stim = hddm.HDDM(data, depends_on={'v': 'stim'})
    m_stim.find_starting_values()
    m_stim.sample(2000, burn=20)


We will skip examining the traces for this model and instead look at the
posteriors of ``v`` for the different conditions. Below you can see that
the drift rate for the low conflict WL condition is substantially
greater than that for the other two conditions, which are fairly similar
to each other.

.. code:: python

    v_WW, v_LL, v_WL = m_stim.nodes_db.node[['v(WW)', 'v(LL)', 'v(WL)']]
    kabuki.analyze.plot_posterior_nodes([v_WW, v_LL, v_WL])

.. image:: hddm_demo_files/hddm_demo_fig_06.png

While it would be easy to provide syntacic sugar for the above
expression there are many cases where you want access to the underlying
distributions. These are stored inside of ``nodes_db`` which is a pandas
``DataFrame`` containing information about each distribution. Here we
retrieve the actual node objects containing the trace from the ``node``
colum.

One benefit of estimating the model in a Bayesian framework is that we
can do significance testing directly on the posterior rather than
relying on frequentist statistics (See Kruschke's book for many examples
of the advantages of this approach). For example, we might be interested
in whether the drift-rate for WW is larger than that for LL, or whether
drift-rate for LL is larger than WL. The below code allows us to examine
the proportion of the posteriors in which the drift rate for one
condition is greater than the other. It can be seen that the posteriors
for LL do not overlap at all for WL, and thus the probability that LL is
greater than WL should be near zero.

.. code:: python

    print "P(WW > LL) = ", (v_WW.trace() > v_LL.trace()).mean()
    print "P(LL > WL) = ", (v_LL.trace() > v_WL.trace()).mean()

.. parsed-literal::

    P(WW > LL) =  0.34696969697
    P(LL > WL) =  0.0


Lets compare the two models using the deviance information criterion (DIC; lower is better). Note that the DIC measures the fit of the model to the data, penalizing for complexity in the addition of degrees of freedom (the model with three drift rates has more dF than the model with one). The DIC is known to be somewhat biased in selecting the model with greater complexity, although alternative forms exist (see Plummer 2008). One should use the DIC with caution, although other forms of model comparison such as the Bayes Factor (BF) have other problems, such as being overly sensitive to the prior parameter distributions of the models. Future versions of HDDM will include the partial Bayes Factor, which allows the BF to be computed based on informative priors taken from a subset of the data, and which we generally believe to provide a better measure of model fit. Nevertheless, DIC can be a useful metric with these caveats in mind.

.. code:: python

    print "Lumped model DIC: %f" % m.dic
    print "Stimulus model DIC: %f" % m_stim.dic

.. parsed-literal::

    Lumped model DIC: 10960.570932
    Stimulus model DIC: 10775.615192


Within-subject effects
``````````````````````

Note that while the ``m_stim`` model we created above estimates
different drift-rates ``v`` for each subject, it implicitly assumes that
the different conditions are completely independent of each other,
because each drift rate was sampled from a separate group prior.
However, there may be individual differences in overall performance, and
if so it is reasonable to assume that someone who would be better at
``WL`` would also be better at ``LL``. To model this intuition we can
use a within-subject model where an intercept is used to capture overall
performance in the 'WL' condition as a baseline, and then the other
``LL`` and ``WW`` conditions are expressed relative to ``WL``. (Perhaps
every subject has a higher drift in WL than LL but there is huge
variance in their overall drift rates. In this scenario, the earlier
model would not have the power to detect the effect of condition on this
within subject effect, because there would be large posterior variance
in all of the drift rates, which would then overlap with each other. In
contrast, the within-subject model would estimate large variance in the
intercept but still allow the model to infer a non-zero effect of
condition with high precision).

``HDDM`` supports this via the ``patsy`` module which transforms model
strings to design matrices.

.. code:: python

    from patsy import dmatrix
    dmatrix("C(stim, Treatment('WL'))", data.head(10))

.. parsed-literal::

    DesignMatrix with shape (10, 3)
      Intercept  C(stim, Treatment('WL'))[T.LL]  C(stim, Treatment('WL'))[T.WW]
              1                               1                               0
              1                               0                               0
              1                               0                               1
              1                               0                               0
              1                               0                               1
              1                               0                               0
              1                               1                               0
              1                               0                               0
              1                               0                               1
              1                               0                               0
      Terms:
        'Intercept' (column 0)
        "C(stim, Treatment('WL'))" (columns 1:3)

``Patsy`` model specifications can be passed to the ``HDDMRegressor``
class as part of a descriptor that contains the string describing the
linear model and the ``outcome`` variable that should be replaced with
the output of the linear model -- in this case ``v``.

.. code:: python

    m_within_subj = hddm.HDDMRegressor(data, "v ~ C(stim, Treatment('WL'))")

.. parsed-literal::

    Adding these covariates:
    ['v_Intercept', "v_C(stim, Treatment('WL'))[T.LL]", "v_C(stim, Treatment('WL'))[T.WW]"]


.. code:: python

    m_within_subj.sample(5000, burn=200)

.. code:: python

    v_WL, v_LL, v_WW = m_within_subj.nodes_db.node[["v_Intercept",
                                                    "v_C(stim, Treatment('WL'))[T.LL]",
                                                    "v_C(stim, Treatment('WL'))[T.WW]"]]
    kabuki.analyze.plot_posterior_nodes([v_WL, v_LL, v_WW])

.. image:: hddm_demo_files/hddm_demo_fig_07.png

Note that in the above plot ``LL`` and ``WW`` are expressed relative to
the ``WL`` condition (i.e. ``v_Intercept``). You can see that the
overall drift rate intercept, here applying to WL condition, is positive
(mode value roughly 0.7), whereas the within subject effects of
condition (WW and LL) are negative and do not overlap with zero.

Fitting regression models
`````````````````````````

As mentioned above, cognitive neuroscience has embraced the DDM as it
enables to link psychological processes to cognitive brain measures. The
Cavanagh et al (2011) study is a great example of this. EEG recordings
provided a trial-ty-trial measure of brain activity (frontal theta), and
it was found that this activity correlated with increases in decision
threshold in high conflict trials. Note that the data set and results
exhibit more features than we consider here for the time being
(specifically the manipulation of deep brain stimulation), but for
illustrative purposes, we replicate here that main theta-threshold
relationship in a model restricted to participants without brain
stimulation. For more information, see
http://ski.clps.brown.edu/papers/Cavanagh\_DBSEEG.pdf

.. code:: python

    m_reg = hddm.HDDMRegressor(data[data.dbs == 0],
                               "a ~ theta:C(conf, Treatment('LC'))",
                               depends_on={'v': 'stim'},
                               group_only_regressors=True)

.. parsed-literal::

    Adding these covariates:
    ['a_Intercept', "a_theta:C(conf, Treatment('LC'))[HC]", "a_theta:C(conf, Treatment('LC'))[LC]"]


Instead of estimating one static threshold per subject across trials,
this model assumes the threshold to vary on each trial according to the
linear model specified above (as a function of their measured theta
activity). We also test whether this effect interacts with decision
conflict. For the stimuli we use dummy treatment coding with the
intercept being set on the WL condition. Internally, HDDM uses Patsy for
the linear model specification, see
https://patsy.readthedocs.org/en/latest/ for more details. The output
notifies us about the different variables that being estimated as part
of the linear model.

Finally, the keyword argument ``group_only_regressors`` instructs HDDM
to only use a group parameter for each regression coefficient. We use
this here for the regression coefficient only as it leads to much better
convergence than trying to estimate individual coefficients for each
subject separately. (We still allow individual subject nodes on all the
other parameters, including the threshold intercept, but due to the much
noisier theta measure, we treat that as a fixed effect within a group of
subjects. Nevertheless, this group parameter should be estimated
separately when the hypothesis tests the assumption that the
relationship between brain activity and threshold would differ between
different groups. Indeed the Cavanagh paper, and results shown later
below, illustrate that this brain/behavior relationship differs as a
function of whether patients are on or off STN deep brain stimulation,
as hypothesized by the model that STN is responsible for increasing the
decision threshold when cortical theta rises).

.. code:: python

    m_reg.sample(5000, burn=200)

.. code:: python

    theta = m_reg.nodes_db.node["a_theta:C(conf, Treatment('LC'))[HC]"]
    kabuki.analyze.plot_posterior_nodes([theta], bins=20)
    print "P(a_theta < 0) = ", (theta.trace() < 0).mean()

.. parsed-literal::

    P(a_theta < 0) =  0.0264583333333


.. image:: hddm_demo_files/hddm_demo_fig_08.png

The above posterior shows that the effect of trial to trial variations
in frontal theta are to increase the estimated decision threshold: the
regression coefficient is positive, and more than 96% of it is greater
than zero.

As noted above, this experiment also tested patients on deep brain
stimulation (dbs). The full model in the paper thus allowed an
additional factor to estimate how dbs interacts with theta-threshold
relationship. Here we show for illustrative purposes that we can capture
the same effect by simply fitting a separate model to data only
including the case when dbs was turned on. You should see below that in
this case, the influence of theta on threshold reverses. This exercise
thus shows that HDDM can be used both to assess the influence of
trial-by-trial brain measures on DDM parameters, but also how parameters
vary when brain state is manipulated.

.. code:: python

    m_reg_off = hddm.HDDMRegressor(data[data.dbs == 1],
                                   "a ~ theta:C(conf, Treatment('LC'))",
                                   depends_on={'v': 'stim'},
                                   group_only_regressors=True)

.. parsed-literal::

    Adding these covariates:
    ['a_Intercept', "a_theta:C(conf, Treatment('LC'))[HC]", "a_theta:C(conf, Treatment('LC'))[LC]"]


.. code:: python

    m_reg_off.sample(5000, burn=200)

.. code:: python

    theta = m_reg_off.nodes_db.node["a_theta:C(conf, Treatment('LC'))[HC]"]
    kabuki.analyze.plot_posterior_nodes([theta], bins=10)
    print "P(a_theta > 0) = ", (theta.trace() > 0).mean()

.. parsed-literal::

    P(a_theta > 0) =  0.0122916666667


.. image:: hddm_demo_files/hddm_demo_fig_09.png

Dealing with outliers
`````````````````````


It is common to have outliers in any data set and RT data is no
exception. Outliers present a serious challenge to likelihood-based
approaches, as used in HDDM. Consider the possibility that 5% of trials
are not generated by the DDM process, but by some other process (e.g.
due to an attentional lapse). The observed data in those trials may be
very unlikely given the best DDM parameters that fit 95% of the data. In
the extreme case, the likelihood of a single trial may be zero (e.g. if
subjects respond very quickly, faster than the non-decision time ``t``
parameter that would fit the rest of the data). Thus this single outlier
would force the DDM parameters to adjust substantially. To see the
effect of this we will generate data with outliers, but fit a standard
DDM model without taking outliers into account.

.. code:: python

    outlier_data, params = hddm.generate.gen_rand_data(params={'a': 2, 't': .4, 'v': .5},
                                                       size=200, n_fast_outliers=10)

.. code:: python

    m_no_outlier = hddm.HDDM(outlier_data)
    m_no_outlier.sample(2000, burn=50)

.. code:: python

    m_no_outlier.plot_posterior_predictive()

.. image:: hddm_demo_files/hddm_demo_fig_10.png

As you can see, the predictive likelihood does not fit the RT data very
well. The model predicts far more RTs near the leading edge of the
distribution than are actually observed. This is because non-decision
time ``t`` is forced to be estimated small enough to account for a few
fast RTs.

What we can do instead is fit a mixture model which assumes that
outliers come from a uniform distribution. (Note, outliers do not have
to be very fast or very slow, and the above example is just an obvious
illustration. Some proportion of the trials can be assumed to simply
come from a different process for which we make no assumptions about its
generation, and hence use a uniform distribution. This allows the model
to find the best DDM parameters that capture the majority of trials).
Here, we specify that we expect roughly 5% outliers in our data.

.. code:: python

    m_outlier = hddm.HDDM(outlier_data, p_outlier=.05)
    m_outlier.sample(2000, burn=20)

.. code:: python

    m_outlier.plot_posterior_predictive()

.. image:: hddm_demo_files/hddm_demo_fig_11.png

As you can see, the model provides a much better fit. The outlier RTs
are having less of an effect because they get assigned to the uniform
outlier distribution.
