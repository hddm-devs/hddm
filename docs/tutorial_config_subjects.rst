.. index:: Tutorial
.. _chap_tutorial_config_subjects:


Creating a hierarchical group model
###################################

Up until now, we have been looking at data that was generated from the
same set of parameters. However, in most experiments, we test multiple
subjects and only gather relatively few trials per subject; this is
often the case in cognitive neuroscience experiments where we may also
image subjects using fMRI during the task, or collect data from
patient populations. Traditionally, we would
either fit a separate model to each individual subject or fit one
model to all subjects. Neither of these approaches are ideal as we
will see below. We can expect that subjects will be similar in many
ways. If we fit separate models we ignore their similarities and need
much more data per subject to make useful inference. If we fit one
model to all subjects we ignore their differences and may get a model
fit that does not well characterize any individual. Ideally, our model
estimation would capture both individual differences and
similarities. The hierarchical Bayesian approach used by HDDM
optimally allocates the information from the group vs the individual
depending on the statistics of the data.

To illustrate this point, consider the following example: we tested 40
subjects on the above task with the easy and hard condition. For
practical reasons, however, we only collected 20 trials per
condition. As an example of what happens when trying to fit separate
models to each subject, we will run HDDM on the first subject. The
file simple_difficulty_subjs_single.csv only contains data from the
first subject. Lets run our model and see what happens:

::

    hddm_fit.py simple_difficulty.conf simple_subjs_difficulty_single.csv

    Creating model...
    Sampling: 100% [0000000000000000000000000000000000] Iterations: 10000
       name       mean   std    2.5q   25q    50q    75q    97.5  mc_err
    a         :  1.936  0.152  1.665  1.830  1.926  2.030  2.259  0.007
    t         :  0.314  0.044  0.214  0.288  0.318  0.347  0.386  0.002
    v('easy',):  0.468  0.248 -0.017  0.298  0.473  0.642  0.929  0.009
    v('hard',):  0.426  0.234 -0.028  0.268  0.428  0.571  0.869  0.008

    logp: -60.214194
    DIC: 113.099890

As you can see, the estimates are far worse (especially in the hard
condition) and the posterior distributions are much wider indicating a
lack of confidence in the estimates. Looking at the posterior
predictive and the ill-shaped RT distribution makes clear that fitting a
DDM to 20 trials is a fruitless attempt.

However, what about the data from the 39 other subjects? We certainly
wouldn't expect everyone to have the exact same parameters, but they
should be fairly similar. Couldn't we combine the data? This is where
the hierarchical approach becomes useful -- we can estimate individual
parameters, but at the same time have the parameters feed into a group
distribution so that the tiny bit we learn from each subject can be
pooled together and constrain the subject fits. Unfortunately,
hierarchical models are more complex in many ways. Fortunately, HDDM
automates this process and reduces this complexity. Simply running
hddm_fit.py on a data file that has a column named 'subj_idx' will
make HDDM create a hierarchical model where each subject gets assigned
its own subject distribution which is assumed to be distributed around
a normal group distribution. Running this example produces the
following output (omitting some lines):

::

    >>hddm_fit.py simple_difficulty.conf simple_subjs_difficulty.csv
    Creating model...
    Sampling: 100% [00000000000000000000000000000000] Iterations: 20000

         name        mean   std    2.5q   25q    50q    75q    97.5  mc_err
    a            :  2.000  0.026  1.950  1.985  1.996  2.018  2.059  0.002
    a0           :  2.002  0.053  1.889  1.977  1.996  2.032  2.120  0.003
    a8           :  1.989  0.051  1.879  1.967  1.990  2.015  2.089  0.002
    a9           :  1.984  0.055  1.847  1.964  1.990  2.014  2.083  0.003
    avar         :  0.042  0.030  0.005  0.016  0.038  0.065  0.103  0.003
    t            :  0.305  0.006  0.293  0.302  0.305  0.309  0.316  0.000
    t0           :  0.308  0.012  0.286  0.301  0.307  0.314  0.337  0.001
    t1           :  0.304  0.012  0.279  0.298  0.305  0.311  0.328  0.000
    t9           :  0.299  0.012  0.269  0.294  0.301  0.307  0.318  0.001
    tvar         :  0.009  0.006  0.002  0.004  0.008  0.013  0.025  0.001
    v('easy',)   :  0.966  0.050  0.863  0.934  0.969  0.998  1.057  0.004
    v('easy',)0  :  0.970  0.092  0.786  0.920  0.971  1.017  1.190  0.004
    v('easy',)1  :  0.983  0.091  0.810  0.929  0.983  1.031  1.180  0.005
    v('easy',)5  :  0.965  0.087  0.788  0.915  0.969  1.013  1.152  0.004
    v('easy',)9  :  0.969  0.089  0.780  0.918  0.972  1.021  1.148  0.004
    v('hard',)   :  0.401  0.045  0.317  0.370  0.400  0.430  0.500  0.003
    v('hard',)0  :  0.488  0.138  0.278  0.387  0.464  0.565  0.803  0.007
    v('hard',)9  :  0.476  0.133  0.264  0.383  0.456  0.553  0.796  0.007
    vvar('easy',):  0.111  0.053  0.030  0.068  0.107  0.145  0.221  0.004
    vvar('hard',):  0.070  0.047  0.017  0.033  0.058  0.097  0.185  0.004

    logp: -1161.693344
    DIC: 2882.445038

The first you can see when examining the recovered parameter values is
that the mean of the group distributions (i.e. a, t, v('hard',) and
v('easy',)) is that they match very well the parameters  used to
generate the data. So by pooling the very sparse data we had on
each subject we can make useful inference about the group parameters.

The second thing you can see is that individual subject parameters
(ending with the index of the subject) are very close to the group
mean (this is also indicated by small group variance -- representing
the spread of the individual subject parameters). This property is
called *shrinkage*. Intuitively, if we can not make meaningful
inference about individual subjects we will assume that they are
distributed as the rest of the group, particularly if the overall data
set is well captured by little variance in subject parameters. The more data we have the less
individual subject estimates will be shrinked to the group mean.

While creating a configuration file and calling hddm_fit.py is quite
easy, this approach is also quite limited. Thus, if you want to build
more sophisticated models or do more advanced analysis, you will have
to use HDDM from Python. Building your own model in Python will be
explored in :ref:`part three of the tutorial <chap_tutorial_python>`.
