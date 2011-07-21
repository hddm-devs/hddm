.. index:: Tutorial
.. _chap_tutorial_config_subjects:


***********************************
Creating a hierarchical group model
***********************************

Up until now, we have been looking at data that was generated from the
same set of parameters. However, in most experiments, we test multiple
subjects and may only gather relatively few trials per
subject. Traditionally, we would either fit a separate model to each
individual subject or fit one model to all subjects. Neither of these
approaches are ideal as we will see below. We can expect that subjects
will be similar in many ways yet have non-negligable individual
differences. If we fit separate models we ignore their similarities
and need much more data per subject to make useful inference. If we
fit one model to all subjects we ignore their differences and may get
a model fit that does not well characterize any individual. It would
be best if the model fitting could determine to what extent all
subjects are similar to the others in some respects, and use this
information as a prior to draw inferences about the paramters of any
given subject. The hierarchical approach optimally allocates the
information from the group vs the individual depending on the
statistics of the data.

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

As you can see, the estimates are far much worse (especially in the
hard condition) and the posterior distributions are much wider
indicating a lack of confidence in the estimates. Looking at the
posterior predictive and ill-shaped RT distribution makes obvious why
fitting a DDM to 10 trials is a fruitless attempt.

However, what about the data from the 39 other subjects? We certainly
wouldn't expect everyone to have the exact same parameters, but they
should be fairly similar. Couldn't we combine the data? This is where
the hierarchical approach becomes useful -- we can estimate individual
parameters, but at the same time have the parameters feed into a group
distribution so that the tiny bit we learn from each subject can be
pooled together and constrain the subject fits. Unfortunately, such a
model is a little bit more difficult to create in
general. Fortunately, HDDM does all of this automatically. Simply
running hddm_fit.py on a datafile that has a column named 'subj_idx'
will make HDDM create a hierarchical model where ach subject gets
assigned it's own distribution that is dependent on a group
distribution. Running this example produces the following output::







