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

To illustrate this point, consider the following example: we tested 30
subjects on the above task with the easy and hard condition. For
practical reasons, however, we only collected ten trials per
subject. As an example of what happens when trying to fit separate
models to each subject, we will run HDDM on the first subject. The
file simple_difficulty_subjs_single.csv only contains data from the
first subject. Lets run our model and see what happens:

::

    hddm_fit.py simple_difficulty.conf simple_difficulty_subjs_single.csv

    Creating model...
    Sampling: 100% [000000000000000000000000000000] Iterations: 10000
       name       mean   std    2.5q   25q    50q    75q    97.5  mc_err
    a         :  1.571  0.219  1.202  1.418  1.552  1.708  2.039  0.012
    t         :  0.461  0.041  0.366  0.439  0.468  0.490  0.519  0.002
    v('easy',): -0.002  0.479 -0.980 -0.333  0.024  0.348  0.980  0.021
    v('hard',):  1.684  0.528  0.614  1.330  1.683  2.053  2.684  0.023

    logp: -17.105420
    DIC: 25.400104

As you can see, the estimates are far from the true values and the
posterior distributions are much wider indicating a lack of confidence
in the estimates. Looking at the posterior predictive and ill-shaped
RT distribution makes obvious why fitting a DDM to 10 trials is a
fruitless attempt.

.. figure::

