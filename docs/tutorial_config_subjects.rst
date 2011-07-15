.. index:: Tutorial
.. _chap_tutorial_config_subjects:


***********************************
Creating a hierarchical group model
***********************************

Up until now, we have been looking at data that was generated from the
same set of parameters. However, in most experiments, we test multiple
subjects and only gather a couple of trials. Traditionally, we would
either fit a separate model to each individual subject or fit one
model to all subjects. Neither of these approaches are ideal as we
will see below. We can expect that subjects will be similar in many
ways yet have non-neglicable individual differences. If we fit
separate models we ignore their similarties and need much more data
per subject to make useful inference. If we fit one model to all
subjects we ignore their differences and get worse fit.

To illustrate this point, consider the following example: we tested 30
subjects on the above task with the easy and hard condition. For
practical reasons, however, we only collected ten trials per
subject. As an example of what happens when trying to fit separate
models to each subject, we will run HDDM on the first subject. The
file simple_difficulty_subjs_single.csv only contains data from the
first subject. Lets run our model and see what happens:

::

    hddmfit simple_difficulty.conf simple_difficulty_subjs_single.csv

    Creating model...
    Sampling: 100% [000000000000000000000000000000] Iterations: 10000
       name       mean   std    2.5q   25q    50q    75q    97.5  mc_err
    a         :  1.571  0.219  1.202  1.418  1.552  1.708  2.039  0.012
    t         :  0.461  0.041  0.366  0.439  0.468  0.490  0.519  0.002
    v('easy',): -0.002  0.479 -0.980 -0.333  0.024  0.348  0.980  0.021
    v('hard',):  1.684  0.528  0.614  1.330  1.683  2.053  2.684  0.023

    logp: -17.105420
    DIC: 25.400104

