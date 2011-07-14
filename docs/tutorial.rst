========
Tutorial
========

Specifiying Models via Configuration File
=========================================

The easiest way to use HDDM is to create a configuration file. First,
you have to prepare your data to be in a specific format (csv). A
possible (truncated) data file might look like this:

::

    rt,response,subj_idx,difficulty
    0.5,1.,1,easy
    1.2,0.,1,hard
    0.7,0.,2,easy
    2.3,1.,2,hard
    ...

The first line contains the header and specifies which columns contain
which data.

IMPORTANT: There must be one column named 'rt' and one named
'response'. If you want to build a group model (i.e. one with
hierarchical structure) you have to include column named 'subj_idx'.

The following the header contain the reaction time of the trial,
followed by a comma, followed by the response made (e.g. 1=correct,
0=error or 1=left, 0=right), followed by the ID of the subject and, in
our example case, a condition. Note, that columns that describe
conditions can be anything you like.

The following configuration file specifies a group-model in which
drift-rate depends on difficulty:

.. literalinclude :: ../hddm/examples/simple_difficulty.conf

The [model] tag specifies that parameters after the tag are model
parameters. In this case, the 'data' parameter tells HDDM where the
input data is to be found.

The optional tag [depends_on] specifies DDM parameters that depend on
data. In our case, we want to estimate separate drift-rates (v) for
the conditions found in the data column 'difficulty'.

The optional [mcmc] tag specifies parameter of the Markov chain
Monte-Carlo estimation such as how many samples to draw from the
posterior and how many samples to discard as burn-in (often, it takes
the MCMC chains some time to converge to the true posterior).

Our model specification is now complete and we can fit the model by
calling:

::

    hddmfit example.conf


Which will generate the following output:

::

    Creating model...
    Sampling: 100% [0000000000000000000000000000000000] Iterations: 5000

       name       mean   std    2.5q   25q    50q    75q    97.5  mc_err
    a         :  2.029  0.034  1.953  2.009  2.028  2.049  2.090  0.002
    t         :  0.297  0.007  0.282  0.292  0.297  0.302  0.311  0.001
    v('easy',):  0.992  0.051  0.902  0.953  0.987  1.028  1.102  0.003
    v('hard',):  0.522  0.049  0.429  0.485  0.514  0.561  0.612  0.002

    logp: -1171.276303
    DIC: 2329.069932

Because we used simulated data in this example, we know the true
parameters that generated the data (i.e. a=2, t=0.3, v_easy=1,
v_hard=0.5). As you can see, the mean posterior values are very close
to the true parameters -- our estimation worked! However, often we are
not only interested in the best fitting value but also how confident
we are in that estimation and how good other values are fitting. This
is one of advantages of the Bayesian approach -- it gives us the
complete posterior distribution. So the next columns are statistics on
the shape of the distribution, such as the standard deviation and
different quantiles to give you a feel for how certain you can be in
the estimates.

Lastly, logp and DIC give you a measure of how well the model fits the
data. These values are not all that useful if looked at in isolation
but they provide a tool to do model comparison. Logp is the summed
log-likelihood of the best-fitting values (higher is better). DIC
stands for deviance information criterion and is a measure that takes
model complexity into account. Lower values are better.

Excercise
+++++++++

Create a new model that ignores the different difficulties (i.e. only
estimate one drift-rate). Compare the resulting DIC score with that of
the previous model -- does the increased complexity of the first model
result in a sufficient increase in model fit to justify using it? Why
does the drift-rate estimate of the second model make sense?

Setting is_subj_model to True tells HDDM to look for the subj_idx
column in the data and create separate distributions for each subject
which feed into one group distribution. This feature is the main
advantage of HDDM compared to other tools to fit the DDM. Suppose you
have a dataset with many subjects, but with only very few trials for
each subject. You can't fit individual DDMs to each subject because
there aren't enough trials and you can't lump all their data together
because there will be individual differences which you are
ignoring. Hierarchical modeling offers a solution to this problem by
fitting parameters to individual parameters which are constrained by a
group distribution (more detail on that can be found in the section on
hierarchical bayesian modeling).


Specifiying Models in Python
============================

As an alternative to the configuration file, HDDM offers model
specification directly from Python. For this, you first import hddm:

.. literalinclude :: ../hddm/examples/simple_model.py
   :lines: 1

Next, we have to load the data into Python. HDDM expects a NumPy
structured array which you can either create yourself or load it from
a csv file. Information on how to create a proper structured NumPy
array can be found here. If you want to load a csv file make sure it
is in the proper format outlined above. You can then load the data as follows:

.. literalinclude :: ../hddm/examples/simple_model.py
   :lines: 4

After you loaded the data you can create the model object which is called Multi because it allows you to dynamically create multiple HDDM models depending on your data. In the simplest case, you'll want to create a simple DDM (default):

.. literalinclude :: ../hddm/examples/simple_model.py
   :lines: 7

You may then sample from the posterior distribution by calling:

.. literalinclude :: ../hddm/examples/simple_model.py
   :lines: 10

Depending on the model and amount of data this can take some time. After enough samples were generated, you may want to print some statistics on the screen:

.. literalinclude :: ../hddm/examples/simple_model.py
   :lines: 13

You can currently generate two plots to examine model fit. If you want to see if your chains converged and what the posteriors for each parameter look like you can call:

.. literalinclude :: ../hddm/examples/simple_model.py
   :lines: 16

To see how well the RT distributions are fit by the mean of the posterior distribution we can plot the theoretical RT distribution on top of our empirical RT distribution by calling:

.. literalinclude :: ../hddm/examples/simple_model.py
   :lines: 17

The closer the two distributions look like, the better the fit. Note
that the RT distribution for the second response is mirrored on the
y-axis.

The final program then looks as follows:

.. literalinclude :: ../hddm/examples/simple_model.py

More complex models can be generated by specifiying different
paremters during model creation. Say we wanted to create a model where
each subject receives it's own set of parameters which are themselves
sampled from a group parameter distribution, making use of the
hierarchical approach HDDM is taking. Morever, as in the example
above, we have two trial types in our data, easy and hard. Based on
previous research, we assume that difficulty affects drift-rate
'v'. Thus, we want to fit different drift rate parameters for those
two conditions while keeping the other parameters fixed across
conditions. Finally, we want to use the full DDM with inter-trial
variability for drift, non-decision time ter and starting point z. The
full model requires integration of these variability parameters. HDDM
implements two methods for this, monte-carlo sampling and full
bayesian integration. Here we will use monte-carlo integration because
full bayesian integration is extremely slow. The model creation and
sampling then might look like this (assuming we imported hddm and
loaded the data as above):

>>> model = hddm.HDDM(data, include=('V','Z','T'), depends_on={'v':'difficulty'})
>>> model.sample(10000, burn=5000)
