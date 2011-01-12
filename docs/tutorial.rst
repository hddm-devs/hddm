========
Tutorial
========

Specifiying Models via Configuration File
=========================================

The simplest model
------------------

The easiest way to use HDDM is to create a simple configuration
file. In our first example, we will consider the easiest usage case
where we will not use subject information or different treatment
influences.

First, you have to prepare your data to be in a specific format
(csv). A possible data file might look like this:

::

    rt,response
    0.5,1.
    1.2,0.
    0.7,0.
    2.3,1.

The first line contains the header and specifies which columns contain
which data.

IMPORTANT: There must be one column named 'rt' and one named 'response'.

The following lines contain the reaction time of the trial, followed
by a comma, followed by the response made (e.g. 1=correct, 0=error or
1=left, 0=right).

Next we will have a look at how the simplest configuration file might look like.

::

	[data]
	load = example.csv # Main data file containing all RTs in csv format
	save = example_out.txt # Estimated parameters and stats will be written to this file

The [data] tag specifies that the parameters after the tag set input
and output variables. In this case, HDDM will load the file
example.csv and write its output statistics and parameter fits to
example_out.txt which you can then analyze.

Our model specification is now complete and we can fit the model by calling:

::

	hddmfit example.conf

Depending on the amount of your data, the complexity and type of the
model used (since we did not specifiy it in this case, HDDM chose the
simple DDM as a default).

After parameter estimation is done we can examine the output file (example_out.txt):

::

	Model type: simple

	General model stats:
      	logp: -106.555642
      	dic: 286.293556

      	Group parameter			Mean		Std
      	a				2.0		0.032740
      	t				0.3		0.003501
      	v				0.5		0.019900


Example Subject Model
---------------------

Lets create a more interesting model. Say we have a dataset where
multiple subjects were tested on three conditions of the moving dot
task. In this task, subjects view randomly moving dots on a screen and
have to decide if more dots are moving to one side or the
other. Depending on the number of coherently moving dots, this can be
harder or easier. In our example, we have two conditions: easy and
hard. The dataset looks like this:

::
	
	rt, response, subj_idx, difficulty
	0.73, 1., 1., 'easy'
	1.35, 0., 1., 'hard'
	1.23, 1., 2., 'easy'
	...

The subj_idx column specifies the index number of the subject. This column must be named 'subj_idx'.

Our configuration file then becomes:

.. literalinclude :: ../hddm/examples/simple_subjs_difficulty.conf

Under the [model] tag we can specify the type of model. Here, we want
to fit a full DDM using Monte-Carlo integration (which is the
fastest). 

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

Under the [depends] tag you can set that certain DDM parameters are
fit to only a subset of the data provided by the column. In this,
drift-rate will depend on task difficulty and HDDM will look for a
'difficulty' column in the data.

After calling HDDM this is what the output looks like:

.. literalinclude :: ../hddm/examples/simple_subj_difficulty.txt


Specifiying Models in Python
============================

As an alternative to the configuration file, HDDM offers model
specification directly from Python. For this, you first import hddm:

>>> import hddm

Next, we have to load the data into Python. HDDM expects a NumPy
structured array which you can either create yourself or load it from
a csv file. Information on how to create a proper structured NumPy
array can be found here. If you want to load a csv file make sure it
is in the proper format outlined above. You can then load the data as follows:

>>> data = hddm.utils.csv2rec('yourdata.csv')

After you loaded the data you can create the model object which is called Multi because it allows you to dynamically create multiple HDDM models depending on your data. In the simplest case, you'll want to create a simple DDM (default):

>>> model = hddm.models.Multi(data)

You may then sample from the posterior distribution by calling:

>>> model.mcmc()

Depending on the model and amount of data this can take some time. After enough samples were generated, you may want to print some statistics on the screen:

>>> print model.summary()

You can currently generate two plots to examine model fit. If you want to see if your chains converged and what the posteriors for each parameter look like you can call:

>>> model.plot_posteriors()

To see how well the RT distributions are fit by the mean of the posterior distribution we can plot the theoretical RT distribution on top of our empirical RT distribution by calling:

>>> model.plot_RT_fit()

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

>>> model = hddm.models.Multi(data, is_subj_model=True, model_type='full_mcmc', depends_on={'v':'difficulty'})
>>> model.mcmc(samples=10000, burn=5000)
