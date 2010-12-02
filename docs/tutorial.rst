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

	hddm example.conf

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

::

     [data]
     load = example_subjs_v.csv
     save = example_subjs_v_out.txt

     [model]
     type = simple
     is_subj_model = True

     [depends]
     v = difficulty    

     [mcmc]
     samples=5000
     burn=2000
     thin=3

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

::

    Model type: simple
    DDM parameter "v" depends on: difficulty

    General model stats:
    logp: -2357.518872
    dic: 4911.638098

    Group parameter			Mean		Std
    a	  				2.295061	0.149153
    v_('easy',)				0.510279	0.015938
    t					0.191997	0.089777
    v_('hard',)				0.962078	0.022484
    v					0.750207	0.208061


Specifiying Models in Python
============================
