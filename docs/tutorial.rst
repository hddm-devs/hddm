========
Tutorial
========

Specifiying Models via Configuration File
=========================================

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





Specifiying Models in Python
============================
