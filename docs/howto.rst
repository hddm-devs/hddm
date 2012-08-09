How-to
======

Code subject responses
----------------------

There are two ways you can code subject responses (these are the values
you put in the 'response' column in your data file). You can either use
accuracy-coding where 1 means correct and 0 means error, or you can
use direction-coding where 1 means left and 0 means right (this could
also code for stimulus A and B instead of left and right). HDDM
interprets 0 and 1 responses as lower and upper boundary responses,
respectively, so it has no preference one way or another.

In most cases it is more direct to use accuracy coding because
drift-rate will be directly associated with performance. However, if a
certain response direction or stimulus type has a higher probability
of being correct and you want to estimate bias, you can *not* use
accuracy coding (see the next paragraph for how to include a bias
parameter). Instead, you should use direction (or stimulus) coding and
estimate separate drift-rates for each condition (e.g. left response
correct vs. right response correct). A sanity check of whether you
coded the responses correctly is that drift-rate for left-response
correct trials (i.e. upper boundary; 1) is positive and drift-rate for
right-response correct trials (i.e. lower boundary; 0) is negative
(assuming subjects are above chance).

Include bias and inter-trial variability
----------------------------------------

Bias and inter-trial variability parameters are optional and can be
included as follows:

::

   model = hddm.HDDM(data, bias=True, include=('sv', 'st', 'sz'))

or:

::

   model = hddm.HDDM(data, include=('z', 'sv', 'st', 'sz'))

Where *sv* is inter-trial variability in drift-rate, *st* is inter-trial
variability in non-decision time and *sz* is inter-trial variability in
starting-point.

Note that you can also include a subset of parameters. This is
relevant because these parameter slow down sampling significantly. In
our experience, sv and st often play a bigger role but it really
depends on your dataset. I often run models without inter-trial
variabilities for exploratory analysis and then later play around with
different configurations. If a certain parameter is estimated very
close to zero or fails to converge you might want to exclude it (or
only include a group-node, see below).

There is also a convenience argument that is identical to the above.

::

   model = hddm.HDDM(data, bias=True, include='all')


Have separate parameters for different conditions using depends_on
--------------------------------------------------------------------

Most psychological experiments test how different conditions
(e.g. drug manipulations) affect certain parameters. You can build
arbitrarily complex models using the depends_on keyword.

::

   model = hddm.HDDM(data, depends_on={'a': 'drug', 'v': ['drug', 'difficulty']})

This will create model in which separate thresholds are estimated for
each drug condition and separate drift-rates for different drug
conditions and different levels of difficulty.

Note that this requires the columns 'drug' and 'difficulty' to be
present in your data array. For readability of which parameter coded
for what it is often useful to use string identifiers (e.g. drug:
off/on rather than drug: 0/1).

As you can see, single or multiple columns can supplied as values.

Assess model convergence
------------------------

When using MCMC sampling it is critical to make sure that our chains
have converged. This basically means that we are sampling from the
actual posterior. Unfortunately, there is no 100% fool-proof way to
assess whether our chains really converged. In reality, however, if
you follow a couple of steps to convince yourself you should be OK in
most cases.

Look at MC error statistic
""""""""""""""""""""""""""

When calling:

::

    model.print_stats()

There is a column called MC error. These values should not be smaller then 1%
of the posterior std. However, this is a very weak statistic and by no
means sufficient to assess convergence.


Visually inspect chains
""""""""""""""""""""""

The next thing to look at are the traces of the posteriors. You can
plot them by calling:

::

   model.plot_posteriors()

This will create a figure for each parameter in your model (which could
be a lot). Here is an example of what a not-converged chain looks
like:

.. figure:: not_converged_trace.png

and an example of what a converged chain looks like:

.. figure:: converged_trace.png

As you can see, there are striking differences. In the not-converged
case, the trace in the upper left corner is very non-stationary. There
are also certain periods where no jumps are performed and the chain is
stuck (horizontal lines in the trace); this is due to the proposal
distribution not being tuned correctly.

Secondly, the autocorrelation (lower left plot) is quite high as you
can see from the long tails of the distribution. This is further
indication that the samples are not independent draws from the
posterior. If the chain seems fine otherwise some autocorrelation must
not be a big deal. To leverage the problem, increase thinning (see
below).

Finally, the histogram (right plot) looks rather jagged in the
non-converged case. This is our approximation of the marginal
posterior distribution for this parameter. Generally, subject and
group mean posteriors are normal distributed (see the converged case)
while group variability posteriors are Gamma distributed.

Posterior predictive analysis
"""""""""""""""""""""""""""""

Another way to assess how good your model fits the data is to perform
posterior predictive analysis:

::

    model.plot_posterior_predictive()

.. TODO: ADD NICE PLOT

This will plot the posterior predictive in blue on top of the RT
histogram in red for each subject and each condition. Since we are
getting a distribution rather than a single parameter in our analysis,
the posterior predictive is the average likelihood evaluated over
different samples from the posterior. The width of the posterior
predictive in light blue corresponds to the standard deviation.


R-hat convergence statistic
"""""""""""""""""""""""""""

Another option to assess chain convergence is to compute the R-hat
(Gelman-Rubin) statistic. This requires multiple chains to be run. If
all chains converged to the same stationary distribution they should
be indistinguishable. The R-hat statistic compares between-chain
variance to within-chain variance. While the core functionality is in
place, an easy way to run multiple chains is currently not implemented
in HDDM but will be made available in the future.

What to do about lack of convergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the simplest case you just need to run a longer chain with more
burn-in and more thinning. E.g.:

::

    model.sample(50000, burn=45000, thin=5)

This will cause the first 45000 samples to be discarded. Of the
remaining 5000 samples only every 5th sample will be saved. Thus,
after sampling our trace will have a length of a 1000 samples.

You might also want to find a good starting point for running your
chains. This is commonly achieved by finding the maximum posterior
(MAP) via optimization. Before sampling, simply call:

::

    model.map()

which will set the starting values to the MAP. Then sample as you
would normally. This is a good idea in general.

If that still does not work you might want to consider simplifying
your model. Certain parameters are just notoriously slow to converge;
especially inter-trial variability parameters. The reason is that
often individual subjects do not provide enough information to
meaningfully estimate these parameters on a per-subject basis. One way
around this is to not even try to estimate individual subject
parameters and instead use only group nodes. This can be achieved via
the group_only_nodes keyword argument:

::

    model = hddm.HDDM(data, include=['sv', 'st'], group_only_nodes=['sv', 'st'])

The resulting model will still have subject nodes for all parameters
but sv and st.


Perform model comparison
------------------------

We can often come up with different viable hypotheses about which
parameters might be influenced by our experimental conditions. Above
you can see how you can create these different models using the
depends_on keyword. To compare which model does a better job at
explaining the data you can compare the DIC_ scores (lower is better)
emitted when calling:

::

    model.print_stats()

DIC, however, is far from being a perfect measure. So it shouldn't be your
only weapon in deciding which model is best.

Save and load models
--------------------

HDDM models can be saved and reloaded in a separate python
session. This is useful if your models need a lot of RAM or you are
running models on a cluster. Note that only the traces
(i.e. samples) get saved, you do have to recreate the model.

::

    # 1 load data and create a model
    data = hddm.load_csv('path_to_my_data')
    model = hddm.HDDM(data, bias=True)  # a very simple model...
    # 2 add commands for saving traces in a file
    model.mcmc(dbname='traces.db', db='pickle')
    # 3 run model. the traces will be saved in the file traces.db in the current working directory (alternatively specify path)
    model.sample(5000, burn=1000)


Now assume that you start a new python session, after the chain
started above is completed.

::

    #4 reconstruct your model
    data = hddm.load_csv('path_to_my_data')
    model = hddm.HDDM(data, bias=True)
    #5 add traces from database
    model.load_db('traces.db')  # not that for this to work you have to be in the same working directory you were in when you started the chain above. otherwise submit full path

    # now you can access the traces as you can when a chain has just completed
    # for example, you can access the contents of the chain for parameter v with
    # len(model.mc.trace("v")[:])

Under the hood, HDDM uses the database backend provided by PyMC. More
information on the types of backends and their properties can be found
in the `PyMC docs`_.


.. _PyMC docs: http://pymc-devs.github.com/pymc/database.html#saving-data-to-disk
.. _DIC: http://www.mrc-bsu.cam.ac.uk/bugs/winbugs/dicpage.shtml
