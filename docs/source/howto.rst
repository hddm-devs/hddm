******
How-to
******

Code subject responses
######################

There are two ways to code subject responses placed in the 'response'
column in your data file.  You can either use *accuracy-coding*, where
1's and 0's correspond to correct and error trials, or you can use
*stimulus-coding*, where 1's and 0's correspond to the choice
(e.g. categorization of the stimulus). HDDM interprets 0 and 1
responses as lower and upper boundary responses, respectively, so in
principle either of these schemes is valid.

In most cases it is more direct to use accuracy coding because the
sign and magnitude of estimated drift-rate will be directly associated
with performance (higher drift rate indicates greater likelihood of
terminating on the accurate boundary). However, if a certain response
direction or stimulus type has a higher probability of selection and
you want to estimate a response bias (which could be captured by a
change in starting point of the drift process; see below), you can
*not* use accuracy coding. (For example if a subject is more likely to
press the left button than the right button, but left and right
responses are equally often correct, one could not capture the
response bias with a starting point toward the incorrect boundary
because it would imply that those trials in which the left response
was correct would be associated with a bias toward the right
response). Thus stimulus coding should be used in this case, using the
HDDMStimCoding model. For this, add a column to your data that codes
which stimulus was correct and instantiate the model like this:

::

    model = hddm.HDDMStimCoding(data, include='z', stim_col='stim', split_param='v')

This model expects data to have a column named stim with two distinct
identifiers. For identifier 1, drift-rate v will be used while for
identifier 2, -v will be used. So ultimately you only estimate one
drift-rate. Alternatively you can use bias z and 1-z if you set
split_param='z'. See the HDDMStimCoding help doc for more information.

Stimulus coding can also be implemented in HDDMRegression. The advantage of doing so is that more complex designs, including within participant designs can be analysed. The disadvantage is the HRRMRegression is slower than HDDMstimcoding.

To implement stimulus coding for z one has to define a link function for hddm regression:
::
    import hddm
    import numpy as np
    from patsy import dmatrix

    def z_link_func(x, data=mydata):
        stim = (np.asarray(dmatrix('0 + C(s,[[1],[-1]])',{'s':data.stimulus.ix[x.index]})))    
        return 1 / (1 + np.exp(-(x * stim)))

Similarly, the link function for v is: 
::

    def v_link_func(x, data=mydata):
        stim = (np.asarray(dmatrix('0 + C(s,[[1],[-1]])',{'s':data.stimulus.ix[x.index]})))    
        return x * stim

To specify a complete model you have to define a complete regression model and submit with the data to the model.
::

    z_reg = {'model': 'z ~ 1 + C(condition)', 'link_func': z_link_func}
    v_reg = {'model': 'v ~ 1 + C(condition)', 'link_func': lambda x : x}
    reg_descr = [z_reg, v_reg]
    hddm_regrssion_model = hddm.HDDMRegressor( data, reg_descr,include='z')

Of course, your model could also regress either z or v.


Include bias and inter-trial variability
########################################

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

There is also a convenience argument that is identical to the above.

::

   model = hddm.HDDM(data, bias=True, include='all')

Note that you can also include a subset of parameters. This is
relevant because these parameters slow down sampling significantly. If
a certain parameter is estimated very close to zero or fails to
converge (which can happen with the sv parameter) you might want to
exclude it (or only include a group-node, see below). Finally,
parameter recovery studies show that it requires a lot of trials to
get meaningful estimates of these parameters.


Estimate parameters for different conditions
############################################

Most psychological experiments test how different conditions
(e.g. drug manipulations) affect certain parameters. You can build
arbitrarily complex models using the depends_on keyword.

::

   model = hddm.HDDM(data, depends_on={'a': 'drug', 'v': ['drug', 'difficulty']})

This will create model in which separate thresholds are estimated for
each drug condition and separate drift-rates for different drug
conditions and levels of difficulty.

Note that this requires the columns 'drug' and 'difficulty' to be
present in your data array. For readability it is often useful to use
string identifiers (e.g. drug: off/on rather than drug: 0/1).

As you can see, single or multiple columns can be supplied as values.


Outliers
########

The presence of outliers is notoriously challenging for likelihood
models, because the likelihood of a few outliers given the generative
model cab be quite low. In practice, even the model we have is
reasonable for a majority of trials, it may be that data from a
minority of trials is not well described by this model (e.g. due to
attentional lapses).  HDDM 0.4 (and upwards) supports estimation of a
mixture model that enables stable parameter estimation even with
outliers present in the data. You can either specify a fixed
probability for obtaining an outlier (e.g. 0.05 will assume 5% of the
RTs are outliers) or estimate this from the data. In practice, the
precise value of p_outlier does not matter.  Values greater than 0.001
and less than 0.1 are sufficient to capture the outliers, and the
effect on the recovered parameters is small (Sofer et al, in
preparation).

To instantiate a model with a fixed probability of getting
an outlier run:

::

    m = hddm.HDDM(data, p_outlier=0.05)

To estimate p_outlier from the data, run:

::

    m = hddm.HDDM(data, include=('p_outlier',))

HDDM assumes that outliers come from a uniform distribution
with a fixed density :math:`w_{outlier}` (as suggested by Ratcliff and Tuerlinckx, 2002).
The resulting likelihood is as follows:

.. math::

   p(RT; v, a, t) = wfpt(RT; v, a, t) * (1-p_{outlier}) + w_{outlier} * p_{outlier}

The default value of :math:`w_{outlier}` is 0.1, which is equivalent to uniform distribution
from 0 to 5 seconds. However, in practice, the outlier model is applied to all RTs, even
those  larger than 5.


Assess model convergence
########################

When using MCMC sampling it is critical to make sure that our chains
have converged, to ensure that we are sampling from the actual
posterior distribution. Unfortunately, there is no 100% fool-proof way to
assess whether chains converged. However, there are various metrics in
the MCMC literature to evaluate convergence problems, and if
you follow some simple steps you can be more confident.

Look at MC error statistic
**************************

When calling:

::

    model.print_stats()

There is a column called MC error. These values should not be smaller then 1%
of the posterior std. However, this is a very weak statistic and by no
means sufficient to assess convergence.


Geweke statistic
****************

The Geweke statistic is a time-series approach that compares the mean
and variance of segments from the beginning and end of a single
chain. You can test your model by running:

::

    from kabuki.analyze import check_geweke
    print check_geweke(model)

This will print `True` if non of the test-statistics is larger than 2
and `False` otherwise. Check the `PyMC documentation` for more
information on this test.


Visually inspect chains
***********************

The next thing to look at are the traces of the posteriors. You can
plot them by calling:

::

   model.plot_posteriors()

This will create a figure for each parameter in your model. Here is an example of what a not-converged chain looks
like:

.. figure:: not_converged_trace.png

and an example of what a converged chain looks like:

.. figure:: converged_trace.png

As you can see, there are striking differences. In the not-converged
case, the trace in the upper left corner is very non-stationary. There
are also certain periods where no jumps are performed and the chain is
stuck (horizontal lines in the trace); this is due to the proposal
distribution not being tuned correctly.

Secondly, the auto-correlation (lower left plot) is quite high as you
can see from the long tails of the distribution. This is a further
indication that the samples are not independent draws from the
posterior.

Finally, the histogram (right plot) looks rather jagged in the
non-converged case. This is our approximation of the marginal
posterior distribution for this parameter. Generally, subject and
group mean posteriors are normal distributed (see the converged case)
while group variability posteriors are Gamma distributed.

Posterior predictive analysis
*****************************

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
***************************

Another option to assess chain convergence is to compute the R-hat
(Gelman-Rubin) statistic. This requires multiple chains to be run. If
all chains converged to the same stationary distribution they should
be indistinguishable. The R-hat statistic compares between-chain
variance to within-chain variance.

To compute the R-hat statistic in kabuki you have to run
multiple copies of your model:

::

   from kabuki.analyze import gelman_rubin

   models = []
   for i in range(5):
       m = hddm.HDDM(data)
       m.map()
       m.sample(5000, burn=1000)
       models.append(m)

   gelman_rubin(models)

The output is a dictionary that provides the R-hat for each parameter:

::

   {'a_trans': 1.0028806196268818,
   't_trans': 1.0100017175108695,
   'v': 1.0232548747719443}


As of HDDM 0.4.1 you can also run multiple chains in parallel. One
convenient way to do this is the IPython parallel module. Note that
you do you have to set up your environment appropriately for this, see the `IPython parallel docs`.

::

   def run_model(id):
       import hddm
       data = hddm.load_csv('mydata.csv')
       m = hddm.HDDM(data)
       m.find_starting_values()
       m.sample(20000, burn=15000, dbname='db%i'%id, db='pickle')
       return m

   from IPython.parallel import Client
   v = Client(profile='hddm')[:]
   jobs = v.map(run_model, range(4))
   models = jobs.get()
   gelman_rubin(models)


What to do about lack of convergence
************************************

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

Estimate a regression model
###########################

HDDM 0.4 (and upwards) includes a regression model that allows
estimation of trial-by-trial influences of a covariate (e.g. a brain
measure like fMRI) onto DDM parameters. For example, if your
prediction is that activity of a particular brain area has a linear
correlation with drift-rate, you could specify the following
regression model (make sure to have a column with the brain activity
in your data, in our example name this column 'BOLD'):

::

   # Define regression function (linear in this case)
   reg_func = lambda args, cols: args[0] + args[1]*cols[:,0]

   # Define regression descriptor
   # regression function to use (func, defined above)
   # args: parameter names (passed to reg_func; v_slope->args[0],
   #                                            v_inter->args[1])
   # covariates: data column to use as the covariate
   #             (in this example, expects a column named
   #             BOLD in the data)
   # outcome: DDM parameter that will be replaced by trial-by-trial
   #          regressor values (drift-rate v in this case)
   reg = {'func': reg_func,
          'args': ['v_inter','v_slope'],
          'covariates': 'BOLD',
          'outcome': 'v'}

   # construct regression model. Second argument must be the
   # regression descriptor. This model will have new parameters defined
   # in args above, these can be used in depends_on like any other
   # parameter.
   m = hddm.HDDMRegressor(data, reg, depends_on={'v_slope':'trial_type'})

Note that in the last line, the regression coefficients become ordinary
model parameters you can use in depends_on.

You can also pass a list to covariates if you want to include multiple
covariates. E.g.:

::

   # Define regression function with interaction with exponential
   # transform

   reg_func = lambda args, cols: np.exp(args[0] + args[1]*cols[:,0] + args[2]*cols[:,1] + args[3]*cols[:,0]*cols[:,1])

   reg = {'func': reg_func,
          'args': ['a_intercept','a_slope_cov1', 'a_slope_cov2', 'a_interaction'],
          'covariates': 'BOLD',
          'outcome': 'a'}

Note that these regression coefficients are often hard to estimate and
require a lot of data. If you have problems with chain convergence,
consider turning the coefficients into group_only_nodes (see above).

If you want to estimate two separate regressions, you can also supply
a list of regression descriptors to HDDMRegressor:

::

    m = hddm.HDDMRegressor(data, [reg_a, reg_t])

Make sure to give all regression coefficients different names.



Perform model comparison
########################

We can often come up with different viable hypotheses about which
parameters might be influenced by our experimental conditions. Above
you can see how you can create these different models using the
depends_on keyword.

DIC
***

To compare which model does a better job at explaining the data you
can compare the DIC_ scores (lower is better) emitted when calling:

::

    model.print_stats()

DIC, however, is far from being a perfect measure. So it shouldn't be your
only weapon in deciding which model is best.

Posterior predictive check
**************************

A very elegant method to compare models is to sample new data sets
from the estimated model and see how well these simulated data sets
corresponds to the actual data on some measurement (e.g. is the mean
RT well recovered by this model?). This test is called posterior
predictive check and you can run it like this:

::

   from hddm.utils import post_pred_check
   post_pred_check(model)

This will return a table of statistics which might look like this:

::

		   observed  credible   quantile       SEM  mahalanobis      mean       std      2.5q       25q       50q       75q     97.5q  NaN
    node stat
    wfpt std_ub    0.353652         1  49.298597  0.000647     0.153912  0.379096  0.165319  0.120420  0.265707  0.354912  0.465269  0.778341    1
	 mean_lb  -0.958116         1  58.200000  0.000400     0.205017 -0.978110  0.097522 -1.206278 -1.030025 -0.971118 -0.911902 -0.811491    0
	 mean_ub   0.958336         1  51.703407  0.000216     0.090950  0.973042  0.161691  0.699320  0.859808  0.949264  1.067915  1.333156    1
	 accuracy  0.200000         1  55.700000  0.000005     0.029034  0.197720  0.078529  0.060000  0.140000  0.180000  0.240000  0.380000    0

The rows correspond to the different observed nodes and summary
statistics that the model was evaluated on (e.g. mean_lb which represents the mean RT of lower boundary responses)). The columns correspond to the
statistics of how the corresponding summary statistic of the real data
relates to the simulated data sets. E.g. `wfpt`, `accuracy`, `Observed`
represents the accuracy of the observed data. `Quantile` represents in
which quantile this mean RT is in the mean RT taken over the simulate
data sets. If our model did a great job at recovering we wanted it to
produce RTs that have the same mean as our actual data. So the closer
this is to the 50th quantile the better.


Save and load models
####################

HDDM models can be saved and reloaded in a separate python
session. Note that you have to save the traces to file by using
the db backend.

::

    model = hddm.HDDM(data, bias=True)  # a very simple model...
    model.sample(5000, burn=1000, dbname='traces.db', db='pickle')
    model.save('mymodel')

Now assume that you start a new python session, after the chain
started above is completed.

::

   model = hddm.load('mymodel')

HDDM uses the pickle module to save and load models.

Stimulus coding with HDDM regression
####################################
In some situations it is useful to fix the magnitude of parameters across stimulus types while also forcing them to have different directions. For example, an independent variable could influences both the drift rate v and the response bias z. A specific example is an experiment on face house discrimination with different difficulty levels, where the drift rate is smaller when the task is more difficult and where the bias to responding house is larger when the task is more difficult.
One way to analyze the effect of difficulty on drift rate and bias in such an experiment is to estimate one drift rate v for each level, and a response bias z such that the bias  for houses-stimuli is z and the bias for face stimuli is 1-z (z = .5 for unbiased decisions in HDDM).
The following example describes how to generate simulated data for such an experiment, how to set up the analysis with HDDMRegression,  and compares true parameter values with those estimated with HDDMRegression.

Model Recovery Test for HDDMRegression
**************************************
The test is performed with simulated data for an experiment with one independent variable with three levels (e.g. three levels of difficulty) which influence both drift rate v and bias z. Responses are "accuracy coded", i.e. correct responses are coded 1 and incorrect responses 0. Further, stimulus coding of the parameter z is implemented. "stimulus coding" of z means that we want to fit a model in which the magnitude of the bias is the same for the two stimuli, but its direction "depends on" the presented stimulus (e.g. faces or house in a face-house discrimination task). Note that this does not mean that we assume that  decision makers adjust their bias after having seen the stimulus. Rather, we want to measure response-bias (in favor of face or house) while assuming the same drift rate for both stimuli. We can achieve this for accuracy coded data by modeling  the bias as moved towards the correct response boundary for one stimulus (e.g. z = .6 for houses) and away from the correct response  boundary for the other stimulus (1-z = .4 for faces).
import python modules
::

    import hddm
    from patsy import dmatrix  # for generation of (regression) design matrices
    import numpy as np         # for basic matrix operations
    from pandas import Series  # to manipulate data-frames generated by hddm
We save the output of stdout to the file 'ModelRecoveryOutput.txt'.
::

    import sys
    sys.stdout = open('ModelRecoveryOutput.txt', 'w')

Creating simulated data for the experiment
******************************************
First we set the number of subject and the number of trials per level for the simulated experiment
::

    n_subjects = 10
    trials_per_level = 150 # and per stimulus

Next we set up parameters of the drift diffusion process for the three levels and the first stimulus. As desribed earlier v and z change accross levels
::

    level1a = {'v':.3, 'a':2, 't':.3,'sv':0,'z':.5,'sz':0,'st':0}
    level2a = {'v':.4, 'a':2, 't':.3,'sv':0,'z':.6,'sz':0,'st':0}
    level3a = {'v':.5, 'a':2, 't':.3,'sv':0,'z':.7,'sz':0,'st':0}

Now we generate the data for stimulus A
::

    data_a, params_a = hddm.generate.gen_rand_data({'level1': level1a,'level2': level2a, 'level3': level3a},size=trials_per_level, subjs=n_subjects)
Next come the parameters for the second stimulus, where v is the same as for the first stimulus. This is different for z. In particular: z(stimulus_b) = 1 - z(stimulus_a). As a result, responses are altogether biased towards responding A. Because we use accuracy coded data, stimulus A is biased towards correct responses, and stimulus B towards incorrect responses. 
::

    level1b = {'v':.3, 'a':2, 't':.3,'sv':0,'z':.5,'sz':0,'st':0}
    level2b = {'v':.4, 'a':2, 't':.3,'sv':0,'z':.4,'sz':0,'st':0}
    level3b = {'v':.5, 'a':2, 't':.3,'sv':0,'z':.3,'sz':0,'st':0}

Now we generate the data for stimulus B
::

    data_b, params_b = hddm.generate.gen_rand_data({'level1': level1b,'level2': level2b, 'level3': level3b},size=trials_per_level, subjs=n_subjects)

We add a column to the data-frame identifying stimulus A as 1 and stimulus B as 2.
::

    data_a['stimulus']= Series(np.ones((len(data_a))), index=data_a.index)
    data_b['stimulus']= Series(np.ones((len(data_b)))*2, index=data_a.index)

Now we merge the data for stimulus A and B
::

    mydata = data_a
    mydata.append(data_b)

Setting up the HDDM regression model
************************************
The parameter z is bound between 0 and 1, but the standard linear regression does not generate values between 0 and 1. Therefore we use a link-function, here the inverse logit 1/(1+exp(-x)),which transforms values between plus and minus infinity into values ranging from (just above) 0 to (nearly) 1. [If this reminds you of link functions for logistic regressions, that’s correct].
Next we need to insure that the bias is z for one stimulus and 1-z for the other stimulus. To achieve this, we can simply multiply the regression output for one stimulus with -1. This is implemented here by dot-multiplying the regression output "x" (which is an array) with equally sized array "stim", which is 1 for all stimulus A trials and -1 for stimulus B trials. We use the patsy command dmatrix to generate such an array from the stimulus column of our simulated data
::

    def z_link_func(x, data=mydata):
        stim = (np.asarray(dmatrix('0 + C(s,[[1],[-1]])',{'s':data.stimulus.ix[x.index]})))    
        return 1 / (1 + np.exp(-(x * stim)))

Now we set up the regression models for z and v and also include the link functions The relevant string here used by patsy is '1 + C(condition)'. This will generate a design matrix with an intercept (that's what the '1' is for) and two dummy variables for remaining levels. (The column in which the levels are coded has the default name 'condition')
::

    z_reg = {'model': 'z ~ 1 + C(condition)', 'link_func': z_link_func}

For v the link function is simply x = x, because no transformations is needed. [However, you could also analyze this experiment with response coded data. Then you would not stimulus code z but v and you would have to multiply the v for one condition with -1, with a link function like the one for z above, but with out the additional logit transform ]
::

    v_reg = {'model': 'v ~ 1 + C(condition)', 'link_func': lambda x : x}

Now we can finally put the regression description for the hddm model together. The general for is [{'model': 'outcome_parameter ~ patsy_design_string', 'link_func': your_link_function }, {...}, ...]
::

    reg_descr = [z_reg, v_reg]

The last step before running the model is to construct the complete hddm regression model by adding data etc.
::

    m_reg = hddm.HDDMRegressor( mydata, reg_descr,include='z')

Now we start the model, and wait for a while (you can go and get several coffees, or read a paper) (Sampling 20000 samples for the example experiment described here took 77 minutes on a macbook pro with a 2.66 GHz Intel Core i7. (for a real experiment with data that are certainly noisier than the simulated data one should sample ca 10 times as many samples)
::

    m_reg.sample(20000,burn = 15000)

Comparing generative and recovered model parameters
***************************************************
First we print the model stats
::

    m_reg.print_stats() 

Here is the relevent output for our purposes:

parameter			mean       std      2.5q       25q       50q       75q     97.5q    mc err 

z_Intercept			-0.044598  0.148731 -0.348728 -0.141392 -0.045055  0.046041  0.271227  0.005647 

z_C(condition)[T.level2]	0.395524  0.049708  0.304394  0.354014  0.402072  0.426116  0.496143  0.004200 

z_C(condition)[T.level3]	0.818458  0.049148  0.712337  0.788209  0.820972  0.850570  0.903171  0.003559 

v_Intercept			0.269770  0.058421  0.151004  0.237380  0.271991  0.303675  0.380508  0.003125 

v_C(condition)[T.level2]	0.159221  0.051821  0.065206  0.123976  0.157030  0.192976  0.271688  0.004290 

v_C(condition)[T.level3]	0.250912  0.059487  0.152756  0.203228  0.251347  0.290904  0.373658  0.004719

Lets first look at v. For level1 this is just the intercept. The value of .27 is in the ball park of the true value of .3. The fit is not perfect, but running a longer chain might help (we are ignoring sophisticated checks of model convergence for this example here). To get the values of v for levels 2 and 3, we have to add the respective parameters (0.16 and .25) to the intercept value. The resulting values of .43 and .52 are again close enough to the true values of .4 and .5. To get the estimated z value we first need to "convert" the regression value with our link function. For level 1 this is 1/(1+exp(-(-0.044))) = .48, which is close to the true value of .5. For level 2 this is 1/(1+exp(-(-0.044+0.396))) = .59, again cloe to the true value of .6, as is the case for level 3 (.68 vs. .7).
In sum, HDDMRegression easily recovered the right order of the parameters z. The recovered parameter values are also close to the true parameter values. The deviations show that (a) we should maybe run longer mcmc chains and, more importantly, (b) that for the relatively small differences in DDM parameters we tested here a larger experiment (i.e. more trials per conditions or more participants) would be better.



.. _PyMC docs: http://pymc-devs.github.com/pymc/database.html#saving-data-to-disk
.. _DIC: http://www.mrc-bsu.cam.ac.uk/bugs/winbugs/dicpage.shtml
.. _PyMC documentation: http://pymc-devs.github.com/pymc/modelchecking.html#formal-methods
.. _IPython Parallel Docs: http://ipython.org/ipython-doc/stable/parallel/index.html
