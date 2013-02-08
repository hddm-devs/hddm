============
Introduction
============

*NOTE*: This document is still under development.  Sequential sampling
models (SSMs) (:cite:`TownsendAshby83`) have established themselves as
the de-facto standard for modeling data from simple decision making
tasks (:cite:`SmithRatcliff04`). Each decision is modeled as a
sequential extraction and accumulation of information from the
environment and/or internal representations. Once the accumulated
evidence crosses a threshold, a corresponding response is
executed. This simple assumption about the underlying psychological
process has the intriguing property of reproducing reaction time
distributions and choice probability in simple two-choice decision
making tasks.

Hierarchical Bayesian methods are quickly gaining popularity in
cognitive sciences (:cite:`LeeWagenmakersIP`). Traditionally,
psychological models where either fit separately to individual
subjects (thus not taking similarities of subjects into account) or to
the whole group (thus not taking differences of subjects into
account). Hierarchical Bayesian methods provide a remedy for this
problem by allowing group and subject parameters to be estimated
simultaniously at different hierarchies. In essence, subject
parameters are assumed to come from a group distribution. In addition,
because these methods are Bayesian they deal naturally with
uncertainty and variability in the parameter estimations.

HDDM_ (Hierarchical Drift Diffusion Modeling) is an open-source
software package written in Python_ which allows (i) the construction
of hierarchical Bayesian drift models and (ii) the estimation of its
posterior parameter distributions via PyMC_
(:cite:`PatilHuardFonnesbeck10`). For efficiency, all runtime critical
functions are coded in Cython_ (:cite:`BehnelBredshawCitroEtAl11`),
heavily optimized and compiled natively. User-defined models can be
constructed via a simple configuration files or directly via HDDM
library calls. To assess model fit, HDDM generates different
statistics and comes with various plotting capabilities. For
illustrative purposes, HDDM includes a graphical demo applet which
simulates individual drift processes under user-specified parameter
combinations. The code is test-covered to assure correct function and
is properly documented. Online documentation and tutorials are
provided.

In sum, the presented software allows researches to construct and fit
complex, user-specified models using state-of-the-art estimation
methods without requiring a strong computer science or math
background.

**************************
Sequential Sampling Models
**************************

SSMs generally fall into one of two classes: (i) diffusion models
which assume that {\it relative`) evidence is accumulated over time
and (ii) race models which assume independent evidence accumulation
and response commitment once the first accumulator crossed a boundary
(:cite:`LaBerge62,Vickers70`). While there are many variants of these
models they are often closely related on a computational level and
sometimes mathematically equivalent under certain assumptions
(:cite:`BogaczBrownMoehlisEtAl06`). As such, I will restrict
discussion to two exemplar models from each class widely used in the
literature: the drift diffusion model (DDM)
(:cite:`RatcliffRouder98,RatcliffMcKoon08`) belonging to the class of
diffusion models and the linear ballistic accumulator (LBA)
(:cite:`BrownHeathcote08`) belonging to the class of race models.

Drift Diffusion Model
=====================

The DDM models decision making in two-choice tasks. Each choice is
represented as and upper and lower boundary. A drift-process
accumulates evidence over time until it crosses one of the two
boundaries and initiates the corresponding response
(:cite:`RatcliffRouder98,SmithRatcliff04`). The speed with which the
accumulation process approaches one of the two boundaries is called
the drift rate and represents the relative evidence for or against a
particular response. Because there is noise in the drift process, the
time of the boundary crossing and the selected response will vary
between trials. The distance between the two boundaries
(i.e. threshold) influences how much evidence must be accumulated
until a response is executed. A lower threshold makes responding
faster in general but increases the influence of noise on decision
making while a higher threshold leads to more cautious
responding. Reaction time, however, is not solely comprised of the
decision making process -- perception, movement initiation and
execution all take time and are summarized into one variable called
non-decision time. The starting point of the drift process relative to
the two boundaries can influence if one response has a prepotent
bias. This pattern gives rise to the reaction time distributions of
both choices (see figure :ref:`ddm`).

.. _ddm:

.. figure:: DDM_drifts_w_labels.svg

    Trajectories of multiple drift-processs (blue and red lines,
    middle panel). Evidence is accumulated over time (x-axis) with
    drift-rate v until one of two boundaries (separated by
    threshold a) is crossed and a response is initiated. Upper (blue)
    and lower (red) panels contain histograms over
    boundary-crossing-times for two possible responses. The histogram
    shapes match closely to that observed in reaction time
    measurements of research participants.

Later on, the DDM was extended to include inter-trial variability in
the drift-rate, the non-decision time and the starting point in order
to account for two phenomena observed in decision making tasks --
early and late errors. Models that take this into account are referred
to as the full DDM (:cite:`RatcliffRouder98`).


Linear Ballistic Accumulator
============================

The Linear Ballistic Accumulator (LBA) model belongs to the class of
race models (:cite:`BrownHeathcote08`). Instead of one drift process
and two boundaries, the LBA contains one drift process for each
possible response with a single boundary each. Thus, the LBA can model
decision making when more than two responses are possible. Moreover,
unlike the DDM, the LBA drift process has no intra-trial variance. RT
variability is obtained by including inter-trial variability in the
drift-rate and the starting point distribution (see figure
:ref:`lba`). Note that the simplifying assumption of a noiseless
drift-process simplifies the math significantly leading to a
computationally faster likelihood function for this model.

In a simulation study it was shown that the LBA and DDM lead to
similar results as to which parameters are affected by certain
manipulations (:cite:`DonkinBrownHeathcoteEtAl11`).

.. _lba:

.. figure:: lba.png

    Two linear ballistic accumulators (left and right) with different
    noiseless drifts (arrows) sampled from a normal distribution
    initiated at different starting points sampled from uniform
    distribution. In this case, accumulator for response alternative 1
    reaches criterion first and gets executed. Because of this race
    between two accumulators towards a common threshold these model
    are called race-models. Reproduced from
    \citet{DonkinBrownHeathcoteEtAl11`).


Relationship to cognitive neuroscience
======================================

SSMs were originally developed from a pure information processing
point of view and primarily used in psychology as a high-level
approximation of the decision process. More recent efforts in
cognitive neuroscience have simultaneously (i) validated core
assumptions of the model by showing that neurons indeed integrate
evidence probabilistically during decision making
(:cite:`SmithRatcliff04,GoldShadlen07`) and (ii) applied this model to
understand and describe neural correlates of cognitive processes
(:cite:`ForstmannAnwanderSchaferEtAl10,CavanaghWieckiCohenEtAl11`).\\

Multiple routes to decision threshold modulation have been
identified. Decision threshold in the speed-accuracy trade-off is
modulated by changes in the functional connectivity between pre-SMA
and striatum (:cite:`ForstmannAnwanderSchaferEtAl10`). Neural network
modeling (:cite:`Frank06,RatcliffFrank12`) validated by studies of PD
patients with a deep-brain-stimulator (DBS) in their subthalamic
nucleus (STN) (:cite:`FrankSamantaMoustafaEtAl07`) suggest that this
node is implicated in raising the decision threshold when there is
conflict between two options associated with similar rewards. This
result was further corroborated by (:cite:`CavanaghWieckiCohenEtAl11`)
who found that frontal theta power (as measured by
electroencelophagraphy and thought to correspond to conflict
(:cite:`CavanaghZambrano-VazquezAllen12`)) is correlated with decision
threshold increase on a trial-by-trial basis. As predicted, this
relationship was broken in PD patients with DBS turned on (but,
critically, not when DBS was turned off thus showing the effect is not
a result of the disease). In other words, by interfering with STN
function through stimulation we were able to show that this brain area
is casually involved in decision threshold modulation despite intact
experience of conflict (as measured by theta power). Interestingly,
these results provide a computational cognitive explanation for the
clinical symptom of impulsivity observed in PD patients receiving DBS
(:cite:`FrankSamantaMoustafaEtAl07`).

------------------------------
Model Fitting
------------------------------

Statistics and machine learning have developed efficient and versatile
Bayesian methods to solve various inference problems
:cite:`Poirier06`. More recently, they have seen wider adoption in
applied fields such as genetics :cite:`StephensBalding09` and
psychology :cite:`ClemensDeSelenEtAl11`. One reason for this
Bayesian revolution is the ability to quantify the certainty one has
in a particular estimation. Moreover, hierarchical Bayesian models
provide an elegant solution to the problem of estimating parameters of
individual subjects outlined above. Under the assumption that
participants within each group are similar to each other, but not
identical, a hierarchical model can be constructed where individual
parameter estimates are constrained by group-level distributions
:cite:`NilssonRieskampWagenmakers11 ShiffrinLeeKim08`.

Bayesian methods require specification of a generative process in form
of a likelihood function that produced the observed data :math:`x` given
some parameters :math:`\theta`. By specifying our prior belief we can use
Bayes formula to invert the generative model and make inference on the
probability of parameters :math:`\theta`:

.. _bayes:

.. math::

    P(\theta|x) = \frac{P(x|\theta) * P(\theta)}{P(x)}


Where :math:`P(x|\theta)` is the likelihood and :math:`P(\theta)` is
the prior probability. Computation of the marginal likelihood :math:`P(x)`
requires integration (or summation in the discrete case) over the
complete parameter space :math:`\Theta`:

.. math::

    P(x) = \int_\Theta P(x|\theta) \, \mathrm{d}\theta


Note that in most scenarios this integral is analytically
intractable. Sampling methods like Markov-Chain Monte Carlo (MCMC)
:cite:`GamermanLopes06` circumvent this problem by providing a way to
produce samples from the posterior distribution. These methods have
been used with great success in many different scenarios
:cite:`GelmanCarlinSternEtAl03` and will be discussed in more detail
below.

Another nice property of the Bayesian method is that it lends itself
naturally to a hierarchical design. In such a design, parameters for
one distribution can themselves come from a different distribution
which allows chaining together of distributions of arbitrary
complexity and map the structure of the data onto the model.

This hierarchical property has a particular benefit to cognitive
modeling where data is often scarce. We can construct a hierarchical
model to more adequately capture the likely similarity structure of
our data. As above, observed data points of each subject
:math:`x_{i,j}` (where :math:`i = 1, \dots, S_j` data points per
subject and :math:`j = 1, \dots, N` for :math:`N` subjects) are
distributed according to some likelihood function :math:`f | \theta`.
We now assume that individual subject parameters :math:`\theta_j` are
normal distributed around a group mean with a specific group variance
(:math:`\lambda = (\mu, \sigma)` with hyperprior :math:`G_0`)
resulting in the following generative description:

.. math::

  \mu, \sigma \sim G_0() \\
  \theta_j \sim \mathcal{N}(\mu, \sigma^2) \\
  x_{i, j} \sim f(\theta_j)

See figure :ref:`graphical_hierarchical` for the corresponding graphical model description.

Another way to look at this hierarchical model is to consider that our
fixed prior on :math:`\theta` from formula (:ref:`bayes`) is actually
a random variable (in our case a normal distribution) parameterized by
:math:`\lambda` which leads to the following posterior formulation:

.. math::

    P(\theta, \lambda | x) = \frac{P(x|\theta) * P(\theta|\lambda) * P(\lambda)}{P(x)}


.. _graphical_hierarchical:

.. figure:: graphical_hierarchical.svg

    Graphical notation of a hierarchical model. Circles represent
    continuous random variables. Arrows connecting circles specify
    conditional dependence between random variables. Shaded circles
    represent observed data. Finally, plates around graphical nodes
    mean that multiple identical, independent distributed random
    variables exist.

Note that we can factorize :math:`P(x|\theta)` and
:math:`P(\theta|\lambda)` due to their conditional independence. This
formulation also makes apparent that the posterior contains estimation
of the individual subject parameters :math:`\theta_j` and group
parameters :math:`\lambda`.

Finally, note that in our computational psychiatry application the
homogeneity assumption that all subjects come from the same normal
distribution is almost certainly violated (see above). To deal with
the heterogeneous data often encountered in psychiatry I will discuss
mixture models further down below. Next, I will describe algorithms to
estimate this posterior distribution.

----------------------------------------------
Hierarchical Bayesian Drift Diffusion Modeling
----------------------------------------------

The graphical model of our hierarchical DDM can be appreciated in
figure 2.

..  figure:: hier_model.svg

.. bibliography:: hddm.bib

.. _HDDM: http://github.com/twiecki/hddm
.. _Python: http://www.python.org/
.. _PyMC: http://code.google.com/p/pymc/
.. _Cython: http://www.cython.org/
