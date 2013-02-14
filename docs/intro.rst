============
Introduction
============

Sequential sampling models (SSMs) (:cite:`TownsendAshby83`) have
established themselves as the de-facto standard for modeling
reaction-time data from simple two-alternative forced choice decision
making tasks (:cite:`SmithRatcliff04`). Each decision is modeled as an
accumulation of noisy information indicative of one choice or the
other, with sequential evaluation of the accumulated evidence at each
time step. Once this evidence crosses a threshold, the corresponding
response is executed. This simple assumption about the underlying
psychological process has the appealing property of reproducing not
only choice probabilities, but the full distribution of response times
for each of the two choices. Models of this class have been used
successfully in mathematical psychology since the 60s and more
recently adopted in cognitive neuroscience investigations. These
studies are typically interested in neural mechanisms associated with
the accumulation process or for regulating the decision threshold (see
e.g. :cite:`ForstmannDutilhBrownEtAl08`,
:cite:`CavanaghWieckiCohenEtAl11`,
:cite:`RatcliffPhiliastidesSadja09`). One issue in such model-based
cognitive neuroscience apoproaches is that the trial numbers in each
condition are often low, making it difficult it difficult to estimate
model parameters. For example, studies with patient populations,
especially if combined with intraoperative recordings, typically have
substantial constraints on the duration of the task. Similarly,
model-based fMRI or EEG studies are often interested not in static
model parameters, but how these dynamically vary with trial-by-trial
variations in recorded brain activity. Efficient and reliable
estimation methods that take advantage of the full statistical
structure available in the data across subjects and conditions are
critical to the success of these endeavours.

Bayesian data analytic methods are quickly gaining popularity in the
cognitive sciences because of their many desirable properties
(:cite:`LeeWagenmakers`, :cite:`Kruschke11`). First, Bayesian methods
allow inference of the full posterior distribution on each parameter
estimate, thus quantifying uncertainty about each parameter, rather
than simply the most likely value. Moreover, hierarchical modeling is
naturally formulated in a Bayesian framework. Traditionally,
psychological models either assume subjects are completely independent
of each other, fitting models separately to each individual, or that
all subjects are the same, fitting models to the group as if they
comprised a single "uber-subject". Both approaches are sub-optimal in
that the former fails to capitalize on statistic strength offered by
the degree to which subjects are similar in one more more model
parameters, whereas the latter approach fails to account for the
differences among subjects, and hence could lead to a situation where
the model cannot fit any individual subject. The same limitations
apply to current DDM software packages such as DMAT_
:cite:`VandekerckhoveTuerlinckx08` and fast-dm_
:cite:`VossVoss07`. Hierarchical Bayesian methods provide a remedy for
this problem by allowing group and subject parameters to be estimated
simultaneously at different hierarchical levels
(:cite:`LeeWagenmakers`, :cite:`Kruschke11`). Subject parameters are
assumed to be drawn from a group distribution, and to the degree that
subject are similar to each other, the variance in the group
distribution will be estimated to be small, which reciprocally has a
greater influence on constraining parameter estimates of any
individual. Even in this scenario, the method still allows the
posterior for any given individual subject to differ substantially
from that of the rest of the group given sufficient data to overwhelm
the group prior. Thus the method capitalizes on statistical strength
shared across the individuals, and can do so to different degrees even
within the same sample and model, depending on the extent to which
subjects are similar to each other in one parameter vs. another. In
the DDM for example, it may be the case that there is relatively
little variability across subjects in the perceptual time for stimulus
encoding, quantified by the "non-decision time" but more variability
in their degree of response caution, quantified by the "decision
threshold". The estimation should be able to capitalize on this
structure so that the non-decision time in any given subject is
anchored by that of the group, potentially allowing for more efficient
estimation of that subjects decision threshold. This approach may be
particularly helpful when relatively few trials per condition are
available for each subject, and when incorporating noisy
trial-by-trial neural data into the estimation of DDM parameters.

HDDM_ is an open-source software package written in Python_ which
allows (i) the flexible construction of hierarchical Bayesian drift
diffusion models and (ii) the estimation of its posterior parameter
distributions via PyMC_ (:cite:`PatilHuardFonnesbeck10`). User-defined
models can be specified via a configuration file thus requiring no
Python knowledge. Alternatively, more sophisticated usage of the
toolbox is available interactively via, for example, IPython_ or by
writing succinct Python scripts. All runtime critical functions are coded in
Cython_ (:cite:`BehnelBredshawCitroEtAl11`) and compiled natively for
speed which allows estimation of complex models in minutes. HDDM
includes many commonly used statistics and plotting functionality
generally used to assess model fit. The code is released under the
permissive BSD 3-clause license, test-covered to assure correct
behavior and well documented. Finally, HDDM allows flexible estimation
of trial-by-trial regressions where an external measurement
(e.g. brain activity as measured by fMRI) is correlated with one or
more decision making parameters.

With HDDM we aim to provide a user-friendly but powerful tool that can
be used by experimentalists to construct and fit complex,
user-specified models using state-of-the-art estimation methods to
test their hypotheses. The purpose of this report is to introduce the
toolbox and provide a tutorial for how to employ it; subsequent
reports will quantitatively characterize its success in recovering
model parameters and advantages relative to non-hierarchical or
non-Bayesian methods as a function of the number of subjects and
trials \cite{SoferWieckiFrankIP}.


==========================
Methods
==========================

**************************
Sequential Sampling Models
**************************


SSMs generally fall into one of two classes: (i) diffusion models
which assume that *relative* evidence is accumulated over time
and (ii) race models which assume independent evidence accumulation
and response commitment once the first accumulator crossed a boundary
(:cite:`LaBerge62`, :cite:`Vickers70`). Currently, HDDM includes two of the most
commonly used SSMs: the drift diffusion model (DDM)
(:cite:`RatcliffRouder98`, :cite:`RatcliffMcKoon08`) belonging to the
class of diffusion models and the linear ballistic accumulator (LBA)
(:cite:`BrownHeathcote08`) belonging to the class of race models.

Drift Diffusion Model
---------------------

The DDM models decision making in two-choice tasks. Each choice is
represented as an upper and lower boundary. A drift-process
accumulates evidence over time until it crosses one of the two
boundaries and initiates the corresponding response
(:cite:`RatcliffRouder98`, :cite:`SmithRatcliff04`). The speed with
which the accumulation process approaches one of the two boundaries is
called drift-rate *v* and represents the relative evidence for or
against a particular response. Because there is noise in the drift
process, the time of the boundary crossing and the selected response
will vary between trials. The distance between the two boundaries
(i.e. threshold *a*) influences how much evidence must be accumulated
until a response is executed. A lower threshold makes responding
faster in general but increases the influence of noise on decision
making and can hence lead to errors or impulsive choice, whereas a
higher threshold leads to more cautious responding (slower, more
skewed RT distributions, but more accurate). Response time, however,
is not solely comprised of the decision making process -- perception,
movement initiation and execution all take time and are lumped in the
DDM by a single non-decision time parameter *t*. The model also allows
for a prepotent bias *z* affecting the starting point of the drift
process relative to the two boundaries. The termination times of this
generative process gives rise to the reaction time distributions of
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

An analytical solution to the resulting probability distribution of
the termination times was provided by :cite:`Feller68`:

.. math::

    f(t|v, a, z) = \frac{\pi}{a^2} \, \text{exp} \left( -vaz-\frac{v^2\,t}{2} \right) \times \sum_{k=1}^{\infty} k\, \text{exp} \left( -\frac{k^2\pi^2 t}{2a^2} \right) \text{sin}\left(k\pi z\right)

Note that the infinite sum requires some form of approximation. HDDM
uses the likelihood formulation provided by :cite:`NavarroFuss09`. 
%which takes on the general form....?

Later on, the DDM was extended to include additional noise parameters
capturing inter-trial variability in the drift-rate, the non-decision
time and the starting point in order to account for two phenomena
observed in decision making tasks, most notably cases where errors are
faster or slower than correct responses. Models that take this into
account are referred to as the full DDM
(:cite:`RatcliffRouder98`). HDDM uses analytical integration of the
likelihood function for variability in drift-rate and numerical
integration for variability in non-decision time and bias. More
information on the model specifics can be found in Sofer, Wiecki, &
Frank (in preparation).


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
computationally more efficient  likelihood function for this model.

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
    :cite:`DonkinBrownHeathcoteEtAl11`.

------------------------------
Hierarchical Bayesian Estimation
------------------------------

Statistics and machine learning have developed efficient and versatile
Bayesian methods to solve various inference problems
:cite:`Poirier06`. More recently, they have seen wider adoption in
applied fields such as genetics :cite:`StephensBalding09` and
psychology :cite:`ClemensDeSelenEtAl11`. One reason for this
Bayesian revolution is the ability to quantify the certainty one has
in a particular estimation. Moreover, hierarchical Bayesian models
provide an elegant solution to the problem of estimating parameters of
individual subjects and groups of subjects, as outlined above. Under the assumption that
participants within each group are similar to each other, but not
identical, a hierarchical model can be constructed where individual
parameter estimates are constrained by group-level distributions
(:cite:`NilssonRieskampWagenmakers11`, :cite:`ShiffrinLeeKim08`).

Bayesian methods require specification of a generative process in form
of a likelihood function that produced the observed data :math:`x` given
some parameters :math:`\theta`. By specifying our prior beliefs (which
can be informed or unbiased) we can use
Bayes formula to invert the generative model and make inference on the
probability of parameters :math:`\theta`:

.. _bayes:

.. math::

    P(\theta|x) = \frac{P(x|\theta) \times P(\theta)}{P(x)}


Where :math:`P(x|\theta)` is the likelihood of observing the data (in
this case choices and RTs) given each parameter value and :math:`P(\theta)` is
the prior probability of the parameters. Computation of the marginal likelihood :math:`P(x)`
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

As noted above, the Bayesian method  lends itself
naturally to a hierarchical design. In such a design, parameters for
one distribution can themselves be drawn from a higher level
distribution. This hierarchical property has a particular benefit to cognitive
modeling where data is often scarce. We can construct a hierarchical
model to more adequately capture the likely similarity structure of
our data. As above, observed data points of each subject
:math:`x_{i,j}` (where :math:`i = 1, \dots, S_j` data points per
subject and :math:`j = 1, \dots, N` for :math:`N` subjects) are
distributed according to some likelihood function :math:`f | \theta`.
We now assume that individual subject parameters :math:`\theta_j` are
normally distributed around a group mean with a specific group variance
(:math:`\lambda = (\mu, \sigma)`, where these group parameters are
estimated from the data given hyperpriors :math:`G_0`), 
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

    P(\theta, \lambda | x) = \frac{P(x|\theta) \times P(\theta|\lambda) \times P(\lambda)}{P(x)}


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

-----------------------------------
Hierarcical Drift-Diffusion Models used in HDDM
-----------------------------------

HDDM includes several hierarchical Bayesian model formulations for the
DDM and LBA. For illustrative purposes we present the graphical model
depiction of the default DDM hierarchical model in
:ref:`graphical_hddm`.

.. _graphical_hddm:

..  figure:: graphical_hddm.svg

    Basic graphical hierarchical model implemented by HDDM for
    estimation of the drift-diffusion model.

Individual graphical nodes are distributed as follows.

.. math::

    \mu_{a} &\sim \mathcal{N}(0, 1) \\
    \mu_{z} &\sim \mathcal{N}(0, 1) \\
    \mu_{v} &\sim \mathcal{N}(0, 1) \\
    \mu_{ter} &\sim \mathcal{N}(0, 1) \\
    \mu_{sv} &\sim \mathcal{N}(0, 1) \\
    \mu_{sz} &\sim \mathcal{N}(0, 1) \\
    \mu_{ster} &\sim \mathcal{N}(0, 1) \\
    \\
    \sigma_{a} &\sim \mathcal{U}(1e^{-10}, 100) \\
    \sigma_{z} &\sim \mathcal{U}(1e^{-10}, 100) \\
    \sigma_{v} &\sim \mathcal{U}(1e^{-10}, 100) \\
    \sigma_{ter} &\sim \mathcal{U}(1e^{-10}, 100) \\
        \sigma_{sv} &\sim \mathcal{U}(1e^{-10}, 100) \\
    \sigma_{sz} &\sim \mathcal{U}(1e^{-10}, 100) \\
    \sigma_{ster} &\sim \mathcal{U}(1e^{-10}, 100) \\
    \\
    a_{j} &\sim \mathcal{N}(\mu_{a}, \sigma_{a}^2) \\
    z_{j} &\sim \mathcal{N}(\mu_{z}, \sigma_{z}^2) \\
    v_{j} &\sim \mathcal{N}(\mu_{v}, \sigma_{v}^2) \\
    ter_{j} &\sim \mathcal{N}(\mu_{ter}, \sigma_{ter}^2) \\
    sv_{j} &\sim \mathcal{N}(\mu_{sv}, \sigma_{sv}^2) \\
    sz_{j} &\sim \mathcal{N}(\mu_{sz}, \sigma_{sz}^2) \\
    ster_{j} &\sim \mathcal{N}(\mu_{ster}, \sigma_{ster}^2) \\
    \\
    x_{i, j} &\sim F(a_{i}, z_{i}, v_{i}, ter_{i}, sv_{i}, sz_{i}, ster_{i})

where :math:`x_{i, j}` represents the observed data consisting of
reaction time and choice and :math:`F` represents the DDM likelihood
function as formulated by :cite:`NavarroFuss09`. As can be seen,
individual subject parameters are expected to be normal distributed
around a group mean :math:`\mu` with variance :math:`\sigma^2`. HDDM
then uses MCMC to estimate the joint posterior distribution of all
model parameters.

Note that the exact form of the model will be user-dependent; consider
as an example a model where separate drift-rates *v* are estimated for
two conditions in an experiment: easy and hard. In this case, HDDM
will create a hierarchical model with group parameters
:math:`\mu_{v_{\text{easy}}}`, :math:`\sigma_{v_{\text{easy}}}`,
:math:`\mu_{v_{\text{hard}}}`, :math:`\sigma_{v_{\text{hard}}}`,and individual subject parameters :math:`v_{j_{\text{easy}}}`, and :math:`v_{j_{\text{hard}}}`.

.. _HDDM: http://github.com/twiecki/hddm
.. _Python: http://www.python.org/
.. _PyMC: http://code.google.com/p/pymc/
.. _Cython: http://www.cython.org/
.. _DMAT: http://ppw.kuleuven.be/okp/software/dmat/
.. _fast-dm: http://seehuhn.de/pages/fast-dm
.. _IPython: http://ipython.org
