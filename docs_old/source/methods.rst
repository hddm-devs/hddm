.. index:: Methods
.. _chap_methods:


*******
Methods
*******


Sequential Sampling Models
##########################


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
*********************

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
both choices.

.. figure:: DDM.svg

    Trajectories of multiple drift-process (blue and red lines,
    middle panel). Evidence is accumulated over time (x-axis) with
    drift-rate v until one of two boundaries (separated by
    threshold a) is crossed and a response is initiated. Upper (blue)
    and lower (red) panels contain histograms over
    boundary-crossing-times for two possible responses. The histogram
    shapes match closely to that observed in reaction time
    measurements of research participants.

An analytic solution to the resulting probability distribution of
the termination times was provided by :cite:`Feller68`:

.. math::

    f(t|v, a, z) = \frac{\pi}{a^2} \, \text{exp} \left( -vaz-\frac{v^2\,t}{2} \right) \times \sum_{k=1}^{\infty} k\, \text{exp} \left( -\frac{k^2\pi^2 t}{2a^2} \right) \text{sin}\left(k\pi z\right)

Since the formula contains an infinite sum, HDDM uses an approximation
provided by :cite:`NavarroFuss09`.

Later on, the DDM was extended to include additional noise parameters
capturing inter-trial variability in the drift-rate, the non-decision
time and the starting point in order to account for two phenomena
observed in decision making tasks, most notably cases where errors are
faster or slower than correct responses. Models that take this into
account are referred to as the full DDM
(:cite:`RatcliffRouder98`). HDDM uses analytic integration of the
likelihood function for variability in drift-rate and numerical
integration for variability in non-decision time and bias. More
information on the model specifics can be found in :cite: `SoferWieckiFrank`.


Linear Ballistic Accumulator
****************************

The Linear Ballistic Accumulator (LBA) model belongs to the class of
race models (:cite:`BrownHeathcote08`). Instead of one drift process
and two boundaries, the LBA contains one drift process for each
possible response with a single boundary each. Thus, the LBA can model
decision making when more than two responses are possible. Moreover,
unlike the DDM, the LBA drift process has no intra-trial variance. RT
variability is obtained by including inter-trial variability in the
drift-rate and the starting point distribution. Note that the
simplifying assumption of a noiseless drift-process simplifies the
math significantly leading to a computationally more efficient
likelihood function for this model.

In a simulation study it was shown that the LBA and DDM lead to
similar results as to which parameters are affected by certain
manipulations (:cite:`DonkinBrownHeathcoteEtAl11`).

.. figure:: lba.png

    Two linear ballistic accumulators (left and right) with different
    noiseless drifts (arrows) sampled from a normal distribution
    initiated at different starting points sampled from a uniform
    distribution. In this case, the accumulator for response
    alternative 1 is more likely to reach the criterion first, and
    therefore gets selected more often. Because of this race between
    two accumulators towards a common threshold these model are called
    race-models. Reproduced from :cite:`DonkinBrownHeathcoteEtAl11`.


Hierarchical Bayesian Estimation
################################

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
of a likelihood function that produced the observed data :math:`x`
given some parameters :math:`\theta`. By specifying our prior beliefs
(which can be informed or non-informed) we can use Bayes formula to
invert the generative model and make inference on the probability of
parameters :math:`\theta`:

.. _bayes:

.. math::

    P(\theta|x) = \frac{P(x|\theta) \times P(\theta)}{P(x)}


Where :math:`P(x|\theta)` is the likelihood of observing the data (in
this case choices and RTs) given each parameter value and
:math:`P(\theta)` is the prior probability of the parameters. In most
cases the computation of the denominator is quite complicated and
requires to compute an analytically intractable integral. Sampling
methods like Markov-Chain Monte Carlo (MCMC) :cite:`GamermanLopes06`
circumvent this problem by providing a way to produce samples from the
posterior distribution. These methods have been used with great
success in many different scenarios :cite:`GelmanCarlinSternEtAl03`
and will be discussed in more detail below.

As noted above, the Bayesian method lends itself naturally to a
hierarchical design. In such a design, parameters for one distribution
can themselves be drawn from a higher level distribution. This
hierarchical property has a particular benefit to cognitive modeling
where data is often scarce. We can construct a hierarchical model to
more adequately capture the likely similarity structure of our
data. As above, observed data points of each subject :math:`x_{i,j}`
(where :math:`i = 1, \dots, S_j` data points per subject and :math:`j
= 1, \dots, N` for :math:`N` subjects) are distributed according to
some likelihood function :math:`f | \theta`.  We now assume that
individual subject parameters :math:`\theta_j` are normally
distributed around a group mean with a specific group variance
(:math:`\lambda = (\mu, \sigma)`, where these group parameters are
estimated from the data given hyper-priors :math:`G_0`), resulting in
the following generative description:

.. math::

  \mu, \sigma &\sim G_0() \\
  \theta_j &\sim \mathcal{N}(\mu, \sigma^2) \\
  x_{i, j} &\sim f(\theta_j)

.. figure:: graphical_hierarchical.svg

    Graphical notation of a hierarchical model. Circles represent
    continuous random variables. Arrows connecting circles specify
    conditional dependence between random variables. Shaded circles
    represent observed data. Finally, plates around graphical nodes
    mean that multiple identical, independent distributed random
    variables exist.

Another way to look at this hierarchical model is to consider that our
fixed prior on :math:`\theta` from above is actually
a random variable (in our case a normal distribution) parameterized by
:math:`\lambda` which leads to the following posterior formulation:

.. math::

    P(\theta, \lambda | x) = \frac{P(x|\theta) \times P(\theta|\lambda) \times P(\lambda)}{P(x)}

Note that we can factorize :math:`P(x|\theta)` and
:math:`P(\theta|\lambda)` due to their conditional independence. This
formulation also makes apparent that the posterior contains estimation
of the individual subject parameters :math:`\theta_j` and group
parameters :math:`\lambda`.


Hierarchical Drift-Diffusion Models used in HDDM
################################################

HDDM includes several hierarchical Bayesian model formulations for the
DDM and LBA. For illustrative purposes we present the graphical model
depiction of a hierarchical DDM model with informative priors and
group only inter-trial variablity parameters. Note, however, that
there is also a model with non-informative priors.

..  figure:: graphical_hddm.svg

    Basic graphical hierarchical model implemented by HDDM for
    estimation of the drift-diffusion model.

Individual graphical nodes are distributed as follows.

.. math::

    \mu_{a} &\sim \mathcal{G}(1.5, 0.75) \\
    \mu_{v} &\sim \mathcal{N}(2, 3) \\
    \mu_{z} &\sim \mathcal{N}(0.5, 0.5) \\
    \mu_{ter} &\sim \mathcal{G}(0.4, 0.2) \\
    \\
    \sigma_{a} &\sim \mathcal{HN}(0.1) \\
    \sigma_{v} &\sim \mathcal{HN}(2) \\
    \sigma_{z} &\sim \mathcal{HN}(0.05) \\
    \sigma_{ter} &\sim \mathcal{HN}(1) \\
    \\
    a_{j} &\sim \mathcal{G}(\mu_{a}, \sigma_{a}^2) \\
    z_{j} &\sim \text{invlogit}(\mathcal{N}(\mu_{z}, \sigma_{z}^2)) \\
    v_{j} &\sim \mathcal{N}(\mu_{v}, \sigma_{v}^2) \\
    ter_{j} &\sim \mathcal{N}(\mu_{ter}, \sigma_{ter}^2) \\
    \\
    sv &\sim \mathcal{HN}(2) \\
    ster &\sim \mathcal{HN}(0.3) \\
    sz &\sim \mathcal{B}(1, 3) \\
    \\
    x_{i, j} &\sim F(a_{i}, z_{i}, v_{i}, ter_{i}, sv, ster, sz)

where :math:`x_{i, j}` represents the observed data consisting of
reaction time and choice and :math:`F` represents the DDM likelihood
function as formulated by :cite:`NavarroFuss09`. :math:`\mathcal{N}`
represents a normal distribution parameterized by mean and standard
deviation, :math:`\mathcal{HN}` represents a half-normal parameterized
standard-deviation, :math:`\mathcal{G}` represents a Gamma
distribution parameterized by mean and rate, :math:`\mathcal{B}`
represents a Beta distribution parameterized by :math:`\alpha` and
:math:`\beta`. Note that in this model we do not attempt to estimate
individual parameters for inter-trial variabilities. The reason is
that the influence of these parameters onto the likelihood is often so
small that very large amounts of data would be required to make
meaningful inference at the individual level.

These priors are created to roughly match parameter values reported in
the literature and collected by :cite:`MatzkeWagenmakers09`. In the
below figure we overlayed those empirical values with the prior
distribution used for each parameter.

.. figure:: hddm_info_priors.svg

HDDM then uses MCMC to estimate the joint posterior distribution of
all model parameters.

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
