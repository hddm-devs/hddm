.. index:: Introduction
.. _chap_introduction:

************
Introduction
************

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
:cite:`RatcliffPhiliastidesSajda09`). One issue in such model-based
cognitive neuroscience approaches is that the trial numbers in each
condition are often low, making it difficult it difficult to estimate
model parameters. For example, studies with patient populations,
especially if combined with intraoperative recordings, typically have
substantial constraints on the duration of the task. Similarly,
model-based fMRI or EEG studies are often interested not in static
model parameters, but how these dynamically vary with trial-by-trial
variations in recorded brain activity. Efficient and reliable
estimation methods that take advantage of the full statistical
structure available in the data across subjects and conditions are
critical to the success of these endeavors.

Bayesian data analytic methods are quickly gaining popularity in the
cognitive sciences because of their many desirable properties
(:cite:`LeeWagenmakers13`, :cite:`Kruschke10`). First, Bayesian methods
allow inference of the full posterior distribution of each parameter,
thus quantifying uncertainty in their estimation, rather
than simply provide their most likely value. Second, hierarchical modeling is
naturally formulated in a Bayesian framework. Traditionally,
psychological models either assume subjects are completely independent
of each other, fitting models separately to each individual, or that
all subjects are the same, fitting models to the group as if they
are all copies of some "average subject". Both approaches are sub-optimal in
that the former fails to capitalize on statistic strength offered by
the degree to which subjects are similar in one or more model
parameters, whereas the latter approach fails to account for the
differences among subjects, and hence could lead to a situation where
the estimated model cannot fit any individual subject. The same limitations
apply to current DDM software packages such as DMAT_
:cite:`VandekerckhoveTuerlinckx08` and fast-dm_
:cite:`VossVoss07`. Hierarchical Bayesian methods provide a remedy for
this problem by allowing group and subject parameters to be estimated
simultaneously at different hierarchical levels
(:cite:`LeeWagenmakers13`, :cite:`Kruschke10`, :cite:`VandekerckhoveTuerlinckxLee11`). Subject parameters are
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
models can be created via a simple python script or be used
interactively via, for example, IPython_ interpreter shell (:cite:PER-GRA2007). All
run-time critical functions are coded in Cython_
(:cite:`BehnelBradshawCitroEtAl11`) and compiled natively for speed
which allows estimation of complex models in minutes. HDDM includes
many commonly used statistics and plotting functionality generally
used to assess model fit. The code is released under the permissive
BSD 3-clause license, test-covered to assure correct behavior and well
documented. Finally, HDDM allows flexible estimation of trial-by-trial
regressions where an external measurement (e.g. brain activity as
measured by fMRI) is correlated with one or more decision making
parameters.

With HDDM we aim to provide a user-friendly but powerful tool that can
be used by experimentalists to construct and fit complex,
user-specified models using state-of-the-art estimation methods to
test their hypotheses. The purpose of this report is to introduce the
toolbox and provide a tutorial for how to employ it; subsequent
reports will quantitatively characterize its success in recovering
model parameters and advantages relative to non-hierarchical or
non-Bayesian methods as a function of the number of subjects and
trials (:cite: `SoferWieckiFrank`).

.. _HDDM: http://github.com/twiecki/hddm
.. _Python: http://www.python.org/
.. _PyMC: http://code.google.com/p/pymc/
.. _Cython: http://www.cython.org/
.. _DMAT: http://ppw.kuleuven.be/okp/software/dmat/
.. _fast-dm: http://seehuhn.de/pages/fast-dm
.. _IPython: http://ipython.org
