============
Introduction
============

Diffusion models have established themselves as the de-facto standard
for fitting simple decision making processes in the last years. Each
decision is modeled as a drift process that once it reaches a certain
threshold executes a response. This simple assumption about the
underlying psychological process has the intriguing property of
reproducing the full shape of reaction times in simple decision making
tasks.

Hierarchical bayesian estimation methods are quickly gaining
popularity in cognitive sciences. Traditionally, models where
either fit separately to individual subjects (thus not taking
similarities of subjects into account) or to the whole group (thus not
taking differences of subjects into account). Hierarchical bayesian
methods provide a remedy for this problem by allowing group and
subject parameters to be estimated simultaniously at different
hierarchies. In essence, subject parameters are assumed to come from a
group distribution. Markov-Chain Monte Carlo (MCMC) methods allow the
estimation of these distributions. In addition, because these methods
are bayesian they deal naturally with uncertainty and variability.

HDDM_ (Hierarchical Drift Diffusion Modeling) is an open-source
software package written in Python_ which allows (i) the construction
of hierarchical bayesian drift models and (ii) the estimation of
posterior parameter distributions via PyMC_. For efficiency, all
runtime critical functions are coded in cython_, heavily optimized and
compiled natively. Models can be constructed via a simple
configuration file. For illustrative purposes, HDDM includes a
graphical demo applet which simulates individual drift processes under
different parameter combinations.

----------------
Diffusion Models
----------------

HDDM implements two drift model types, (i) the Ratcliff drift
diffusion model (DDM) and (ii) the linear ballistic accumulator
(LBA). Both of these models implement decision making as an evidence
accumulation process that executes a response upon crossing a decision
threshold. The speed of the accumulation is called the drift rate and
influences how swiftly a particular reponse is executed; it depends
stimulus properties. The distance of the decision threshold from the
starting point of evidence accumulation also influences the speed of
responding, but unlike the drift rate, influences the speed of all
responses. A lower threshold makes responding faster in general but
more random while a higher threshold leads to more cautious
responding. Reaction time, however, is not solely comprised of the
decision making process -- perception, movement initiation and
execution all take time and are summarized into one variable called
non-decision time.

Ratcliff Drift Diffusion Model
------------------------------

The Ratcliff DDM models decision making in two-choice tasks -- each
choice is represented as and upper and lower boundary. A drift process
accumulates evidence over time until it crosses one of the two
boundaries and initiates the corresponding response. Because there is
noise in the drift process, the time of the boundary crossing and the
selected response will vary between trials. The starting point of the
drift process relative to the two boundaries can influence if one
response has a prepotent bias. This pattern gives rise to the reaction
time distributions of both choices and will henceforth be called the
simple DDM.

Early on, Ratcliff noticed that this simple DDM could not account for
two phenomena observed in decision making -- early and late
errors. This lead to inclusion of intertrial variability in the
drift-rate, the non-decision time and the starting point. Models that
take this into account are henceforth called full DDM.


Linear Ballistic Accumulator
----------------------------


------------------------------
Hierarchical Bayesian Modeling
------------------------------

.. _HDDM: http://code.google.com/p/hddm/
.. _Python: http://www.python.org/
.. _PyMC: http://code.google.com/p/pymc/
.. _Cython: http://www.cython.org/
