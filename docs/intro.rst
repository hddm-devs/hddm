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


Ratcliff Drift Diffusion Model
------------------------------


Linear Ballistic Accumulator
----------------------------


------------------------------
Hierarchical Bayesian Modeling
------------------------------

.. _HDDM: http://code.google.com/p/hddm/
.. _Python: http://www.python.org/
.. _PyMC: http://code.google.com/p/pymc/
.. _Cython: http://www.cython.org/
