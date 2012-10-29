
.. _CHANGES:

=============
Release Notes
=============

HDDM 0.4
========

New features
------------

* Handling of outliers via mixture likelihood.
* New model HDDMRegression to allow estimation of trial-by-trial
  regressions with a covariate.
* New model HDDMStimulusCoding.
* New model HLBA -- a hierarchical Linear Ballistic Accumulator.
* Posterior predictive quantile plots (see model.plot_posterior_quantiles()).

Bugfixes
--------

* model.load_db() is working again.

HDDM 0.3 (6 Sep 2012)
========

* Complete rewrite of the underlying model creation engine (kabuki) to
  allow for more flexible model creation including transforms. This
  enabled development of a new HDDM default model without explicit
  parameter bounds.
* Group mean distributions are now Gibbs sampled and group variability
  distributions are now slice sampled leading to much improved
  convergence and mixing.
* MAP approximation of hierarchical models for better initialization.
* Improved documentation (check out the `How-to`_ section).
* Chi-square fitting using the Ratcliff quantile method.
* Posterior predictive checks.

HDDM 0.2 (First publicly announced release)
===========================================

* Better model initialization that shouldn't fail.
* Many bugfixes.
* Major internal overhaul.

HDDM 0.1 (20 July 2011) (Semi-private MathPsych release)
========================================================

* Flexible HDDM model class to fit group and subject models.
* Heavily optimized cython likelihoods.

.. How-to: http://ski.clps.brown.edu/hddm_docs/howto.html
