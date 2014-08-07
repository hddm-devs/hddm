.. _CHANGES:

=============
Release Notes
=============

HDDM 0.5.5 (bugfix release)
===========================

* Upgrade dependency to pymc 2.3.3
* Remove LBA model as likelihood seems broken

HDDM 0.5.3 (bugfix release)
===========================

* Compatibility with pandas > 0.13.
* Fix problem that causes stats to not be generated when
  loading model.
* Update packages to work with anaconda 1.9.

HDDM 0.5.2 (bugfix release)
===========================

* Refactored posterior predictive plots and added tutorial:
  http://ski.clps.brown.edu/hddm_docs/tutorial_post_pred.html
* Smaller bugfixes.
* Works with PyMC 2.3.
* Experimental features:
    * Updated HLBA model but currently has bad recovery.
    * Added sample_emcee() to use the emcee parallel sampler.
      Seems to work but requires some tuning and does not seem
      to beat slice sampling.

HDDM 0.5
========

* New and improved HDDM model with the following changes:
    * Priors: by default model will use informative priors
      (see http://ski.clps.brown.edu/hddm_docs/methods.html#hierarchical-drift-diffusion-models-used-in-hddm)
      If you want uninformative priors, set ``informative=False``.
    * Sampling: This model uses slice sampling which leads to faster
      convergence while being slower to generate an individual
      sample. In our experiments, burnin of 20 is often good enough.
    * Inter-trial variablity parameters are only estimated at the
      group level, not for individual subjects.
    * The old model has been renamed to ``HDDMTransformed``.
    * HDDMRegression and HDDMStimCoding are also using this model.
* HDDMRegression takes patsy model specification strings. See
  http://ski.clps.brown.edu/hddm_docs/howto.html#estimate-a-regression-model
  and
  http://ski.clps.brown.edu/hddm_docs/tutorial_regression_stimcoding.html#chap-tutorial-hddm-regression
* Improved online documentation at
  http://ski.clps.brown.edu/hddm_docs
* A new HDDM demo at http://ski.clps.brown.edu/hddm_docs/demo.html
* Ratcliff's quantile optimization method for single subjects and
  groups using the ``.optimize()`` method
* Maximum likelihood optimization.
* Many bugfixes and better test coverage.
* hddm_fit.py command line utility is depracated.

HDDM 0.4.1
==========

* Models are now pickable.
  (This means they can be loaded and saved.
  Critically, it is now also trivial to run multiple
  models in parallel that way.)

HDDM 0.4
========

License
-------

HDDM 0.4 is now distributed under the simplified BSD license (see the
LICENSE file) instead of GPLv3.

New features
------------

* Handling of outliers via mixture model.
  http://ski.clps.brown.edu/hddm_docs/howto.html#deal-with-outliers
* New model HDDMRegression to allow estimation of trial-by-trial
  regressions with a covariate.
  http://ski.clps.brown.edu/hddm_docs/howto.html#estimate-a-regression-model
* New model HDDMStimulusCoding.
  http://ski.clps.brown.edu/hddm_docs/howto.html#code-subject-responses
* New model HLBA -- a hierarchical Linear Ballistic Accumulator model (hddm.HLBA).
* Posterior predictive quantile plots (see model.plot_posterior_quantiles()).

Bugfixes
--------

* model.load_db() is working again.


HDDM 0.3.1
==========

* Fixed annoying bug that broke plotting of posterior predictive.

HDDM 0.3 (6 Sep 2012)
======================

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
