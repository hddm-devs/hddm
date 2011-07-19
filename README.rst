************
Introduction
************

:Date: July 19, 2011
:Author: Thomas V. Wiecki, Imri Sofer, Michael J. Frank
:Contact: thomas_wiecki@brown.edu, imri_sofer@brown.edu, michael_frank@brown.edu
:Web site: http://github.com/hddm-devs/hddm
:Copyright: This document has been placed in the public domain.
:License: HDDM is released under the GPLv3.
:Version: 0.1

Purpose
=======

HDDM is a python module that implements Hierarchical Bayesian estimation of Drift Diffusion Models (via PyMC).

Features
========

HDDM provides functionalities to make Drift Diffusion analysis of
error rates and reaction time distributions as painless as 
possible. Here is a short list of some of its features:

* Uses hierarchical bayesian estimation (via PyMC) of DDM parameters
  to allow simultaneous estimation of subject and group parameters,
  where individual subjects are assumed to be drawn from a group
  distribution. HDDM should thus produce better estimates when less RT
  values are measured compared to other methods using maximum
  likelihood for individual subjects (i.e. `DMAT`_ or `fast-dm`_). 

* Heavily optimized likelihood functions for speed.

* Flexible creation of complex models tailored to specific hypotheses
  (e.g. separate drift-rate or other parameters for different task
  conditions, or predicted changes in model parameters as a function
  of other indicators like brain activity).

* Easy specification of models via configuration file fosters exchange of models and research results.

* Built-in Bayesian hypothesis testing and several convergence and goodness-of-fit diagnostics.

Usage
=====

The easiest way to use HDDM is by creating a configuration file for your model:

example.conf
::

    [depends]
    v = difficulty

    [mcmc]
    samples=10000 # Sample 5000 posterior samples
    burn=5000 # Discard the first 100 samples as burn-in
    thin=3 # Discard every third sample to remove autocorrelations

Then call hddm:

::

    hddm_fit.py example.conf mydata.csv

Installing
==========

HDDM has the following dependencies:

* Python

* NumPy

* SciPy

* Matplotlib

* Cython_

* PyMC_ (installation instructions: http://pymc.googlecode.com/svn/doc/installation.html)

* kabuki_ 

Windows
-------

The easiest way is to download and install the `Enthought Python
Distribution`_ which has a free version for academic use.

After this open cmd.exe and type ::

    easy_install hddm


Linux (Debian based, such as Ubuntu)
-----------------------------------------------------------

The following commands require admin rights

::

    aptitude install python python-numpy python-scipy python-matplotlib cython python-pip gfortran lapack-dev

You can either install the package automatically from pypi:

::

    pip install hddm

Or, you can install the package from the source directory:

::

    python setup.py install


Getting started
===============

Check out the documentation_ for a manual and tutorial for how to use HDDM.

.. _HDDM: http://code.google.com/p/hddm/
.. _Python: http://www.python.org/
.. _PyMC: http://code.google.com/p/pymc/
.. _Cython: http://www.cython.org/
.. _DMAT: http://ppw.kuleuven.be/okp/software/dmat/
.. _fast-dm: http://seehuhn.de/pages/fast-dm
.. _documentation: http://ski.cog.brown.edu/hddm_docs
.. _kabuki: https://github.com/hddm-devs/kabuki
.. _Enthought Python Distribution: http://www.enthought.com/products/edudownload.php
