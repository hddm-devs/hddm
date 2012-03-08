************
Introduction
************

:Date: March 8, 2012
:Author: Thomas V. Wiecki, Imri Sofer, Michael J. Frank
:Contact: thomas_wiecki@brown.edu, imri_sofer@brown.edu, michael_frank@brown.edu
:Web site: http://github.com/hddm-devs/hddm
:Mailing list: https://groups.google.com/group/hddm-users/
:Copyright: This document has been placed in the public domain.
:License: HDDM is released under the GPLv3.
:Version: 0.2RC5

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

* Heavily optimized likelihood functions for speed (Navarro & Fuss, 2009).

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

Below we provide instructions for how to install HDDM and all it's dependencies.

HDDM relies on the following packages:

* Python

* NumPy

* PyMC_ (installation instructions: http://pymc.googlecode.com/svn/doc/installation.html)

* kabuki_ 

* SciPy (optional)

* Matplotlib (optional)

* Cython_ (optional)


Windows
-------

The easiest way is to download and install the `Enthought Python
Distribution`_ (EPD) which is free for academic use.

After this open cmd.exe and type ::

    easy_install hddm

to install the binary version.

If you want to compile on windows, I found that the EPD
does not seem to have OpenMP support. Downloading the mingw32 compiler
solved the problem and compiled HDDM successfully.

Linux (Debian based, such as Ubuntu)
------------------------------------

The following commands require admin rights

::

    aptitude install python python-dev python-numpy python-scipy python-matplotlib cython python-pip gfortran liblapack-dev

You can either install the package automatically from pypi:

::

    pip install hddm

Or, you can install the package from the source directory:

::

    python setup.py install

OSX
---

We provide an automatic installer based on the scipy superpack by Chris Fonnesbeck. Simply download and run this script_ which should install all dependencies.

Getting started
===============

Check out the documentation_ or the tutorial_ on how to use HDDM.

Join our low-traffic `mailing list`_.

.. _HDDM: http://code.google.com/p/hddm/
.. _Python: http://www.python.org/
.. _PyMC: http://code.google.com/p/pymc/
.. _Cython: http://www.cython.org/
.. _DMAT: http://ppw.kuleuven.be/okp/software/dmat/
.. _fast-dm: http://seehuhn.de/pages/fast-dm
.. _documentation: http://ski.cog.brown.edu/hddm_docs
.. _tutorial: http://ski.cog.brown.edu/hddm_docs/tutorial.html
.. _manual: http://ski.cog.brown.edu/hddm_docs/manual.html
.. _kabuki: https://github.com/hddm-devs/kabuki
.. _Enthought Python Distribution: http://www.enthought.com/products/edudownload.php
.. _script: https://raw.github.com/hddm-devs/hddm/master/install_osx.sh
.. _mailing list: https://groups.google.com/group/hddm-users/
