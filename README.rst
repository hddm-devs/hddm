************
Introduction
************

:Author: Thomas V. Wiecki, Imri Sofer, Michael J. Frank
:Contact: thomas_wiecki@brown.edu, imri_sofer@brown.edu, michael_frank@brown.edu
:Web site: http://ski.clps.brown.edu/hddm_docs
:Github: http://github.com/hddm-devs/hddm
:Mailing list: https://groups.google.com/group/hddm-users/
:Copyright: This document has been placed in the public domain.
:License: HDDM is released under the BSD 2 license.
:Version: 0.5

.. image:: https://secure.travis-ci.org/hddm-devs/hddm.png?branch=develop

Purpose
=======

HDDM is a python toolbox for hierarchical Bayesian parameter
estimation of the Drift Diffusion Model (via PyMC). Drift Diffusion
Models are used widely in psychology and cognitive neuroscience to
study decision making.

Features
========

* Uses hierarchical Bayesian estimation (via PyMC) of DDM parameters
  to allow simultaneous estimation of subject and group parameters,
  where individual subjects are assumed to be drawn from a group
  distribution. HDDM should thus produce better estimates when less RT
  values are measured compared to other methods using maximum
  likelihood for individual subjects (i.e. `DMAT`_ or `fast-dm`_).

* Heavily optimized likelihood functions for speed (Navarro & Fuss, 2009).

* Flexible creation of complex models tailored to specific hypotheses
  (e.g. estimation of separate drift-rates for different task
  conditions; or predicted changes in model parameters as a function
  of other indicators like brain activity).

* Easy specification of models via configuration file fosters exchange
  of models and research results.

* Built-in Bayesian hypothesis testing and several convergence and
  goodness-of-fit diagnostics.

Quick-start
===========

The following is a minimal python script to load data, run a model and
examine its parameters and fit.

::

   import hddm

   # Load data from csv file into a NumPy structured array
   data = hddm.load_csv('simple_difficulty.csv')

   # Create a HDDM model multi object
   model = hddm.HDDM(data, depends_on={'v':'difficulty'})

   # Create model and start MCMC sampling
   model.sample(10000, burn=5000)

   # Print fitted parameters and other model statistics
   model.print_stats()

   # Plot posterior distributions and theoretical RT distributions
   model.plot_posteriors(save=True)
   model.plot_posterior_predictive(save=True)


Installation
============

Windows
-------

The easiest way is to download and install `Anaconda`_ or the
`Enthought Python Distribution`_ (EPD) which is free for academic
use.

We recommend using pip to download and install HDDM. The easiest way
to install pip is via easy_install. Start the windows command shell
(cmd.exe) and type ::

    easy_install pip

Then install kabuki and HDDM:

::

    pip install kabuki
    pip install hddm


Linux (Debian based, such as Ubuntu)
------------------------------------

Most of HDDM's dependencies are available from your repository, you can install them by typing

::

    apt-get install python python-dev python-numpy python-scipy python-matplotlib cython python-pip gfortran liblapack-dev

which requires sudo rights.

Optional dependencies for hddm_demo.py can be installed via

::

    apt-get install python-wxgtk2.8 python-traitsui

Then install kabuki and HDDM:

::

    sudo pip install kabuki
    sudo pip install hddm

OSX
---

We recommend installing the `SciPy Superpack`_ maintained by Chris Fonnesbeck.

The Superpack requires you to install XCode which apparently does not bundle with gcc anymore (which is required by HDDM). This repository provides some appropriate installers:

https://github.com/kennethreitz/osx-gcc-installer/downloads

Then install kabuki and HDDM:

::

    sudo pip install kabuki
    sudo pip install hddm


How to cite
===========

If HDDM was used in your research, please cite the following:

Wiecki, T. V., Sofer, I. and Frank, M. J. Hierarchical Bayesian estimation of the drift diffusion model: quantitative comparison with maximum likelihood. Program No. 494.13/CCC30. 2012 Neuroscience Meeting Planner. New Orleans, LA: Society for Neuroscience, 2012.

Getting started
===============

Check out the tutorial_ on how to get started. Further information can be found in howto_ and the documentation_.

Join our low-traffic `mailing list`_.

.. _HDDM: http://code.google.com/p/hddm/
.. _Python: http://www.python.org/
.. _PyMC: http://pymc-devs.github.com/pymc/
.. _Cython: http://www.cython.org/
.. _DMAT: http://ppw.kuleuven.be/okp/software/dmat/
.. _fast-dm: http://seehuhn.de/pages/fast-dm
.. _documentation: http://ski.clps.brown.edu/hddm_docs
.. _tutorial: http://ski.clps.brown.edu/hddm_docs/tutorial.html
.. _howto: http://ski.clps.brown.edu/hddm_docs/howto.html
.. _manual: http://ski.clps.brown.edu/hddm_docs/manual.html
.. _kabuki: https://github.com/hddm-devs/kabuki
.. _Enthought Python Distribution: http://www.enthought.com/products/edudownload.php
.. _mailing list: https://groups.google.com/group/hddm-users/
.. _SciPy Superpack: http://fonnesbeck.github.com/ScipySuperpack/
.. _Anaconda: http://continuum.io/anaconda
