.. HDDM documentation master file, created by
   sphinx-quickstart on Sat Sep 11 16:47:10 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to HDDM's documentation!
================================

HDDM is a python toolkit to fit Drift Diffusion Models to reaction times using Hierarchical Bayesian estimation (via PyMC_).

Quickstart for the impatient
============================

Homepage: http://code.google.com/p/hddm
License: GPLv3

Installation
------------
::

    easy_install hddm

or get the source from the project page or the repository and

::
    setup.py install

Usage
-----

The easiest way to use HDDM is by creating a configuration file for your model:

example.conf
::

    [data]
    load = data/example_subj.csv # Main data file containing all RTs in csv format
    save = data/example_subj_out.txt # Estimated parameters and stats will be written to this file

    [model]
    type = simple # Use the simple DDM model that does not take intra-trial variabilites into account (and is faster).
    is_subj_model = True # Create separate distributions for each subject that feed into a group level distribution.

    [depends]

    [mcmc]
    samples=5000 # Sample 5000 posterior samples
    burn=2000 # Discard the first 100 samples as burn-in
    thin=3 # Discard every third sample to remove autocorrelations

Then call hddm:

::
    hddm.py example.conf


Features
========

HDDM provides functionalities to make Drift Diffusion analysis of reaction times as painless as 
possible. Here is a short list of some of its features:

* Uses hierchical bayesian estimation (via PyMC_) of DDM parameters to allow simultanious estimation of subject and group parameters. HDDM can thus produce better estimates when less RT values are measured compared to other methods (i.e. `DMAT`_ or `fast-dm`_).

* Heavily optimized likelihood functions for speed (with experimental GPU support).

* Flexible creation of complex models tailored to specific hypotheses (e.g. separate drift-rate parameters for different stimulus types).

* Provides many base classes for common model types but also an option to specify a model via a simple configuration file.

* Supports the following Drift Model types:
    * Simple DDM: Ratcliff DDDM without taking intra-trial variabilities into account.
    * Full-average DDM: Ratcliff DDM taking intra-trial variabilities into account by using an averaging method.
    * Full DDM: Ratcliff DDM with full hierarchical bayesian parameter estimation of intra-trial variabilites.
    * LBA: Linear Ballistic Accumulator (Brown et al.)

* Includes a demo application that generates and displays drift
  processes with user-specified parameters in real time.

* Several convergence and goodness-of-fit diagnostics.

Table of Contents:

.. toctree::
    :maxdepth: 2
    
    abstract
    intro
    tutorial
    manual


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _HDDM: http://code.google.com/p/hddm/
.. _Python: http://www.python.org/
.. _PyMC: http://code.google.com/p/pymc/
.. _Cython: http://www.cython.org/
.. _DMAT: http://ppw.kuleuven.be/okp/software/dmat/
.. _fast-dm: http://seehuhn.de/pages/fast-dm
