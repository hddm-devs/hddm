************
Introduction
************

:Date: July 4, 2011
:Author: Thomas V. Wiecki, Imri Sofer
:Contact: thomas_wiecki@brown.edu, imri_sofer@brown.edu
:Web site: http://github.com/twiecki/hddm
:Copyright: This document has been placed in the public domain.
:License: HDDM is released under the GPLv3.
:Version: 0.1RC1

Purpose
=======

HDDM is a python module that implements Hierarchical Bayesian estimation of Drift Diffusion Models (via PyMC).

Features
========

HDDM provides functionalities to make Drift Diffusion analysis of reaction times as painless as 
possible. Here is a short list of some of its features:

* Uses hierarchical bayesian estimation (via PyMC) of DDM parameters to allow simultanious estimation of subject and group parameters. HDDM can thus produce better estimates when less RT values are measured compared to other methods using maximum likelihood (i.e. `DMAT`_ or `fast-dm`_).

* Heavily optimized likelihood functions for speed.

* Flexible creation of complex models tailored to specific hypotheses (e.g. separate drift-rate parameters for different stimulus types).

* Easy specification of models via configuration file fosters exchange of models and research results.

* Built-in Bayesian hypothesis testing and several convergence and goodness-of-fit diagnostics.

Usage
=====

The easiest way to use HDDM is by creating a configuration file for your model:

example.conf
::

    [model]
    data = examples/simple_difficulty.csv

    [depends]
    v = difficulty

    [mcmc]
    samples=10000 # Sample 5000 posterior samples
    burn=5000 # Discard the first 100 samples as burn-in
    thin=3 # Discard every third sample to remove autocorrelations

Then call hddm:

::

    hddmfit example.conf

Installing
==========

HDDM has the following dependencies:

* Python

* NumPy

* SciPy

* Matplotlib

* Cython

* PyMC

Linux (Debian based, such as Ubuntu)
------------------------------------

::

    sudo aptitude install python python-numpy python-scipy python-matplotlib cython
    pip install hddm

Getting started
===============

This guide provides all the information needed to install HDDM, create configuration files, build your own models and save and visualize the results.
More `examples`_ of usage as well as `tutorials`_  are available from the PyMC web site.
