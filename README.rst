************
Introduction
************

:Date: September 11, 2010
:Author: Thomas Wiecki
:Contact: thomas_wiecki@brown.edu
:Web site: http://code.google.com/p/hddm
:Copyright: This document has been placed in the public domain.
:License: PyMC is released under the GPLv3.
:Version: 0.1

Purpose
=======

HDDM is a python module that implements Hierarchical Bayesian estimation of Drift Diffusion Models (via PyMC).

Features
========

HDDM provides functionalities to make Drift Diffusion analysis of reaction times as painless as 
possible. Here is a short list of some of its features:

* Uses hierchical bayesian estimation (via PyMC) of DDM parameters to allow simultanious estimation of subject and group parameters. HDDM can thus produce better estimates when less RT values are measured compared to other methods using maximum likelihood (i.e. `DMAT`_ or `fast-dm`_).

* Heavily optimized likelihood functions for speed (with experimental GPU support).

* Flexible creation of complex models tailored to specific hypotheses (e.g. separate drift-rate parameters for different stimulus types).

* Easy specification of models via configuration file fosters exchange of models and research results.

* Built-in Bayesian hypothesis testing and several convergence and goodness-of-fit diagnostics.

* Supports the following Drift Model types:
    * Simple DDM: Ratcliff DDDM without taking intra-trial variabilities into account.
    * Full-MC DDM: Ratcliff DDM taking intra-trial variabilities into account by using Monte-Carlo integration.
    * Full DDM: Ratcliff DDM with full hierarchical bayesian parameter estimation of intra-trial variabilites.
    * LBA: Linear Ballistic Accumulator (Brown et al.)


Usage
=====

The easiest way to use HDDM is by creating a configuration file for your model:

example.conf
::

    [data]
    load = data/example_subj.csv # Main data file containing all RTs in csv format
    save = data/example_subj_out.txt # Estimated parameters and stats will be written to this file

    [model]
    type = simple # Use the simple DDM model that takes not intra-trial variabilites into account (and is faster).
    is_subj_model = True # Create separate distributions for each subject that feed into a group level distribution.

    [depends]

    [mcmc]
    samples=5000 # Sample 5000 posterior samples
    burn=2000 # Discard the first 100 samples as burn-in
    thin=3 # Discard every third sample to remove autocorrelations

Then call hddm:

::
    hddm.py example.conf

Getting started
===============

This guide provides all the information needed to install HDDM, create configuration files, build your own models and save and visualize the results.
More `examples`_ of usage as well as `tutorials`_  are available from the PyMC web site.
