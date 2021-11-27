************
Introduction
************

:Author: Thomas V. Wiecki, Imri Sofer, Mads L. Pedersen, Alexander Fengler, Michael J. Frank
:Contact: thomas.wiecki@gmail.com, imri_sofer@brown.edu, madslupe@gmail.com, alexander_fengler@brown.edu, michael_frank@brown.edu
:Web site: https://hddm.readthedocs.io
:Github: http://github.com/hddm-devs/hddm
:Mailing list: https://groups.google.com/group/hddm-users/
:Copyright: This document has been placed in the public domain.
:License: HDDM is released under the BSD 2 license.
:Version: 0.9.0

.. image:: https://secure.travis-ci.org/hddm-devs/hddm.png?branch=master

Purpose
=======

HDDM is a python toolbox for hierarchical Bayesian parameter
estimation of the Drift Diffusion Model (via PyMC). Drift Diffusion
Models are used widely in psychology and cognitive neuroscience to
study decision making.

Check out the tutorial_ on how to get started. Further information can be found below as well as in the howto_ section and the documentation_.

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

* Estimate trial-by-trial correlations between a brain measure
  (e.g. fMRI BOLD) and a diffusion model parameter using the
  `HDDMRegression` model.

* Built-in Bayesian hypothesis testing and several convergence and
  goodness-of-fit diagnostics.

* As of version 0.7.1 HDDM includes modules for analyzing reinforcement learning data with the reinforcement learning drift diffusion   
  model (RLDDM), including a module for estimating the impact of continuous regressors onto RLDDM parameters, and a reinforcement learning 
  (RL) model. See tutorial for the RLDDM and RL modules here: https://nbviewer.jupyter.org/github/hddm-devs/hddm/blob/master/hddm/examples/demo_RLHDDMtutorial.ipynb and in the paper here: https://rdcu.be/b4q6Z
  
* HDDM 0.9.0 brings a host of new features. HDDM includes `likelihood approximation networks`_ via the **HDDMnn**, **HDDMnnRegressor** and **HDDMnnStimCoding** classes. 
  This allows fitting of a number of variants of sequential sampling models. You can now easily use custom likelihoods
  for model fitting. We included a range of new **simulators**, which allow data generation for a host of variants of sequential sampling models.
  There are some new out of the box **plots**, in the **hddm.plotting** module. Fast posterior predictives for regression based models.
  Some sampler settings are now exposed to the user via a customizable **model_config dictionary**. Lastly you are now able to save and load **HDDMRegression** models with 
  custom link functions. Please see the **documentation** (under **LAN Extension**) for illustrations on how to use the new features.


Comparison to other packages
============================

A recent paper by Roger Ratcliff quantitatively compared DMAT, fast-dm, and EZ, and concluded: "We found that the hierarchical diffusion method [as implemented by HDDM] performed very well, and is the method of choice when the number of observations is small."

Find the paper here: http://www.ncbi.nlm.nih.gov/pmc/articles/PMC4517692/

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
   model.sample(2000, burn=20)

   # Print fitted parameters and other model statistics
   model.print_stats()

   # Plot posterior distributions and theoretical RT distributions
   model.plot_posteriors()
   model.plot_posterior_predictive()


For more information about the software and theories behind it,
please see the main `publication`_.

Installation
============

For **HDDM >= 0.9.0**, currently in beta release, the most convenient way to install HDDM, is to directly 
install via git. In a fresh environment type:

:: 
    pip install cython
    pip install pymc
    pip install git+htpts://github.com/hddm-devs/kabuki
    pip install git+https://github.com/hddm-devs/hddm
    # Optional
    pip install torch torchvision torchaudio

To make use of the LAN fuctionalities, need actually need to install `pytorch`_ .

A common issue is that the installation of the **pymc** package (a necessary dependency),
is hampered by issues with compiling its fortran code. Try downgrading you the version of your
**gcc* compiler. This can be done on a MAC (not the new M1 versions tragically), via 

::

    brew install gcc@9

In case you do not have the **brew** command, install `Homebrew <https://brew.sh/>`_ first.

You usually do not run into problems with **linux** machines.


(Previous instructions for **HDDM <= 0.8.0**)
As of release 0.6.0, HDDM is compatible with Python 3 which we encourage.

The easiest way to install HDDM is through Anaconda (available for
Windows, Linux and OSX):

1. Download and install `Anaconda`_.
2. In a shell (Windows: Go to Start->Programs->Anaconda->Anaconda command prompt) type:

::

    conda install -c pymc hddm

If you want to use pip instead of conda, type:

::

    pip install pandas
    pip install pymc
    pip install kabuki
    pip install hddm

This might require super-user rights via sudo. Note that this
installation method is discouraged as it leads to all kinds of
problems on various platforms.

If you are having installation problems please contact the `mailing list`_.

And if you're a mac user, check out this `thread`_ for advice on installation.

How to cite
===========

If HDDM was used in your research, please cite the publication_:

Wiecki TV, Sofer I and Frank MJ (2013). HDDM: Hierarchical Bayesian estimation of the Drift-Diffusion Model in Python.
Front. Neuroinform. 7:14. doi: 10.3389/fninf.2013.00014

Published papers using HDDM
===========================

HDDM has been used in over 400 `published papers`_.

Testimonials
============

James Rowe (Cambridge University): "The HDDM modelling gave insights into the effects of disease that were simply not visible from a traditional analysis of RT/Accuracy. It provides a clue as to why many disorders including PD and PSP can give the paradoxical combination of akinesia and impulsivity. Perhaps of broader interest, the hierarchical drift diffusion model turned out to be very robust. In separate work, we have found that the HDDM gave accurate estimates of decision parameters with many fewer than 100 trials, in contrast to the hundreds or even thousands one might use for ‘traditional’ DDMs. This meant it was realistic to study patients who do not tolerate long testing sessions."

Getting started
===============

Check out the tutorial_ on how to get started. Further information can be found in howto_ and the documentation_.

Join our low-traffic `mailing list`_.

.. _likelihood approximation networks: https://elifesciences.org/articles/65074
.. _pytorch: http://pytorch.org
.. _HDDM: http://code.google.com/p/hddm/
.. _Python: http://www.python.org/
.. _PyMC: http://pymc-devs.github.com/pymc/
.. _Cython: http://www.cython.org/
.. _DMAT: http://ppw.kuleuven.be/okp/software/dmat/
.. _fast-dm: http://seehuhn.de/pages/fast-dm
.. _documentation: https://hddm.readthedocs.io
.. _tutorial: https://hddm.readthedocs.io/en/latest/tutorial.html
.. _howto: https://hddm.readthedocs.io/en/latest/howto.html
.. _manual: http://ski.clps.brown.edu/hddm_docs/manual.html
.. _kabuki: https://github.com/hddm-devs/kabuki
.. _mailing list: https://groups.google.com/group/hddm-users/
.. _SciPy Superpack: http://fonnesbeck.github.com/ScipySuperpack/
.. _Anaconda: http://docs.continuum.io/anaconda/install.html
.. _publication: http://www.frontiersin.org/Journal/10.3389/fninf.2013.00014/abstract
.. _published papers: https://scholar.google.com/scholar?oi=bibs&hl=en&cites=17737314623978403194
.. _thread: https://groups.google.com/forum/#!topic/hddm-users/bdQXewfUzLs
