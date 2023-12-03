************
Introduction
************

:Author: Thomas V. Wiecki, Imri Sofer, Mads L. Pedersen, Alexander Fengler, Lakshmi Govindarajan, Krishn Bera, Michael J. Frank
:Contact: thomas.wiecki@gmail.com, imri_sofer@brown.edu, madslupe@gmail.com, alexander_fengler@brown.edu, krishn_bera@brown.edu, michael_frank@brown.edu
:Web site: https://hddm.readthedocs.io
:Github: http://github.com/hddm-devs/hddm
:Mailing list: https://groups.google.com/group/hddm-users/
:Copyright: This document has been placed in the public domain.
:License: HDDM is released under the BSD 2 license.
:Version: 1.0.1

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
  
* HDDM 0.9.0 brings a host of new features. 
             HDDM includes `likelihood approximation networks`_ via the **HDDMnn**, **HDDMnnRegressor** and **HDDMnnStimCoding** classes. 
             This allows fitting of a number of variants of sequential sampling models. You can now easily use custom likelihoods
             for model fitting. We included a range of new **simulators**, which allow data generation for a host of variants of sequential sampling models.
             There are some new out of the box **plots**, in the **hddm.plotting** module. Fast posterior predictives for regression based models.
             Some sampler settings are now exposed to the user via a customizable **model_config dictionary**. Lastly you are now able to save and load **HDDMRegression** models with 
             custom link functions. Please see the **documentation** (under **LAN Extension**) for illustrations on how to use the new features.

* HDDM 0.9.1 improved documentation for LAN models. 
             Comprehensive tutorial using LAN included. Bugfixes for ``simulator_h_c()`` function. 

* HDDM 0.9.2 major overhaul of the plotting functions under hddm.plotting. 
             Old capabilities are preserved under ``hddm.plotting_old``, but will be deprecated. 
             The new plotting functions replicate the existing functionality, but improve on various aspects of the plot and provide a more abstracted and extensible interface.
             Fixes an error with posterior predictive sampling using hierarchical regression models based on LANs with ``HDDMnnRegressor()``. ``HDDMnnRegressor()`` now issues a 
             single warning for boundary condition violations instead of flagging all occurences.

* HDDM 0.9.3 Lot's of minor improvements.
             Model plots now working for race and lca models with n > 2 choices (use **_plot_func_model_n** as **plot_func** argument in **hddm.plotting.plot_posterior_predictive**).
             **model_config** files are simplified and class construction is a bit more robust to lack of specification, improving ease of use with custom models.
             Various plots received a bit more styling features.
             Better defaults for **simulator_h_c** function in **hddm.simulators.hddm_dataset_generators**.
             Posterior predictives now properly take into account the *p_outlier* parameter when generating data from the implicit mixture model. *p_outlier* percent of the data,
             now explicitly come from random choices with uniform reaction times.
             The documentation is updated to reflect / illustrate the improvements.

* HDDM 0.9.4 Bug fixes and one major new functionality.
             **HDDMnnRegressor** now allows you to define **indirect regressors**, latent parameters that are driven by their own regression and link to model parameters.
             See the documentation for more information on this. **Note** this functionality is experimental for now. Model fitting will work, but extraenous functionality may not,
             including posterior predictives for models that include such indirect regressors. Including indirect regressors might demand you to think carefully about the supplied 
             **model_config**. E.g. in the **race_no_bias_3** model, the usual lower bounds on the **v0, v1 ,v2, v3** parameters are 0. If we allow these parameters to be driven by an 
             indirect regressor **v**, which is added to the regressions of **v0, v1, v2, v3**, then **v0, v1, v2, v3**

* HDDM 0.9.5 Bug fixes and another new functionality.
             **HDDMnnRegressor** now allows you to also define **indirect betas**, latent parameters that can be used in regression models. 
             E.g. in the **race_no_bias_3** model, you can define a **v** beta, which will be expressed in the regression models 
             for **v0**, **v1**, **v2** like so:

             **v0 = beta0 * covariate0 + ... + v * covariate_v_0**

             **v1 = beta0 * covariate0 + ... + v * covariate_v_1**

             **v2 = beta0 * covariate0 + ... + v * covariate_v_2**

             The **v0**, **v1**, **v2** parameters might be drifts in a preference based choice task, and dedicated to respective choice 
             options in a stimulus screen. You may be interested in making these slopes dependent on the respective values of these options
             in a given trial, but wish to use a central **beta**, (**v** in this example) that relates option value to drift for all drifts.

             Note that first, this is **complementary** to the **indirect regressors** defined above, and second that the usual modelling caveats,
             such as potential convergence problems apply. 

             Note also that the usage of **indirect betas** as well as **indirect regressors** may affect the speed of sampling in general.
             Both translate into more computational work at the stage of regression likelihood evaluation.

* HDDM 0.9.6 brings a host of new features. 
             HDDM now includes use of `likelihood approximation networks`_ in conjunction with reinforcement learning models via the **HDDMnnRL** class. 
             This allows researchers to study not only the across-trial dynamics of learning but the within-trial dynamics of choice processes, using a single model. 
             This module greatly extends the previous functionality for fitting RL+DDM models (via HDDMrl class) by allowing fitting of a number of variants of sequential sampling models in conjuction with a learning process (RL+SSM models).
             We have included a new **simulator**, which allows data generation for a host of variants of sequential sampling models in conjunction with the Rescorla-Wagner update rule on a 2-armed bandit task environment.
             There are some new, out-of-the-box **plots** and **utility function** in the **hddm.plotting** and **hddm.utils** modules, respectively, to facilitate posterior visualization and posterior predictive checks.
             Lastly you can also save and load **HDDMnnRL** models. 
             Please see the **documentation** (under **HDDMnnRL Extension**) for illustrations on how to use the new features.

* HDDM 0.9.7 adds the **HDDMnnRLRegressor** class, the equivalent to the **HDDMrlRegressor** with support for many more *SSMs* via *LANs*. 
             Please check the documentation for usage examples.

* HDDM 0.9.8 adds a **breaking change**. 
             To accommodate some user requests, **all parameters** that should be estimated for a given model should be made explicitly
             in the *include* argument of the calling class (HDDM, HDDMnn, etc.). The remaining parameters can now be set to arbitrary, user defined, defaults.
             Check the documentation for a **new tutorial on parameter defaults**.
             Moreover, **model plot** is made much more flexible (new tutorial included to showcase some of the options).
             Two **tutorials** are added to showcase the capabilities for simulation based inference via **custom likelihoods**. 
             The legacy models with "vanilla" in their name are globally renamed to instead include "hddm_base". 
             The simulator backend is now completely outsourced to the  `ssms`_ package (severe code simplifications).


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
please see the `main publication`_.

Installation
============

For **HDDM >= 0.9.0**, currently in beta release, the most convenient way to install HDDM, is to directly 
install via **github**. In a fresh environment (we recommend to use **python 3.7**) type.

We recommend you to open a **conda** environment first. 

::

    conda create -n hddm python=3.7
    conda activate hddm

If you do not have **hdf5** or **netcdf4** installed, you can use conda to install them.

::

    conda install -c conda-forge hdf5
    conda install -c conda-forge netcdf4

Then install **hddm** via **github**.

:: 

    pip install cython
    pip install pymc==2.3.8 # backend probabilistic programming framework (DO NOT USE CONDA HERE)
    pip install git+https://github.com/hddm-devs/kabuki # backbone package for hddm to connect to pymc
    pip install git+https://github.com/hddm-devs/hddm 
    
    # Optional
    pip install torch torchvision torchaudio # The LAN extension makes use of these

To make use of the LAN fuctionalities, you need to install `pytorch`_.

::
    pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1

A common issue on new machines is that the installation of the **pymc** package (a necessary dependency),
is hampered by problems with compiling its fortran code. Try downgrading the version of your
**gcc** compiler. This can be done on a MAC (not the new M1/M2 versions tragically), via 

::

    brew install gcc@9

In case you do not have the **brew** command, install `Homebrew <https://brew.sh/>`_ first.

You usually do not run into problems with **linux** machines, however downgrading **gcc** can still be necessary.

(Previous instructions for **HDDM <= 0.8.0**, DISCOURAGED)
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
    pip install pymc==2.3.8
    pip install kabuki
    pip install hddm

This might require super-user rights via sudo. Note that this
installation method is discouraged as it leads to all kinds of
problems on various platforms.

If you are having installation problems please contact the `mailing list`_.

And if you're a mac user, check out this `thread`_ for advice on installation.

How to cite
===========

If HDDM was used in your research, please cite the `main publication`_:

Wiecki TV, Sofer I and Frank MJ (2013). HDDM: Hierarchical Bayesian estimation of the Drift-Diffusion Model in Python.
Front. Neuroinform. 7:14. doi: 10.3389/fninf.2013.00014

If you use the HDDMrl, please cite the `original HDDM RL tutorial paper`_:

Pedersen, M. L., & Frank, M. J. (2020). Simultaneous hierarchical bayesian parameter estimation for reinforcement learning and drift diffusion models: a tutorial and links to neural data. 
Computational Brain & Behavior, 3(4), 458-471.

If you use any of the HDDMnn, HDDMnnRegressor, HDDMnnStimCoding or HDDMnnRL classes, please cite the `lan extension`_ and the `new tutorial paper`_:

Alexander Fengler, Lakshmi N Govindarajan, Tony Chen, Michael J Frank (2021). Likelihood approximation networks (LANs) for fast inference of simulation models in cognitive neuroscience.
eLife 10:e65074. doi: 10.7554/eLife.65074

Fengler, A., Bera, K., Pedersen, M. L., & Frank, M. J. (2022). Beyond Drift Diffusion Models: Fitting a Broad Class of Decision and Reinforcement Learning Models with HDDM. 
Journal of Cognitive Neuroscience, 34(10), 1780-1805.


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
.. _tutorial: https://hddm.readthedocs.io/en/latest/tutorial_basic_hddm.html
.. _howto: https://hddm.readthedocs.io/en/latest/howto.html
.. _manual: http://ski.clps.brown.edu/hddm_docs/manual.html
.. _kabuki: https://github.com/hddm-devs/kabuki
.. _mailing list: https://groups.google.com/group/hddm-users/
.. _SciPy Superpack: http://fonnesbeck.github.com/ScipySuperpack/
.. _Anaconda: http://docs.continuum.io/anaconda/install.html
.. _main publication: http://www.frontiersin.org/Journal/10.3389/fninf.2013.00014/abstract
.. _lan extension: https://elifesciences.org/articles/65074
.. _new tutorial paper: https://direct.mit.edu/jocn/article/34/10/1780/112585/Beyond-Drift-Diffusion-Models-Fitting-a-Broad
.. _original HDDM RL tutorial paper: https://link.springer.com/article/10.1007/s42113-020-00084-w
.. _published papers: https://scholar.google.com/scholar?oi=bibs&hl=en&cites=17737314623978403194
.. _thread: https://groups.google.com/forum/#!topic/hddm-users/bdQXewfUzLs
.. _ssms: https://github.com/AlexanderFengler/ssms
