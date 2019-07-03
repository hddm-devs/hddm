************
Introduction
************

:Author: Thomas V. Wiecki, Imri Sofer, Michael J. Frank
:Contact: thomas.wiecki@gmail.com, imri_sofer@brown.edu, michael_frank@brown.edu
:Web site: http://ski.clps.brown.edu/hddm_docs
:Github: http://github.com/hddm-devs/hddm
:Mailing list: https://groups.google.com/group/hddm-users/
:Copyright: This document has been placed in the public domain.
:License: HDDM is released under the BSD 2 license.
:Version: 0.7.1

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

How to cite
===========

If HDDM was used in your research, please cite the publication_:

Wiecki TV, Sofer I and Frank MJ (2013). HDDM: Hierarchical Bayesian estimation of the Drift-Diffusion Model in Python.
Front. Neuroinform. 7:14. doi: 10.3389/fninf.2013.00014

Published papers using HDDM
===========================

* Cavanagh, J. F., Wiecki, T. V, Cohen, M. X., Figueroa, C. M., Samanta, J., Sherman, S. J., & Frank, M. J. (2011). Subthalamic nucleus stimulation reverses mediofrontal influence over decision threshold. Nature Neuroscience, 14(11), 1462–7. doi:10.1038/nn.2925

* Jahfari, S., Ridderinkhof, K. R., & Scholte, H. S. (2013). Spatial frequency information modulates response inhibition and decision-making processes. PloS One, 8(10), e76467. doi:10.1371/journal.pone.0076467

* Zhang, J., & Rowe, J. B. (2014). Dissociable mechanisms of speed-accuracy tradeoff during visual perceptual learning are revealed by a hierarchical drift-diffusion model. Frontiers in Neuroscience, 8, 69. doi:10.3389/fnins.2014.00069

* Cavanagh, J. F., Wiecki, T. V, Kochar, A., & Frank, M. J. (2014). Eye Tracking and Pupillometry Are Indicators of Dissociable Latent Decision Processes. Journal of Experimental Psychology. General. doi:10.1037/a0035813

* Dunovan, K. E., Tremel, J. J., & Wheeler, M. E. (2014). Prior probability and feature predictability interactively bias perceptual decisions. Neuropsychologia. doi:10.1016/j.neuropsychologia.2014.06.024

* Michmizos, K. P., & Krebs, H. I. (2014). Reaction time in ankle movements: a diffusion model analysis. Experimental Brain Research. doi:10.1007/s00221-014-4032-8

* Wedel, M., & Pieters, R. (2014). The Buffer Effect: The Role of Color When Advertising Exposures Are Brief and Blurred. Marketing Science. doi:10.1287/mksc.2014.0882 

* Ratcliff, R. & Childers, R. (2014). Individual Differences and Fitting Methods for the Two-Choice Diffusion Model of Decision Making. http://star.psy.ohio-state.edu/coglab/People/roger/pdf/lownfinaldec14.pdf

* Frank, M.J., Gagne, C., Nyhus, E., Masters, S., Wiecki, T.V., Cavanagh, J.F. & Badre, D. (2015). fMRI and EEG Predictors of Dynamic Decision Parameters during Human Reinforcement Learning. Journal of Neuroscience, 35, 484-494.

* Tremel, J.J., & Wheeler M.E. (2015) Content-specific evidence accumulation in inferior temporal cortex during perceptual decision-making. NeuroImage, 109, 35-49

* Assink, N., van der Lubbe, R.H.J., & Fox, J-P (2015) Does Time Pressure Induce Tunnel Vision? An examination with the Eriksen Flanker Task by applying the Hierarchical Drift Diffusion Model. New Developments in Computational Intelligence and Computer Science http://www.inase.org/library/2015/vienna/bypaper/APNE/APNE-04.pdf

* Zhang, J., Rittman, T., Nombela, C., Fois, A., Coyle-Gilchrist, I., Barker, R. A., Hughes, L. E., Rowe, J. B, (2015) Different decision deficits impair response inhibition in progressive supranuclear palsy and Parkinson’s disease. Brain, 1-13. http://brain.oxfordjournals.org/content/brain/early/2015/11/17/brain.awv331.full.pdf

* Nie, Q., Maurer, M., Müller, H., & Conci, M., (2016) Inhibition drives configural superiority of illusory Gestalt: Combined behavioral and drift–diffusion model evidence. Cognition, 150, 150-162 http://www.sciencedirect.com/science/article/pii/S001002771630035X

* Herz, D. M., Zavala, B. A., Bogacz, R., Brown, P. (2016) Neural Correlates of Decision Thresholds in the Human Subthalamic Nucleus. Current Biology, http://www.cell.com/current-biology/fulltext/S0960-9822%2816%2900133-0

* Mathias SR., Knowles EE., Barrett J., Leach O., Buccheri S., Beetham T., Blangero J., Poldrack R.A., Glahn D.C. (2017) The Processing-Speed Impairment in Psychosis Is More Than Just Accelerated Aging.. Schizophr Bull. https://www.ncbi.nlm.nih.gov/pubmed/28062652

* Steinweg, B., & Mast, F. W. (2017). Semantic incongruity influences response caution in audio-visual integration. Experimental brain research, 235(1), 349-363. 

* de Gee JW, Colizoli O, Kloosterman NA, Knapen T, Nieuwenhuis S & Donner TH (2017). Dynamic modulation of decision biases by brainstem arousal systems. eLife. 2017 Apr 6;6. pii: e23232. doi: 10.7554/eLife.23232.

Testimonials
============

James Rowe (Cambridge University): "The HDDM modelling gave insights into the effects of disease that were simply not visible from a traditional analysis of RT/Accuracy. It provides a clue as to why many disorders including PD and PSP can give the paradoxical combination of akinesia and impulsivity. Perhaps of broader interest, the hierarchical drift diffusion model turned out to be very robust. In separate work, we have found that the HDDM gave accurate estimates of decision parameters with many fewer than 100 trials, in contrast to the hundreds or even thousands one might use for ‘traditional’ DDMs. This meant it was realistic to study patients who do not tolerate long testing sessions."

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
.. _Anaconda: http://docs.continuum.io/anaconda/install.html
.. _publication: http://www.frontiersin.org/Journal/10.3389/fninf.2013.00014/abstract
