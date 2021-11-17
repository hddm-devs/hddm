.. index:: Abstract
.. _chap_abstract:

HDDM: Hierarchical Bayesian estimation of the Drift-Diffusion Model in Python
=============================================================================

Thomas V. Wiecki, Imri Sofer, Michael J. Frank

Correspondence: thomas.wiecki@gmail.com

********
Abstract
********

The diffusion model is a commonly used tool to infer latent
psychological processes underlying decision making, and to link them
to neural mechanisms. Although efficient open source software has been
made available to quantitatively fit the model to data, current
estimation methods require an abundance of reaction time measurements
to recover meaningful parameters, and only provide point estimates of
each parameter.  In contrast, hierarchical Bayesian parameter
estimation methods are useful for enhancing statistical power,
allowing for simultaneous estimation of individual subject parameters
and the group distribution that they are drawn from, while also
providing measures of uncertainty in these parameters in the posterior
distribution. Here, we present a novel Python-based toolbox called
HDDM (hierarchical drift diffusion model), which allows fast and
flexible estimation of the the drift-diffusion model and the related
linear ballistic accumulator model. HDDM requires less data per
subject / condition than non-hierarchical method, allows for full
Bayesian data analysis, and can handle outliers in the data.  Finally,
HDDM supports the estimation of how trial-by-trial measurements
(e.g. fMRI) influence decision making parameters. This paper will
first describe the theoretical background of drift-diffusion model and
Bayesian inference. We then illustrate usage of the toolbox on a
real-world data set from our lab. The software and documentation can
be downloaded at: http://ski.clps.brown.edu/hddm_docs/
