Abstract
========

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
HDDM supports the estimation of how trial-by-trial measurements of
brain activity (e.g. fMRI) influence decision making parameters.
