Abstract
========

Diffusion models are a popular tool to infer latent psychological
processes underlying decision making. Efficient open source software
to fit these models to data has been made available in different
programming languages. However, current estimation methods require an
abundance of reaction time measurements to recover meaningful
parameters. Here, we present HDDM which stands for Hierarchical Drift
Diffusion Modeling and allows fast, flexible, and easy hierarchical
Bayesian estimation of the the drift-diffusion model and the linear
ballistic accumulator. Instead of estimating parameters for each
subject separately like current solutions, HDDM allows simultaneous
estimation of group and subject parameters and thus requires less data
per subject. HDDM does not only provide best fitting parameters, but
rather the whole posterior distribution of each parameter and thus
allows for full Bayesian data analysis. Finally, HDDM supports the
estimation of how trial-by-trial measurements of brain activity
(e.g. fMRI) influence decision making parameters.
