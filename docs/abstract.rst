Abstract
========

Diffusion models are used to infer psychological processes underlying
decision making. Efficient open source software to fit these models to
data has been made available in different programming
languages. However, current estimation methods require an abundance of
reaction time measurements to recover meaningful parameters. Here, we
present HDDM which stands for Hierarchical Drift Diffusion Modeling
and allows fast and easy bayesian estimation of the model parameters
using PyMC. Instead of estimating parameters for each subject
individually like other tools, HDDM allows simultanious estimation of
group and subject parameters and thus requires less data per
subject. HDDM does not only provide best fitting parameters, but
rather the whole posterior distribution of each parameter which allows
intuitive goodness-of-fit statistics.
