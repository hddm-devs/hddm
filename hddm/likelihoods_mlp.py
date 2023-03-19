from inspect import modulesbyfile
import numpy as np
from functools import partial
from kabuki.utils import stochastic_from_dist
from copy import deepcopy

import hddm
from hddm.simulators import *
from hddm.utils import *


def __prepare_indirect_regressors(model_config={}):
    # Prepare indirect regressors
    # From dictionary that has indirect regressors as keys and links to parameters
    # To dictionary that has parameters as keys and links them to any potential indirect regressor
    param_links = {}
    indirect_regressors_present = False
    if "indirect_regressors" in model_config:
        indirect_regressors_present = True
        for indirect_regressor_tmp in model_config["indirect_regressors"].keys():
            for links_to_tmp in model_config["indirect_regressors"][
                indirect_regressor_tmp
            ]["links_to"]:
                if links_to_tmp in param_links.keys():
                    param_links[links_to_tmp].add(indirect_regressor_tmp)
                else:
                    param_links[links_to_tmp] = set()
                    param_links[links_to_tmp].add(indirect_regressor_tmp)

    # For remaining parameters that haven't been linked to anything
    # we let them link to an empty set
    # If there are not indirect_regressors, all parameters link to the empty set
    for param in model_config["params"]:
        if param in param_links:
            pass
        else:
            param_links[param] = set()

    return param_links, indirect_regressors_present


def __prepare_indirect_betas(model_config={}):
    # Prepare indirect betas
    param_links_betas = {}
    indirect_betas_present = False
    # Loop over indirect betas
    if "indirect_betas" in model_config:
        indirect_betas_present = True
        for indirect_beta_tmp in model_config["indirect_betas"].keys():
            # For particular indirect beta loop over the
            # parameters it links to
            for links_to_tmp in model_config["indirect_betas"][indirect_beta_tmp][
                "links_to"
            ].keys():
                # If param links has respective key already
                # just add the indirect beta to it
                if links_to_tmp in param_links_betas.keys():
                    param_links_betas[links_to_tmp].add(
                        (
                            indirect_beta_tmp,
                            model_config["indirect_betas"][indirect_beta_tmp][
                                "links_to"
                            ][links_to_tmp],
                        )
                    )

                # Otherwise first crete empty set then add the indirect
                # regressor
                else:
                    param_links_betas[links_to_tmp] = set()
                    param_links_betas[links_to_tmp].add(
                        (
                            indirect_beta_tmp,
                            model_config["indirect_betas"][indirect_beta_tmp][
                                "links_to"
                            ][links_to_tmp],
                        )
                    )

    # For remaining parameters that haven't been linked to anything
    # we let them link to an empty set
    # If there are not indirect_parameters, all parameters link to the empty set
    for param in model_config["params"]:
        if param in param_links_betas:
            pass
        else:
            param_links_betas[param] = set()

    return param_links_betas, indirect_betas_present


# LIKELIHOODS
def make_mlp_likelihood(model=None, model_config=None, wiener_params=None, **kwargs):
    """Defines the likelihoods for the MLP networks.

    :Arguments:
        model: str <default='ddm'>
            String that determines which model you would like to fit your data to.
            Currently available models are: 'ddm', 'full_ddm', 'angle', 'weibull', 'ornstein', 'levy'
        model_config: dict <default=None>
            Model config supplied via the calling HDDM class. Necessary for construction of likelihood.
            Should have the structure of model_configs in the hddm.model_config.model_config dictionary.
        kwargs: dict
            Dictionary of additional keyword arguments.
            Importantly here, this carries the preloaded CNN.

    :Returns:
        Returns a pymc.object stochastic object as defined by PyMC2
    """

    def random(
        self,
        keep_negative_responses=True,
        add_model=False,
        add_model_parameters=False,
        add_outliers=False,
        keep_subj_idx=False,
    ):
        """
        Generate random samples from a given model (the dataset matches the size of the respective observated dataset supplied as an attribute of self).
        """

        # This can be simplified so that we pass parameters directly to the simulator ...
        theta = np.array(model_config["params_default"], dtype=np.float32)
        keys_tmp = self.parents.value.keys()
        cnt = 0

        for param in model_config["params"]:
            if param in keys_tmp:
                theta[cnt] = np.array(self.parents.value[param]).astype(np.float32)
            cnt += 1

        sim_out = simulator(theta=theta, model=model, n_samples=self.shape[0], max_t=20)

        # Add outliers:
        if add_outliers:
            if self.parents.value["p_outlier"] > 0.0:
                sim_out = hddm_dataset_generators._add_outliers(
                    sim_out=sim_out,
                    p_outlier=self.parents.value["p_outlier"],
                    max_rt_outlier=1 / wiener_params["w_outlier"],
                )

        sim_out_proc = hddm_preprocess(
            sim_out,
            keep_negative_responses=keep_negative_responses,
            keep_subj_idx=keep_subj_idx,
            add_model_parameters=add_model_parameters,
        )

        if add_model:
            sim_out_proc["model"] = model

        return sim_out_proc

    def pdf(self, x):
        # Check if model supplied has only two choice options
        # If yes --> check if two-dimensional input (rt, response) or one-dimensional input (rt) --> processing depends on it
        # If not --> input x has to be two dimensional (rt, response) becasuse we can't deduce response from rt
        x = np.array(x, dtype=np.float32)

        if len(x.shape) == 1 or x.shape[1] == 1:
            rt = x
            response = rt / np.abs(rt)
            rt = np.abs(rt)
        elif x.shape[1] == 2:
            rt = x[:, 0]
            response = x[:, 1]

        params = np.array(
            [self.parents[param] for param in model_config["params"]]
        ).astype(np.float32)

        return hddm.wfpt.wiener_like_nn_mlp_pdf(
            rt,
            response,
            params,
            p_outlier=self.parents.value["p_outlier"],
            w_outlier=wiener_params["w_outlier"],
            network=kwargs["network"],
        )

    def cdf(self, x):
        # TODO: Implement the CDF method for neural networks
        return "Not yet implemented"

    def make_likelihood():
        likelihood_str = make_likelihood_str_mlp(
            config=model_config, wiener_params=wiener_params
        )
        exec(likelihood_str)
        my_fun = locals()["custom_likelihood"]
        return my_fun

    # TODO: Allow for rt's of -999 in LAN likelihoods
    def make_likelihood_missing_data():
        return

    likelihood_ = make_likelihood()

    wfpt_nn = stochastic_from_dist("Wienernn_" + model, partial(likelihood_, **kwargs))

    wfpt_nn.pdf = pdf
    wfpt_nn.cdf_vec = None  # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
    wfpt_nn.cdf = cdf
    wfpt_nn.random = random
    return wfpt_nn


def make_mlp_likelihood_rlssm(
    model=None, model_config=None, model_config_rl=None, wiener_params=None, **kwargs
):
    """Defines the likelihoods for the MLP networks for RLSSMs.

    :Arguments:
        model: str <default='ddm'>
            String that determines which model you would like to fit your data to.
            Currently available models are: 'ddm', 'full_ddm', 'angle', 'weibull', 'ornstein', 'levy'
        model_config: dict <default=None>
            Config dictionary for the sequential sampling model, necessary for construction of likelihood. In the style of what you find under hddm.model_config.
        model_config_rl: dict <default=None>
            Config dictionary for the reinforcement learning model, necessary for construction of likelihood. In the style of what you find under hddm.model_config_rl.
        kwargs: dict
            Dictionary of additional keyword arguments.
            Importantly here, this carries the preloaded CNN.

    :Returns:
        Returns a pymc.object stochastic object as defined by PyMC2
    """

    def make_likelihood():
        likelihood_str = make_likelihood_str_mlp_rlssm(
            model=model,
            config=model_config,
            config_rl=model_config_rl,
            wiener_params=wiener_params,
        )
        exec(likelihood_str)
        my_fun = locals()["custom_likelihood"]
        return my_fun

    likelihood_ = make_likelihood()

    wfpt_nn_rl = stochastic_from_dist(
        "WienernnRL_" + model, partial(likelihood_, **kwargs)
    )

    return wfpt_nn_rl


# REGRESSOR LIKELIHOODS
def make_mlp_likelihood_reg(
    model=None, model_config=None, wiener_params=None, **kwargs
):
    """Defines the regressor likelihoods for the MLP networks.

    :Arguments:
        model: str <default='ddm'>
            String that determines which model you would like to fit your data to.
            Currently available models are: 'ddm', 'full_ddm', 'angle', 'weibull', 'ornstein', 'levy'
        model_config: dict <default=None>
            Model config supplied via the calling HDDM class. Necessary for construction of likelihood.
            Should have the structure of model_configs in the hddm.model_config.model_config dictionary.
        kwargs: dict
            Dictionary of additional keyword arguments.
            Importantly here, this carries the preloaded CNN.

    :Returns:
        Returns a pymc.object stochastic object as defined by PyMC2
    """

    # Need to rewrite these random parts !
    def random(
        self,
        keep_negative_responses=True,
        add_model=False,
        add_model_parameters=False,
        add_outliers=False,
        keep_subj_idx=False,
    ):
        """
        Function to sample from a regressor based likelihood. Conditions on the covariates.
        """
        param_dict = deepcopy(self.parents.value)
        del param_dict["reg_outcomes"]

        param_data = np.zeros(
            (self.value.shape[0], len(model_config["params"])), dtype=np.float32
        )

        cnt = 0
        for tmp_str in model_config["params"]:
            if tmp_str in self.parents["reg_outcomes"]:
                # param_data[:, cnt] = param_dict[tmp_str].values
                param_data[:, cnt] = param_dict[tmp_str].loc[self.value.index].values

                for linked_indirect_regressor in param_links[tmp_str]:
                    # param_data[:, cnt] = (
                    #     param_data[:, cnt]
                    #     + param_dict[linked_indirect_regressor].values
                    # )

                    param_data[:, cnt] = (
                        param_data[:, cnt]
                        + param_dict[linked_indirect_regressor]
                        .loc[self.value.index]
                        .values
                    )

                for linked_indirect_beta in param_links_betas[tmp_str]:
                    param_data[:, cnt] = (
                        param_data[:, cnt]
                        + param_dict[linked_indirect_beta[0]]
                        * self.value[linked_indirect_beta[1]]
                    )
            else:
                param_data[:, cnt] = param_dict[tmp_str]
            cnt += 1

        sim_out = simulator(
            theta=param_data, model=model, n_samples=1, max_t=20  # n_trials = size,
        )

        # Add outliers:
        if add_outliers:
            if self.parents.value["p_outlier"] > 0.0:
                sim_out = hddm_dataset_generators._add_outliers(
                    sim_out=sim_out,
                    p_outlier=self.parents.value["p_outlier"],
                    max_rt_outlier=1 / wiener_params["w_outlier"],
                )

        sim_out_proc = hddm_preprocess(
            sim_out,
            keep_negative_responses=keep_negative_responses,
            add_model_parameters=add_model_parameters,
            keep_subj_idx=keep_subj_idx,
        )

        if add_model:
            sim_out_proc["model"] = model

        return sim_out_proc

    def pdf(self, x):
        return "Not yet implemented"

    def cdf(self, x):
        # TODO: Implement the CDF method for neural networks
        return "Not yet implemented"

    def make_likelihood():
        if indirect_betas_present or indirect_regressors_present:
            likelihood_str = make_reg_likelihood_str_mlp(
                config=model_config,
                wiener_params=wiener_params,
                param_links=param_links,
                param_links_betas=param_links_betas,
            )
        else:
            likelihood_str = make_reg_likelihood_str_mlp_basic(
                config=model_config,
                wiener_params=wiener_params,
            )

        exec(likelihood_str)
        my_fun = locals()["custom_likelihood_reg"]
        return my_fun

    # TODO: Allow for missing data in LAN likelihoods
    def make_likelihood_missing_data():
        return

    param_links, indirect_regressors_present = __prepare_indirect_regressors(
        model_config=model_config
    )
    param_links_betas, indirect_betas_present = __prepare_indirect_betas(
        model_config=model_config
    )

    likelihood_ = make_likelihood()
    stoch = stochastic_from_dist("wfpt_reg", partial(likelihood_, **kwargs))
    stoch.pdf = pdf
    stoch.cdf = cdf
    stoch.random = random
    return stoch


def make_mlp_likelihood_reg_nn_rl(
    model=None, model_config=None, model_config_rl=None, wiener_params=None, **kwargs
):
    """Defines the regressor likelihoods for the LAN-based RLSSM class.

    :Arguments:
        model: str <default='ddm'>
            String that determines which model you would like to fit your data to.
            Currently available models are: 'ddm', 'full_ddm', 'angle', 'weibull', 'ornstein', 'levy'
        model_config: dict <default=None>
            Config dictionary for the sequential sampling model, necessary for construction of likelihood. In the style of what you find under hddm.model_config.
        model_config_rl: dict <default=None>
            Config dictionary for the reinforcement learning model, necessary for construction of likelihood. In the style of what you find under hddm.model_config_rl.
        kwargs: dict
            Dictionary of additional keyword arguments.
            Importantly here, this carries the preloaded CNN.

    :Returns:
        Returns a pymc.object stochastic object as defined by PyMC2
    """

    def make_likelihood():
        if indirect_betas_present or indirect_regressors_present:
            raise NotImplementedError(
                "Indirect regressors are not yet implemented for RLSSM models."
            )
        else:
            likelihood_str = make_reg_likelihood_str_mlp_basic_nn_rl(
                model=model,
                config=model_config,
                config_rl=model_config_rl,
                wiener_params=wiener_params,
            )

        exec(likelihood_str)
        my_fun = locals()["custom_likelihood_reg"]
        return my_fun

    param_links, indirect_regressors_present = __prepare_indirect_regressors(
        model_config=model_config
    )
    param_links_betas, indirect_betas_present = __prepare_indirect_betas(
        model_config=model_config
    )

    likelihood_ = make_likelihood()
    stoch_nn_rl = stochastic_from_dist("wfpt_reg_nn_rl", partial(likelihood_, **kwargs))
    return stoch_nn_rl
