import numpy as np
from functools import partial
from kabuki.utils import stochastic_from_dist
from copy import deepcopy

import hddm
from hddm.simulators import *

# from hddm.model_config import model_config
from hddm.utils import *


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

    likelihood_ = make_likelihood()

    wfpt_nn = stochastic_from_dist("Wienernn_" + model, partial(likelihood_, **kwargs))

    wfpt_nn.pdf = pdf
    wfpt_nn.cdf_vec = None  # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
    wfpt_nn.cdf = cdf
    wfpt_nn.random = random
    return wfpt_nn

def make_mlp_likelihood_rlssm(model=None, model_config=None, wiener_params=None, **kwargs):
    print(" in make_mlp: ", model)
    def make_likelihood():
        likelihood_str = make_likelihood_str_mlp_rlssm(
            model=model, config=model_config, wiener_params=wiener_params
        )
        exec(likelihood_str)
        my_fun = locals()["custom_likelihood"]
        return my_fun

    likelihood_ = make_likelihood()

    wfpt_nn_rl = stochastic_from_dist("WienernnRL_" + model, partial(likelihood_, **kwargs))

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
        for tmp_str in model_config["params"]:  # ['v', 'a', 'z', 't']:
            if tmp_str in self.parents["reg_outcomes"]:
                param_data[:, cnt] = param_dict[tmp_str].values
                # previously iloc[self.value.index]
                # changed from iloc[self.value.index][0]
            else:
                param_data[:, cnt] = param_dict[tmp_str]
            cnt += 1

        sim_out = simulator(
            theta=param_data, model=model, n_samples=1, max_t=20  # n_trials = size,
        )

        # Add outliers:
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

    # if model == 'custom':
    def make_likelihood():
        likelihood_str = make_reg_likelihood_str_mlp(
            config=model_config, wiener_params=wiener_params
        )
        exec(likelihood_str)
        my_fun = locals()["custom_likelihood_reg"]
        return my_fun

    likelihood_ = make_likelihood()
    stoch = stochastic_from_dist("wfpt_reg", partial(likelihood_, **kwargs))
    stoch.pdf = pdf
    stoch.cdf = cdf
    stoch.random = random
    return stoch


# KEEP BELOW AS AN EXAMPLE FOR HOW MLP LIKELIHOODS WERE DEFINED
# Test likelihood function
# Potentially useful later
# if model == "test":
#     def wienernn_like_test(p_outlier = 0, w_outlier = 0.1, network = None, **kws):
#         """
#         LAN Log-likelihood for the DDM
#         """
#         print(kws)
#         print(locals())
#         return hddm.wfpt.wiener_like_nn_mlp(
#             kws['x']["rt"].values,
#             kws['x']["response"].values,
#             np.array([kws[param] for param in model_config[model]["params"]], dtype = np.float32),
#             p_outlier=p_outlier,
#             w_outlier=w_outlier,
#             network=network,
#         )  # **kwargs)

#     def pdf_test(self, x):
#         rt = np.array(x, dtype=np.float32)
#         response = rt / np.abs(rt)
#         rt = np.abs(rt)
#         params = np.array(
#             [self.parents[param] for param in model_config[model]["params"]]
#         ).astype(np.float32)
#         out = hddm.wfpt.wiener_like_nn_mlp_pdf(
#             rt, response, params, network=kwargs["network"], **self.parents
#         )  # **kwargs) # This may still be buggy !
#         return out

#     def cdf_test(self, x):
#         # TODO: Implement the CDF method for neural networks
#         return "Not yet implemented"

#     #Create wfpt class
#     wfpt_nn = stochastic_from_dist(
#         "Wienernn_" + model, partial(wienernn_like_test, network = kwargs["network"])
#     )

#     # wfpt_nn = stochastic_from_dist(
#     #     "Wienernn_" + model, partial(kwargs["likelihood_fun"], network = kwargs["network"])
#     # )

#     # wfpt_nn = stochastic_from_dist(
#     #     "Wienernn_" + model, partial(wienernn_like_test, network = kwargs["network"])
#     # )

#     wfpt_nn.pdf = pdf_test
#     wfpt_nn.cdf_vec = None  # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
#     wfpt_nn.cdf = cdf_test
#     wfpt_nn.random = random
#     return wfpt_nn


# Keeping below function as example for how regressor likelihoods were defined
# for how likelihood functions were defined previously
# now --> util.py --> make_reg_likelihood_str_mlp

# def wiener_multi_like_nn_test(
#     value, v, a, z, t, reg_outcomes, p_outlier=0, w_outlier=0.1, **kwargs
# ):
#     params = locals()
#     size = int(value.shape[0])
#     data = np.zeros((size, data_frame_width), dtype=np.float32)
#     data[:, n_params:] = np.stack(
#         [
#             np.absolute(value["rt"]).astype(np.float32),
#             value["response"].astype(np.float32),
#         ],
#         axis=1,
#     )

#     cnt = 0
#     for tmp_str in model_parameter_names:  # model_config[model]['params']:
#         if tmp_str in reg_outcomes:
#             data[:, cnt] = params[tmp_str].loc[value["rt"].index].values
#             if (data[:, cnt].min() < model_parameter_lower_bounds[cnt]) or (
#                 data[:, cnt].max() > model_parameter_upper_bounds[cnt]
#             ):
#                 print("boundary violation of regressor part")
#                 return -np.inf
#         else:
#             data[:, cnt] = params[tmp_str]
#         cnt += 1

#     # Has optimization potential --> AF-TODO: For next version!
#     return hddm.wfpt.wiener_like_multi_nn_mlp(
#         data, p_outlier=p_outlier, w_outlier=w_outlier, network=kwargs["network"]
#     )  # **kwargs
