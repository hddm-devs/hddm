import numpy as np
import hddm
from functools import partial
from kabuki.utils import stochastic_from_dist
from hddm.simulators import *

# import data_simulators
from copy import deepcopy

# Defining the likelihood functions
def make_cnn_likelihood(model, pdf_multiplier=1, **kwargs):
    """Defines the likelihoods for the CNN networks.

    :Arguments:
        model: str <default='ddm>
            String that determines which model you would like to fit your data to.
            Currently available models are: 'ddm', 'full_ddm', 'angle', 'weibull', 'ornstein', 'levy'
        pdf_multiplier: int <default=1>
            Currently not used. Is meant to adjust for the bin size on which CNN RT histograms were based, to get
            the right proportionality constant.
        **kwargs: dict
            Dictionary of additional keyword arguments.
            Importantly here, this carries the preloaded CNN.
    :Returns:
        pymc.object: Returns a stochastic object as defined by PyMC2
    """

    def random(self):

        # this can be simplified so that we pass parameters directly to the simulator ...
        theta = np.array(model_config[model]["default_params"], dtype=np.float32)
        keys_tmp = self.parents.value.keys()
        cnt = 0

        for param in model_config[model]["params"]:
            if param in keys_tmp:
                theta[cnt] = np.array(self.parents.value[param]).astype(np.float32)
            cnt += 1

        sim_out = simulator(
            theta=theta, model=model, n_samples=self.shape[0], max_t=20.0
        )
        return hddm_preprocess(sim_out)

    def pdf(self, x):
        rt = np.array(x, dtype=np.int_)
        response = rt.copy()
        response[rt < 0] = 0
        response[rt > 0] = 1
        response = response.astype(np.int_)
        rt = np.abs(rt)

        # this can be simplified so that we pass parameters directly to the simulator ...
        theta = np.array(model_config[model]["default_params"], dtype=np.float32)
        keys_tmp = self.parents.value.keys()
        cnt = 0

        for param in model_config[model]["params"]:
            if param in keys_tmp:
                theta[cnt] = np.array(self.parents.value[param]).astype(np.float32)
            cnt += 1

        out = hddm.wfpt.wiener_pdf_cnn_2(
            x=rt, response=response, network=kwargs["network"], parameters=theta
        )  # **kwargs) # This may still be buggy !
        return out

    def cdf(self, x):
        # TODO: Implement the CDF method for neural networks
        return "Not yet implemented"

    def wienernn_like_ddm(x, v, a, z, t, p_outlier=0, w_outlier=0, **kwargs):  # theta

        return hddm.wfpt.wiener_like_cnn_2(
            x["rt_binned"].values,
            x["response_binned"].values,
            np.array([v, a, z, t], dtype=np.float32),
            p_outlier=p_outlier,  # TODO: ACTUALLY USE THIS
            w_outlier=w_outlier,
            **kwargs
        )

    def wienernn_like_weibull(
        x, v, a, alpha, beta, z, t, p_outlier=0, w_outlier=0, **kwargs
    ):  # theta

        return hddm.wfpt.wiener_like_cnn_2(
            x["rt_binned"].values,
            x["response_binned"].values,
            np.array([v, a, z, t, alpha, beta], dtype=np.float32),
            p_outlier=p_outlier,  # TODO: ACTUALLY USE THIS
            w_outlier=w_outlier,
            **kwargs
        )

    def wienernn_like_levy(
        x, v, a, alpha, z, t, p_outlier=0, w_outlier=0, **kwargs
    ):  # theta

        return hddm.wfpt.wiener_like_cnn_2(
            x["rt_binned"].values,
            x["response_binned"].values,
            np.array([v, a, z, alpha, t], dtype=np.float32),
            p_outlier=p_outlier,  # TODO: ACTUALLY USE THIS
            w_outlier=w_outlier,
            **kwargs
        )

    def wienernn_like_ornstein(
        x, v, a, g, z, t, p_outlier=0, w_outlier=0, **kwargs
    ):  # theta

        return hddm.wfpt.wiener_like_cnn_2(
            x["rt_binned"].values,
            x["response_binned"].values,
            np.array([v, a, z, g, t], dtype=np.float32),
            p_outlier=p_outlier,  # TODO: ACTUALLY USE THIS
            w_outlier=w_outlier,
            **kwargs
        )

    def wienernn_like_full_ddm(
        x, v, sv, a, z, sz, t, st, p_outlier=0, w_outlier=0, **kwargs
    ):

        return hddm.wfpt.wiener_like_cnn_2(
            x["rt_binned"].values,
            x["response_binned"].values,
            np.array([v, a, z, t, sz, sv, st], dtype=np.float32),
            p_outlier=p_outlier,
            w_outlier=w_outlier,
            **kwargs
        )

    def wienernn_like_angle(x, v, a, theta, z, t, p_outlier=0, w_outlier=0, **kwargs):

        return hddm.wfpt.wiener_like_cnn_2(
            x["rt_binned"].values,
            x["response_binned"].values,
            np.array([v, a, z, t, theta], dtype=np.float32),
            p_outlier=p_outlier,
            w_outlier=w_outlier,
            **kwargs
        )

    likelihood_funs = {}
    likelihood_funs["ddm"] = wienernn_like_ddm
    likelihood_funs["weibull"] = wienernn_like_weibull
    likelihood_funs["angle"] = wienernn_like_angle
    likelihood_funs["levy"] = wienernn_like_levy
    likelihood_funs["ornstein"] = wienernn_like_ornstein
    likelihood_funs["full_ddm"] = wienernn_like_full_ddm

    # Create wfpt class
    wfpt_nn = stochastic_from_dist(
        "Wienernn_" + model, partial(likelihood_funs[model], **kwargs)
    )

    wfpt_nn.pdf = pdf
    wfpt_nn.cdf_vec = None  # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
    wfpt_nn.cdf = cdf
    wfpt_nn.random = random
    return wfpt_nn


def generate_wfpt_nn_ddm_reg_stochastic_class(model=None, **kwargs):
    """Defines the regressor likelihoods for the CNN networks.

    :Arguments:
        model: str <default='ddm>
            String that determines which model you would like to fit your data to.
            Currently available models are: 'ddm', 'full_ddm', 'angle', 'weibull', 'ornstein', 'levy'
        **kwargs: dict
            Dictionary of additional keyword arguments.
            Importantly here, this carries the preloaded CNN.
    :Returns:
        pymc.object: Returns a stochastic object as defined by PyMC2
    """

    # Need to rewrite these random parts !
    def random(self):
        param_dict = deepcopy(self.parents.value)
        del param_dict["reg_outcomes"]
        sampled_rts = self.value.copy()

        size = sampled_rts.shape[0]
        n_params = model_config[model]["n_params"]
        param_data = np.zeros((size, n_params), dtype=np.float32)

        cnt = 0
        for tmp_str in model_config[model]["params"]:  # ['v', 'a', 'z', 't']:
            if tmp_str in self.parents["reg_outcomes"]:
                param_data[:, cnt] = param_dict[tmp_str].values[:, 0]
            else:
                param_data[:, cnt] = param_dict[tmp_str]
            cnt += 1

        sim_out = simulator(theta=param_data, n_samples=1, model=model, max_t=20)
        return hddm_preprocess(sim_out, keep_negative_responses=True)

    def pdf(self, x):
        return "Not yet implemented"

    def cdf(self, x):
        # TODO: Implement the CDF method for neural networks
        return "Not yet implemented"

    def wiener_multi_like_nn_ddm(
        value, v, a, z, t, reg_outcomes, p_outlier=0, w_outlier=0, **kwargs
    ):  # theta

        params = {"v": v, "a": a, "z": z, "t": t}
        n_params = 4
        size = int(value.shape[0])
        data = np.zeros((size, n_params), dtype=np.float32)

        cnt = 0
        for tmp_str in ["v", "a", "z", "t"]:

            if tmp_str in reg_outcomes:
                data[:, cnt] = (
                    params[tmp_str].loc[value["rt_binned"].index].values[:, 0]
                )
                if (
                    data[:, cnt].min() < model_config[model]["param_bounds"][0][cnt]
                ) or (data[:, cnt].max() > model_config[model]["param_bounds"][1][cnt]):
                    print("boundary violation of regressor part")
                    return -np.inf
            else:
                data[:, cnt] = params[tmp_str]

            cnt += 1

        return hddm.wfpt.wiener_like_reg_cnn_2(
            value["rt_binned"],
            value["response_binned"],
            data,
            p_outlier=p_outlier,  # TODO: ACTUALLY USE THIS
            w_outlier=w_outlier,
            **kwargs
        )

    def wiener_multi_like_nn_full_ddm(
        value,
        v,
        sv,
        a,
        z,
        sz,
        t,
        st,
        reg_outcomes,
        p_outlier=0,
        w_outlier=0.1,
        **kwargs
    ):

        params = {"v": v, "a": a, "z": z, "t": t, "sz": sz, "sv": sv, "st": st}

        n_params = int(7)
        size = int(value.shape[0])
        data = np.zeros((size, n_params), dtype=np.float32)

        cnt = 0
        for tmp_str in ["v", "a", "z", "t", "sz", "sv", "st"]:

            if tmp_str in reg_outcomes:
                data[:, cnt] = (
                    params[tmp_str].loc[value["rt_binned"].index].values[:, 0]
                )
                if (
                    data[:, cnt].min() < model_config[model]["param_bounds"][0][cnt]
                ) or (data[:, cnt].max() > model_config[model]["param_bounds"][1][cnt]):
                    print("boundary violation of regressor part")
                    return -np.inf
            else:
                data[:, cnt] = params[tmp_str]

            cnt += 1

        # Has optimization potential --> AF-TODO: For next version!
        return hddm.wfpt.wiener_like_reg_cnn_2(
            value["rt_binned"],
            value["response_binned"],
            data,
            p_outlier=p_outlier,  # TODO: ACTUALLY USE THIS
            w_outlier=w_outlier,
            **kwargs
        )

    def wiener_multi_like_nn_angle(
        value, v, a, theta, z, t, reg_outcomes, p_outlier=0, w_outlier=0.1, **kwargs
    ):

        """Log-likelihood for the full DDM using the interpolation method"""

        params = {"v": v, "a": a, "z": z, "t": t, "theta": theta}
        n_params = int(5)
        size = int(value.shape[0])
        data = np.zeros((size, n_params), dtype=np.float32)

        cnt = 0
        for tmp_str in ["v", "a", "z", "t", "theta"]:

            if tmp_str in reg_outcomes:
                data[:, cnt] = (
                    params[tmp_str].loc[value["rt_binned"].index].values[:, 0]
                )
                if (
                    data[:, cnt].min() < model_config[model]["param_bounds"][0][cnt]
                ) or (data[:, cnt].max() > model_config[model]["param_bounds"][1][cnt]):
                    print("boundary violation of regressor part")
                    return -np.inf
            else:
                data[:, cnt] = params[tmp_str]

            cnt += 1

        # Has optimization potential --> AF-TODO: For next version!
        return hddm.wfpt.wiener_like_reg_cnn_2(
            value["rt_binned"],
            value["response_binned"],
            data,
            p_outlier=p_outlier,  # TODO: ACTUALLY USE THIS
            w_outlier=w_outlier,
            **kwargs
        )

    def wiener_multi_like_nn_levy(
        value, v, a, alpha, z, t, reg_outcomes, p_outlier=0, w_outlier=0.1, **kwargs
    ):

        """Log-likelihood for the full DDM using the interpolation method"""
        params = {"v": v, "a": a, "z": z, "alpha": alpha, "t": t}
        n_params = int(5)
        size = int(value.shape[0])
        data = np.zeros((size, n_params), dtype=np.float32)

        cnt = 0
        for tmp_str in ["v", "a", "z", "alpha", "t"]:
            if tmp_str in reg_outcomes:
                data[:, cnt] = (
                    params[tmp_str].loc[value["rt_binned"].index].values[:, 0]
                )
                if (
                    data[:, cnt].min() < model_config[model]["param_bounds"][0][cnt]
                ) or (data[:, cnt].max() > model_config[model]["param_bounds"][1][cnt]):
                    print("boundary violation of regressor part")
                    return -np.inf
            else:
                data[:, cnt] = params[tmp_str]

            cnt += 1

        # Has optimization potential --> AF-TODO: For next version!
        return hddm.wfpt.wiener_like_reg_cnn_2(
            value["rt_binned"],
            value["response_binned"],
            data,
            p_outlier=p_outlier,  # TODO: ACTUALLY USE THIS
            w_outlier=w_outlier,
            **kwargs
        )

    def wiener_multi_like_nn_ornstein(
        value, v, a, g, z, t, reg_outcomes, p_outlier=0, w_outlier=0.1, **kwargs
    ):

        params = {"v": v, "a": a, "z": z, "g": g, "t": t}
        n_params = int(5)
        size = int(value.shape[0])
        data = np.zeros((size, n_params), dtype=np.float32)

        cnt = 0
        for tmp_str in ["v", "a", "z", "g", "t"]:

            if tmp_str in reg_outcomes:
                data[:, cnt] = (
                    params[tmp_str].loc[value["rt_binned"].index].values[:, 0]
                )
                if (
                    data[:, cnt].min() < model_config[model]["param_bounds"][0][cnt]
                ) or (data[:, cnt].max() > model_config[model]["param_bounds"][1][cnt]):
                    print("boundary violation of regressor part")
                    return -np.inf
            else:
                data[:, cnt] = params[tmp_str]

            cnt += 1

        # Has optimization potential --> AF-TODO: For next version!
        return hddm.wfpt.wiener_like_reg_cnn_2(
            value["rt_binned"],
            value["response_binned"],
            data,
            p_outlier=p_outlier,  # TODO: ACTUALLY USE THIS
            w_outlier=w_outlier,
            **kwargs
        )

    def wiener_multi_like_nn_weibull(
        value,
        v,
        a,
        alpha,
        beta,
        z,
        t,
        reg_outcomes,
        p_outlier=0,
        w_outlier=0.1,
        **kwargs
    ):

        params = {"v": v, "a": a, "z": z, "t": t, "alpha": alpha, "beta": beta}
        n_params = int(6)
        size = int(value.shape[0])
        data = np.zeros((size, n_params), dtype=np.float32)

        cnt = 0
        for tmp_str in ["v", "a", "z", "t", "alpha", "beta"]:

            if tmp_str in reg_outcomes:
                data[:, cnt] = (
                    params[tmp_str].loc[value["rt_binned"].index].values[:, 0]
                )
                if (
                    data[:, cnt].min() < model_config[model]["param_bounds"][0][cnt]
                ) or (data[:, cnt].max() > model_config[model]["param_bounds"][1][cnt]):
                    print("boundary violation of regressor part")
                    return -np.inf
            else:
                data[:, cnt] = params[tmp_str]

            cnt += 1

        # Has optimization potential --> AF-TODO: For next version!
        return hddm.wfpt.wiener_like_reg_cnn_2(
            value["rt_binned"],
            value["response_binned"],
            data,
            p_outlier=p_outlier,  # TODO: ACTUALLY USE THIS
            w_outlier=w_outlier,
            **kwargs
        )

    likelihood_funs = {}
    likelihood_funs["ddm"] = wiener_multi_like_nn_ddm
    likelihood_funs["full_ddm"] = wiener_multi_like_nn_full_ddm
    likelihood_funs["angle"] = wiener_multi_like_nn_angle
    likelihood_funs["levy"] = wiener_multi_like_nn_levy
    likelihood_funs["ornstein"] = wiener_multi_like_nn_ornstein
    likelihood_funs["weibull"] = wiener_multi_like_nn_weibull

    stoch = stochastic_from_dist("wfpt_reg", partial(likelihood_funs[model], **kwargs))
    stoch.pdf = pdf
    stoch.cdf = cdf
    stoch.random = random

    return stoch
