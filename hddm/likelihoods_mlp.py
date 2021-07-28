import numpy as np
import hddm
from functools import partial
from kabuki.utils import stochastic_from_dist
from hddm.simulators import *
from hddm.model_config import model_config


# import data_simulators
from copy import deepcopy


def make_mlp_likelihood(model, **kwargs):
    """Defines the likelihoods for the MLP networks.

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

    def random(self):
        """
        Generate random samples from a given model (the dataset matches the size of the respective observated dataset supplied as an attribute of 'self').
        """

        # this can be simplified so that we pass parameters directly to the simulator ...
        theta = np.array(model_config[model]["default_params"], dtype=np.float32)
        keys_tmp = self.parents.value.keys()
        cnt = 0

        for param in model_config[model]["params"]:
            if param in keys_tmp:
                theta[cnt] = np.array(self.parents.value[param]).astype(np.float32)
            cnt += 1

        sim_out = simulator(theta=theta, model=model, n_samples=self.shape[0], max_t=20)
        return hddm_preprocess(sim_out, keep_negative_responses=True)

    def pdf(self, x):
        rt = np.array(x, dtype=np.float32)
        response = rt / np.abs(rt)
        rt = np.abs(rt)
        params = np.array(
            [self.parents[param] for param in model_config[model]["params"]]
        ).astype(np.float32)
        return hddm.wfpt.wiener_like_nn_mlp_pdf(
            rt, response, params, network=kwargs["network"], **self.parents
        )  # **kwargs) # This may still be buggy !

    def cdf(self, x):
        # TODO: Implement the CDF method for neural networks
        return "Not yet implemented"

    # if model == "custom":
    #     wfpt_nn = stochastic_from_dist(
    #         "Wienernn_" + model, partial(kwargs["likelihood_fun"], network = kwargs["network"])
    #         )

    #     wfpt_nn.pdf = pdf
    #     wfpt_nn.cdf_vec = None  # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
    #     wfpt_nn.cdf = cdf
    #     wfpt_nn.random = random
    #     return wfpt_nn

    if model == "test":
        #print('passing through exec statement')
        #likelihood_dict = {}
        #exec('def wienernn_like_test(x, v, a, z, t, p_outlier=0, w_outlier=0.1, network = None):\n    return hddm.wfpt.wiener_like_nn_mlp(x["rt"], x["response"], np.array([v, a, z, t], dtype=np.float32), p_outlier=p_outlier, w_outlier=w_outlier, network=network)', globals(), locals())
        #print(likelihood_dict)
        #custom_likelihood = likelihood_dict['wienernn_like_test']
        #exec('def myfun():\n    print("I am the new myfun output")\n    return', globals(), locals())
        #def wienernn_like_test(x, v, a, z, t, p_outlier=0, w_outlier=0.1, **kwargs):
        #def wienernn_like_test(x, v, a, z, t, p_outlier = 0, w_outlier = 0.1, **kwargs):
        def wienernn_like_test(p_outlier = 0, w_outlier = 0.1, network = None, **kws):
            """
            LAN Log-likelihood for the DDM
            """
            print(kws)
            print(locals())
            return hddm.wfpt.wiener_like_nn_mlp(
                kws['x']["rt"].values,
                kws['x']["response"].values,
                np.array([kws[param] for param in model_config[model]["params"]], dtype = np.float32),
                p_outlier=p_outlier,
                w_outlier=w_outlier,
                network=network,
            )  # **kwargs)
        #print(myfun)
        #myfun()
        #print(wienernn_like_test)
        
        def pdf_test(self, x):
            rt = np.array(x, dtype=np.float32)
            response = rt / np.abs(rt)
            rt = np.abs(rt)
            params = np.array(
                [self.parents[param] for param in model_config[model]["params"]]
            ).astype(np.float32)
            out = hddm.wfpt.wiener_like_nn_mlp_pdf(
                rt, response, params, network=kwargs["network"], **self.parents
            )  # **kwargs) # This may still be buggy !
            return out

        def cdf_test(self, x):
            # TODO: Implement the CDF method for neural networks
            return "Not yet implemented"

        #Create wfpt class
        wfpt_nn = stochastic_from_dist(
            "Wienernn_" + model, partial(wienernn_like_test, network = kwargs["network"])
        )

        # wfpt_nn = stochastic_from_dist(
        #     "Wienernn_" + model, partial(kwargs["likelihood_fun"], network = kwargs["network"])
        # )

        # wfpt_nn = stochastic_from_dist(
        #     "Wienernn_" + model, partial(wienernn_like_test, network = kwargs["network"])
        # )

        wfpt_nn.pdf = pdf_test
        wfpt_nn.cdf_vec = None  # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
        wfpt_nn.cdf = cdf_test
        wfpt_nn.random = random
        return wfpt_nn

    def wienernn_like_ddm(x, v, a, z, t, p_outlier=0, w_outlier=0.1, **kwargs):
        """
        LAN Log-likelihood for the DDM
        """
        return hddm.wfpt.wiener_like_nn_mlp(
            x["rt"].values,
            x["response"].values,
            np.array([v, a, z, t]).astype(np.float32),
            p_outlier=p_outlier,
            w_outlier=w_outlier,
            network=kwargs["network"],
        )  # **kwargs)

    def wienernn_like_weibull(
        x, v, a, alpha, beta, z, t, p_outlier=0, w_outlier=0.1, **kwargs
    ):
        """
        LAN Log-likelihood for the DDM
        """
        return hddm.wfpt.wiener_like_nn_mlp(
            x["rt"].values,
            x["response"].values,
            np.array([v, a, z, t, alpha, beta]).astype(np.float32),
            p_outlier=p_outlier,
            w_outlier=w_outlier,
            network=kwargs["network"],
        )  # **kwargs)

    def wienernn_like_ddm_sdv(x, v, sv, a, z, t, p_outlier=0, w_outlier=0.1, **kwargs):
        """
        LAN Log-likelihood for the DDM
        """
        return hddm.wfpt.wiener_like_nn_mlp(
            x["rt"].values,
            x["response"].values,
            np.array([v, a, z, t, sv]).astype(np.float32),
            p_outlier=p_outlier,
            w_outlier=w_outlier,
            network=kwargs["network"],
        )  # **kwargs)

    def wienernn_like_levy(x, v, a, alpha, z, t, p_outlier=0, w_outlier=0.1, **kwargs):
        """
        LAN Log-likelihood for the DDM
        """
        return hddm.wfpt.wiener_like_nn_mlp(
            x["rt"].values,
            x["response"].values,
            np.array([v, a, z, alpha, t]).astype(np.float32),
            p_outlier=p_outlier,
            w_outlier=w_outlier,
            network=kwargs["network"],
        )  # **kwargs)

    def wienernn_like_ornstein(x, v, a, g, z, t, p_outlier=0, w_outlier=0.1, **kwargs):
        """
        LAN Log-likelihood for the DDM
        """
        return hddm.wfpt.wiener_like_nn_mlp(
            x["rt"].values,
            x["response"].values,
            np.array([v, a, z, g, t]).astype(np.float32),
            p_outlier=p_outlier,
            w_outlier=w_outlier,
            network=kwargs["network"],
        )  # **kwargs)

    def wienernn_like_full_ddm(
        x, v, sv, a, z, sz, t, st, p_outlier=0, w_outlier=0.1, **kwargs
    ):
        """
        LAN Log-likelihood for the DDM
        """
        return hddm.wfpt.wiener_like_nn_mlp(
            x["rt"].values,
            x["response"].values,
            np.array([v, a, z, t, sz, sv, st]).astype(np.float32),
            p_outlier=p_outlier,
            w_outlier=w_outlier,
            network=kwargs["network"],
        )  # **kwargs)

    def wienernn_like_angle(x, v, a, theta, z, t, p_outlier=0, w_outlier=0.1, **kwargs):
        """
        LAN Log-likelihood for the DDM
        """
        return hddm.wfpt.wiener_like_nn_mlp(
            x["rt"].values,
            x["response"].values,
            np.array([v, a, z, t, theta]).astype(np.float32),
            p_outlier=p_outlier,
            w_outlier=w_outlier,
            network=kwargs["network"],
        )  # **kwargs)

    def wienernn_like_par2(
        x, vh, vl1, vl2, a, zh, zl1, zl2, t, p_outlier=0.0, w_outlier=0.0, **kwargs
    ):
        """
        LAN Log-likelihood for the DDM
        """
        return hddm.wfpt.wiener_like_nn_mlp(
            x["rt"].values,
            x["response"].values,
            np.array([vh, vl1, vl2, a, zh, zl1, zl2, t]).astype(np.float32),
            p_outlier=p_outlier,
            w_outlier=w_outlier,
            network=kwargs["network"],
        )  # **kwargs)

    def wienernn_like_seq2(
        x, vh, vl1, vl2, a, zh, zl1, zl2, t, p_outlier=0.0, w_outlier=0.0, **kwargs
    ):
        """
        LAN Log-likelihood for the DDM
        """
        return hddm.wfpt.wiener_like_nn_mlp(
            x["rt"].values,
            x["response"].values,
            np.array([vh, vl1, vl2, a, zh, zl1, zl2, t]).astype(np.float32),
            p_outlier=p_outlier,
            w_outlier=w_outlier,
            network=kwargs["network"],
        )  # **kwargs)

    def wienernn_like_mic2(
        x, vh, vl1, vl2, a, zh, zl1, zl2, d, t, p_outlier=0.0, w_outlier=0.0, **kwargs
    ):
        """
        LAN Log-likelihood for the DDM
        """
        return hddm.wfpt.wiener_like_nn_mlp(
            x["rt"].values,
            x["response"].values,
            np.array([vh, vl1, vl2, a, zh, zl1, zl2, d, t]).astype(np.float32),
            p_outlier=p_outlier,
            w_outlier=w_outlier,
            network=kwargs["network"],
        )  # **kwargs)

    likelihood_funs = {}
    likelihood_funs["ddm"] = wienernn_like_ddm
    likelihood_funs["weibull"] = wienernn_like_weibull
    likelihood_funs["angle"] = wienernn_like_angle
    likelihood_funs["ddm_sdv"] = wienernn_like_ddm_sdv
    likelihood_funs["ddm_sdv_analytic"] = wienernn_like_ddm_sdv
    likelihood_funs["levy"] = wienernn_like_levy
    likelihood_funs["ornstein"] = wienernn_like_ornstein
    likelihood_funs["full_ddm"] = wienernn_like_full_ddm
    likelihood_funs["ddm_par2"] = wienernn_like_par2
    likelihood_funs["ddm_seq2"] = wienernn_like_seq2
    likelihood_funs["ddm_mic2"] = wienernn_like_mic2
    likelihood_funs["custom"] = kwargs["likelihood_fun"]

    wfpt_nn = stochastic_from_dist(
        "Wienernn_" + model, partial(likelihood_funs[model], **kwargs)
    )
    wfpt_nn.pdf = pdf
    wfpt_nn.cdf_vec = None  # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
    wfpt_nn.cdf = cdf
    wfpt_nn.random = random
    return wfpt_nn

# REGRESSOR LIKELIHOODS
def make_mlp_likelihood_reg(model=None, **kwargs):
    """Defines the regressor likelihoods for the MLP networks.

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

    # Model specific inits ------------------------------------------
    n_params = model_config[model]["n_params"]
    data_frame_width = n_params + 2
    model_parameter_names = model_config[model]["params"]
    model_parameter_lower_bounds = model_config[model]["param_bounds"][0]
    model_parameter_upper_bounds = model_config[model]["param_bounds"][1]
    # ---------------------------------------------------------------

    # Need to rewrite these random parts !
    def random(
        self,
        keep_negative_responses=True,
        add_model_parameters=False,
        keep_subj_idx=False,
    ):
        """
        Function to sample from a regressor based likelihood. Conditions on the covariates.
        """
        param_dict = deepcopy(self.parents.value)
        del param_dict["reg_outcomes"]

        # size = sampled_rts.shape[0]
        n_params = model_config[model]["n_params"]
        param_data = np.zeros((self.value.shape[0], n_params), dtype=np.float32)

        cnt = 0
        for tmp_str in model_config[model]["params"]:  # ['v', 'a', 'z', 't']:
            if tmp_str in self.parents["reg_outcomes"]:
                param_data[:, cnt] = param_dict[tmp_str].iloc[self.value.index, 0]
            else:
                param_data[:, cnt] = param_dict[tmp_str]
            cnt += 1

        sim_out = simulator(
            theta=param_data, model=model, n_samples=1, max_t=20  # n_trials = size,
        )

        return hddm_preprocess(
            sim_out,
            keep_negative_responses=keep_negative_responses,
            add_model_parameters=add_model_parameters,
            keep_subj_idx=keep_subj_idx,
        )

    def pdf(self, x):
        return "Not yet implemented"

    def cdf(self, x):
        # TODO: Implement the CDF method for neural networks
        return "Not yet implemented"

    def wiener_multi_like_nn_test(
        value, v, a, z, t, reg_outcomes, p_outlier=0, w_outlier=0.1, **kwargs
    ):
        params = locals()
        size = int(value.shape[0])
        data = np.zeros((size, data_frame_width), dtype=np.float32)
        data[:, n_params:] = np.stack(
            [
                np.absolute(value["rt"]).astype(np.float32),
                value["response"].astype(np.float32),
            ],
            axis=1,
        )

        cnt = 0
        for tmp_str in model_parameter_names:  # model_config[model]['params']:
            if tmp_str in reg_outcomes:
                data[:, cnt] = params[tmp_str].loc[value["rt"].index].values
                if (data[:, cnt].min() < model_parameter_lower_bounds[cnt]) or (
                    data[:, cnt].max() > model_parameter_upper_bounds[cnt]
                ):
                    print("boundary violation of regressor part")
                    return -np.inf
            else:
                data[:, cnt] = params[tmp_str]
            cnt += 1

        # Has optimization potential --> AF-TODO: For next version!
        return hddm.wfpt.wiener_like_multi_nn_mlp(
            data, p_outlier=p_outlier, w_outlier=w_outlier, network=kwargs["network"]
        )  # **kwargs

    def wiener_multi_like_nn_ddm(
        value, v, a, z, t, reg_outcomes, p_outlier=0, w_outlier=0.1, **kwargs
    ):

        """LAN Log-likelihood for the DDM"""
        params = locals()
        size = int(value.shape[0])
        data = np.zeros((size, data_frame_width), dtype=np.float32)
        data[:, n_params:] = np.stack(
            [
                np.absolute(value["rt"]).astype(np.float32),
                value["response"].astype(np.float32),
            ],
            axis=1,
        )

        cnt = 0
        for tmp_str in model_parameter_names:  # model_config[model]['params']:
            if tmp_str in reg_outcomes:
                data[:, cnt] = params[tmp_str].loc[value["rt"].index].values
                if (data[:, cnt].min() < model_parameter_lower_bounds[cnt]) or (
                    data[:, cnt].max() > model_parameter_upper_bounds[cnt]
                ):
                    print("boundary violation of regressor part")
                    return -np.inf
            else:
                data[:, cnt] = params[tmp_str]
            cnt += 1

        # Has optimization potential --> AF-TODO: For next version!
        return hddm.wfpt.wiener_like_multi_nn_mlp(
            data, p_outlier=p_outlier, w_outlier=w_outlier, network=kwargs["network"]
        )  # **kwargs

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
        """
        LAN Log-likelihood for the FULL DDM
        """
        # params = {"v": v, "a": a, "z": z, "t": t, "sz": sz, "sv": sv, "st": st}

        params = locals()
        size = int(value.shape[0])
        data = np.zeros((size, data_frame_width), dtype=np.float32)
        data[:, n_params:] = np.stack(
            [
                np.absolute(value["rt"]).astype(np.float32),
                value["response"].astype(np.float32),
            ],
            axis=1,
        )

        cnt = 0
        for tmp_str in model_parameter_names:  # model_config[model]['params']:
            if tmp_str in reg_outcomes:
                data[:, cnt] = params[tmp_str].loc[value["rt"].index].values
                if (data[:, cnt].min() < model_parameter_lower_bounds[cnt]) or (
                    data[:, cnt].max() > model_parameter_upper_bounds[cnt]
                ):
                    print("boundary violation of regressor part")
                    return -np.inf
            else:
                data[:, cnt] = params[tmp_str]
            cnt += 1

        # Has optimization potential --> AF-TODO: For next version!
        return hddm.wfpt.wiener_like_multi_nn_mlp(
            data, p_outlier=p_outlier, w_outlier=w_outlier, network=kwargs["network"]
        )  # **kwargs

    def wiener_multi_like_nn_angle(
        value, v, a, theta, z, t, reg_outcomes, p_outlier=0, w_outlier=0.1, **kwargs
    ):

        """
        LAN Log-likelihood for the ANGLE MODEL
        """
        # params = {"v": v, "a": a, "z": z, "t": t, "theta": theta}
        params = locals()
        size = int(value.shape[0])
        data = np.zeros((size, data_frame_width), dtype=np.float32)
        data[:, n_params:] = np.stack(
            [
                np.absolute(value["rt"]).astype(np.float32),
                value["response"].astype(np.float32),
            ],
            axis=1,
        )

        cnt = 0
        for tmp_str in model_parameter_names:  # model_config[model]['params']:
            if tmp_str in reg_outcomes:
                data[:, cnt] = params[tmp_str].loc[value["rt"].index].values
                if (data[:, cnt].min() < model_parameter_lower_bounds[cnt]) or (
                    data[:, cnt].max() > model_parameter_upper_bounds[cnt]
                ):
                    print("boundary violation of regressor part")
                    return -np.inf
            else:
                data[:, cnt] = params[tmp_str]
            cnt += 1

        # Has optimization potential --> AF-TODO: For next version!
        return hddm.wfpt.wiener_like_multi_nn_mlp(
            data, p_outlier=p_outlier, w_outlier=w_outlier, network=kwargs["network"]
        )  # **kwargs

    def wiener_multi_like_nn_levy(
        value, v, a, alpha, z, t, reg_outcomes, p_outlier=0, w_outlier=0.1, **kwargs
    ):

        """
        LAN Log-likelihood for the LEVY MODEL
        """
        # params = {"v": v, "a": a, "z": z, "alpha": alpha, "t": t}
        params = locals()
        size = int(value.shape[0])
        data = np.zeros((size, data_frame_width), dtype=np.float32)
        data[:, n_params:] = np.stack(
            [
                np.absolute(value["rt"]).astype(np.float32),
                value["response"].astype(np.float32),
            ],
            axis=1,
        )

        cnt = 0
        for tmp_str in model_parameter_names:  # model_config[model]['params']:
            if tmp_str in reg_outcomes:
                data[:, cnt] = params[tmp_str].loc[value["rt"].index].values
                if (data[:, cnt].min() < model_parameter_lower_bounds[cnt]) or (
                    data[:, cnt].max() > model_parameter_upper_bounds[cnt]
                ):
                    print("boundary violation of regressor part")
                    return -np.inf
            else:
                data[:, cnt] = params[tmp_str]
            cnt += 1

        # Has optimization potential --> AF-TODO: For next version!
        return hddm.wfpt.wiener_like_multi_nn_mlp(
            data, p_outlier=p_outlier, w_outlier=w_outlier, network=kwargs["network"]
        )  # **kwargs

    def wiener_multi_like_nn_ornstein(
        value, v, a, g, z, t, reg_outcomes, p_outlier=0, w_outlier=0.1, **kwargs
    ):

        """
        LAN Log-likelihood for the ORNSTEIN MODEL
        """
        # params = {"v": v, "a": a, "z": z, "g": g, "t": t}
        params = locals()
        size = int(value.shape[0])
        data = np.zeros((size, data_frame_width), dtype=np.float32)
        data[:, n_params:] = np.stack(
            [
                np.absolute(value["rt"]).astype(np.float32),
                value["response"].astype(np.float32),
            ],
            axis=1,
        )

        cnt = 0
        for tmp_str in model_parameter_names:  # model_config[model]['params']:
            if tmp_str in reg_outcomes:
                data[:, cnt] = params[tmp_str].loc[value["rt"].index].values
                if (data[:, cnt].min() < model_parameter_lower_bounds[cnt]) or (
                    data[:, cnt].max() > model_parameter_upper_bounds[cnt]
                ):
                    print("boundary violation of regressor part")
                    return -np.inf
            else:
                data[:, cnt] = params[tmp_str]
            cnt += 1

        # Has optimization potential --> AF-TODO: For next version!
        return hddm.wfpt.wiener_like_multi_nn_mlp(
            data, p_outlier=p_outlier, w_outlier=w_outlier, network=kwargs["network"]
        )  # **kwargs

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

        """
        LAN Log-likelihood for the WEIBULL MODEL
        """
        # params = {"v": v, "a": a, "z": z, "t": t, "alpha": alpha, "beta": beta}
        params = locals()
        size = int(value.shape[0])
        data = np.zeros((size, data_frame_width), dtype=np.float32)
        data[:, n_params:] = np.stack(
            [
                np.absolute(value["rt"]).astype(np.float32),
                value["response"].astype(np.float32),
            ],
            axis=1,
        )

        cnt = 0
        for tmp_str in model_parameter_names:  # model_config[model]['params']:
            if tmp_str in reg_outcomes:
                data[:, cnt] = params[tmp_str].loc[value["rt"].index].values
                if (data[:, cnt].min() < model_parameter_lower_bounds[cnt]) or (
                    data[:, cnt].max() > model_parameter_upper_bounds[cnt]
                ):
                    print("boundary violation of regressor part")
                    return -np.inf
            else:
                data[:, cnt] = params[tmp_str]
            cnt += 1

        # Has optimization potential --> AF-TODO: For next version!
        return hddm.wfpt.wiener_like_multi_nn_mlp(
            data, p_outlier=p_outlier, w_outlier=w_outlier, network=kwargs["network"]
        )  # **kwargs

    def wiener_multi_like_nn_par2(
        value,
        vh,
        vl1,
        vl2,
        a,
        zh,
        zl1,
        zl2,
        t,
        reg_outcomes,
        p_outlier=0,
        w_outlier=0.1,
        **kwargs
    ):

        """
        LAN Log-likelihood for the WEIBULL MODEL
        """
        # params = {"v_h": v_h, "v_l_1": v_l_1, "v_l_2": v_l_2, "a": a,
        #          "z_h": z_h, "z_l_1": z_l_1, "z_l_2": z_l_2, "t": t}
        params = locals()
        size = int(value.shape[0])
        data = np.zeros((size, data_frame_width), dtype=np.float32)
        data[:, n_params:] = np.stack(
            [
                np.absolute(value["rt"]).astype(np.float32),
                value["response"].astype(np.float32),
            ],
            axis=1,
        )

        cnt = 0
        for tmp_str in model_parameter_names:  # model_config[model]['params']:
            if tmp_str in reg_outcomes:
                data[:, cnt] = params[tmp_str].loc[value["rt"].index].values
                if (data[:, cnt].min() < model_parameter_lower_bounds[cnt]) or (
                    data[:, cnt].max() > model_parameter_upper_bounds[cnt]
                ):
                    print("boundary violation of regressor part")
                    return -np.inf
            else:
                data[:, cnt] = params[tmp_str]
            cnt += 1

        # Has optimization potential --> AF-TODO: For next version!
        return hddm.wfpt.wiener_like_multi_nn_mlp(
            data, p_outlier=p_outlier, w_outlier=w_outlier, network=kwargs["network"]
        )  # **kwargs

    def wiener_multi_like_nn_seq2(
        value,
        vh,
        vl1,
        vl2,
        a,
        zh,
        zl1,
        zl2,
        t,
        reg_outcomes,
        p_outlier=0,
        w_outlier=0.1,
        **kwargs
    ):

        """
        LAN Log-likelihood for the WEIBULL MODEL
        """
        # params = {"v_h": v_h, "v_l_1": v_l_1, "v_l_2": v_l_2, "a": a,
        #           "z_h": z_h, "z_l_1": z_l_1, "z_l_2": z_l_2, "t": t}
        params = locals()
        size = int(value.shape[0])
        data = np.zeros((size, data_frame_width), dtype=np.float32)
        data[:, n_params:] = np.stack(
            [
                np.absolute(value["rt"]).astype(np.float32),
                value["response"].astype(np.float32),
            ],
            axis=1,
        )

        cnt = 0
        for tmp_str in model_parameter_names:  # model_config[model]['params']:
            if tmp_str in reg_outcomes:
                data[:, cnt] = params[tmp_str].loc[value["rt"].index].values
                if (data[:, cnt].min() < model_parameter_lower_bounds[cnt]) or (
                    data[:, cnt].max() > model_parameter_upper_bounds[cnt]
                ):
                    print("boundary violation of regressor part")
                    return -np.inf
            else:
                data[:, cnt] = params[tmp_str]
            cnt += 1

        # Has optimization potential --> AF-TODO: For next version!
        return hddm.wfpt.wiener_like_multi_nn_mlp(
            data, p_outlier=p_outlier, w_outlier=w_outlier, network=kwargs["network"]
        )  # **kwargs

    def wiener_multi_like_nn_mic2(
        value,
        vh,
        vl1,
        vl2,
        a,
        zh,
        zl1,
        zl2,
        d,
        t,
        reg_outcomes,
        p_outlier=0,
        w_outlier=0.1,
        **kwargs
    ):

        """
        LAN Log-likelihood for the WEIBULL MODEL
        """
        # params = {"v_h": v_h, "v_l_1": v_l_1, "v_l_2": v_l_2, "a": a,
        #           "z_h": z_h, "z_l_1": z_l_1, "z_l_2": z_l_2, "d": d, "t": t}
        params = locals()
        size = int(value.shape[0])
        data = np.zeros((size, data_frame_width), dtype=np.float32)
        data[:, n_params:] = np.stack(
            [
                np.absolute(value["rt"]).astype(np.float32),
                value["response"].astype(np.float32),
            ],
            axis=1,
        )

        cnt = 0
        for tmp_str in model_parameter_names:  # model_config[model]['params']:
            if tmp_str in reg_outcomes:
                data[:, cnt] = params[tmp_str].loc[value["rt"].index].values
                if (data[:, cnt].min() < model_parameter_lower_bounds[cnt]) or (
                    data[:, cnt].max() > model_parameter_upper_bounds[cnt]
                ):
                    print("boundary violation of regressor part")
                    return -np.inf
            else:
                data[:, cnt] = params[tmp_str]
            cnt += 1

        # Has optimization potential --> AF-TODO: For next version!
        return hddm.wfpt.wiener_like_multi_nn_mlp(
            data, p_outlier=p_outlier, w_outlier=w_outlier, network=kwargs["network"]
        )  # **kwargs

    likelihood_funs = {}
    likelihood_funs["test"] = wiener_multi_like_nn_test
    likelihood_funs["ddm"] = wiener_multi_like_nn_ddm
    likelihood_funs["full_ddm"] = wiener_multi_like_nn_full_ddm
    likelihood_funs["angle"] = wiener_multi_like_nn_angle
    likelihood_funs["levy"] = wiener_multi_like_nn_levy
    likelihood_funs["ornstein"] = wiener_multi_like_nn_ornstein
    likelihood_funs["weibull"] = wiener_multi_like_nn_weibull
    likelihood_funs["ddm_par2"] = wiener_multi_like_nn_par2
    likelihood_funs["ddm_seq2"] = wiener_multi_like_nn_seq2
    likelihood_funs["ddm_mic2"] = wiener_multi_like_nn_mic2
    likelihood_funs["custom"] = kwargs["likelihood_fun"]

    stoch = stochastic_from_dist("wfpt_reg", partial(likelihood_funs[model], **kwargs))
    stoch.pdf = pdf
    stoch.cdf = cdf
    stoch.random = random

    return stoch
