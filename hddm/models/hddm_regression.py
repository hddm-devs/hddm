from copy import deepcopy
import math
import numpy as np
import pymc as pm
import pandas as pd
from patsy import dmatrix

import hddm
from hddm.models import HDDM
import kabuki
from kabuki import Knode
from kabuki.utils import stochastic_from_dist
import kabuki.step_methods as steps

from hddm.simulators import *

# To fix regression
from hddm.model_config import model_config

# AF TEMPORARY
# def v_link_func(x, data):
#     stim = pd.Series(1, index = x.index)
#     data = data.loc[x.index]
#     stim.loc[data.tar_trial_type == 'nontarget'] = -1.
#     return x * stim


def generate_wfpt_reg_stochastic_class(
    wiener_params=None, sampling_method="cdf", cdf_range=(-5, 5), sampling_dt=1e-4
):
    # set wiener_params
    if wiener_params is None:
        wiener_params = {
            "err": 1e-4,
            "n_st": 2,
            "n_sz": 2,
            "use_adaptive": 1,
            "simps_err": 1e-3,
            "w_outlier": 0.1,
        }
    wp = wiener_params

    def wiener_multi_like(value, v, sv, a, z, sz, t, st, reg_outcomes, p_outlier=0.05):
        """Log-likelihood for the full DDM using the interpolation method"""
        params = {"v": v, "sv": sv, "a": a, "z": z, "sz": sz, "t": t, "st": st}
        for reg_outcome in reg_outcomes:
            params[reg_outcome] = params[reg_outcome].loc[value["rt"].index].values
        return hddm.wfpt.wiener_like_multi(
            value["rt"].values,
            params["v"],
            params["sv"],
            params["a"],
            params["z"],
            params["sz"],
            params["t"],
            params["st"],
            1e-4,
            reg_outcomes,
            w_outlier=wp["w_outlier"],
            p_outlier=p_outlier,
        )

    def random(
        self,
        keep_negative_responses=True,
        add_model_parameters=False,
        keep_subj_idx=False,
    ):
        assert sampling_method in ["drift", "cssm"], "Sampling method is invalid!"
        # AF add: exchange this with new simulator
        param_dict = deepcopy(self.parents.value)
        del param_dict["reg_outcomes"]
        sampled_rts = self.value.copy()

        if sampling_method == "drift":
            for i in self.value.index:
                # get current params
                for p in self.parents["reg_outcomes"]:
                    param_dict[p] = np.asscalar(self.parents.value[p].loc[i])
                # sample
                samples = hddm.generate.gen_rts(
                    method=sampling_method, size=1, dt=sampling_dt, **param_dict
                )

                sampled_rts.loc[i, "rt"] = hddm.utils.flip_errors(samples).rt.iloc[0]

            return sampled_rts

        if sampling_method == "cssm":
            param_data = np.zeros(
                (
                    self.value.shape[0],
                    len(model_config["full_ddm_hddm_base"]["params"]),
                ),
                dtype=np.float32,
            )
            cnt = 0
            for tmp_str in model_config["full_ddm_hddm_base"]["params"]:
                if tmp_str in self.parents["reg_outcomes"]:
                    # NOTE: Need to use .loc here to subset the regression parameters
                    # to account for eventual grouping
                    param_data[:, cnt] = (
                        param_dict[tmp_str].loc[self.value.index].values
                    )
                else:
                    param_data[:, cnt] = param_dict[tmp_str]
                cnt += 1

            sim_out = simulator(
                theta=param_data, model="full_ddm_hddm_base", n_samples=1, max_t=20
            )

            sim_out_proc = hddm_preprocess(
                sim_out,
                keep_negative_responses=keep_negative_responses,
                add_model_parameters=add_model_parameters,
                keep_subj_idx=keep_subj_idx,
            )

            sim_out_proc = hddm.utils.flip_errors(sim_out_proc)

            return sim_out_proc

    stoch = stochastic_from_dist("wfpt_reg", wiener_multi_like)
    stoch.random = random

    return stoch


wfpt_reg_like = generate_wfpt_reg_stochastic_class(sampling_method="cssm")  # "drift"

################################################################################################


class KnodeRegress(kabuki.hierarchical.Knode):
    def __init__(self, *args, **kwargs):
        # Whether or not to keep regressor trace
        self.keep_regressor_trace = kwargs.pop("keep_regressor_trace", False)

        # Initialize kabuki.hierarchical.Knode
        super(KnodeRegress, self).__init__(*args, **kwargs)

    def create_node(self, name, kwargs, data):
        reg = kwargs["regressor"]

        # order parents according to user-supplied args
        args = []
        for arg in reg["params"]:
            for parent_name, parent in kwargs["parents"].items():
                if parent_name == arg:
                    args.append(parent)

        parents = {"args": args}

        def func(
            args,
            design_matrix=dmatrix(
                reg["model"],
                data=self.data,  # Note: data is hardcoded here
                return_type="dataframe",
                NA_action="raise",
            ),
            link_func=reg["link_func"],
            # knode_data=data, AF-NOTE: I think this can be dropped
        ):
            # Convert parents to matrix
            params = np.matrix(args)
            design_matrix = design_matrix.loc[data.index]

            # Apply design matrix to input data
            if design_matrix.shape[1] != params.shape[1]:
                raise NotImplementedError(
                    "Missing columns in design matrix. You need data for all conditions for all subjects."
                )

            # predictor is the final regression outcome --> our parameter of interest
            predictor = link_func(design_matrix.dot(params.T)[0])
            return predictor

        # Build pymc node based on the information provided
        return self.pymc_node(
            func, kwargs["doc"], name, parents=parents, trace=self.keep_regressor_trace
        )


class HDDMRegressor(HDDM):
    """HDDMRegressor allows estimation of the DDM where parameter
    values are linear models of a covariate (e.g. a brain measure like
    fMRI or different conditions).
    """

    def __init__(
        self,
        data,
        models,
        group_only_regressors=True,
        keep_regressor_trace=False,
        **kwargs
    ):
        """Instantiate a regression model.

        :Arguments:

            * data : pandas.DataFrame
                data containing 'rt' and 'response' column and any
                covariates you might want to use.
            * models : str or list of str
                Patsy linear model specifier.
                E.g. 'v ~ cov'
                You can include multiple linear models that influence
                separate DDM parameters.

        :Optional:

            * group_only_regressors : bool (default=True)
                Do not estimate individual subject parameters for all regressors.
            * keep_regressor_trace : bool (default=False)
                Whether to keep a trace of the regressor. This will use much more space,
                but needed for posterior predictive checks.
            * Additional keyword args are passed on to HDDM.

        :Note:

            Internally, HDDMRegressor uses patsy which allows for
            simple yet powerful model specification. For more information see:
            http://patsy.readthedocs.org/en/latest/overview.html

        :Example:

            Consider you have a trial-by-trial brain measure
            (e.g. fMRI) as an extra column called 'BOLD' in your data
            frame. You want to estimate whether BOLD has an effect on
            drift-rate. The corresponding model might look like
            this:
                ```python
                HDDMRegressor(data, 'v ~ BOLD')
                ```

            This will estimate an v_Intercept and v_BOLD. If v_BOLD is
            positive it means that there is a positive correlation
            between BOLD and drift-rate on a trial-by-trial basis.

            This type of mechanism also allows within-subject
            effects. If you have two conditions, 'cond1' and 'cond2'
            in the 'conditions' column of your data you may
            specify:
                ```python
                HDDMRegressor(data, 'v ~ C(condition)')
                ```

            This will lead to estimation of 'v_Intercept' for cond1
            and v_C(condition)[T.cond2] for cond1+cond2.
        """
        self.keep_regressor_trace = keep_regressor_trace
        if isinstance(models, (str, dict)):
            models = [models]

        group_only_nodes = list(kwargs.get("group_only_nodes", ()))
        self.reg_outcomes = (
            set()
        )  # holds all the parameters that are going to modeled as outcome of regression models

        kwargs["group_only_nodes"] = self._prepare_model_descriptors(
            data=data,
            models=models,
            group_only_regressors=group_only_regressors,
            group_only_nodes=group_only_nodes,
        )

        # set wfpt_reg_class
        self.wfpt_reg_class = deepcopy(wfpt_reg_like)

        super(HDDMRegressor, self).__init__(data, **kwargs)

        # Sanity checks
        for model_descr in self.model_descrs:
            for param in model_descr["params"]:
                assert (
                    len(self.depends[param]) == 0
                ), "When using a regression on a given model parameter, you can not add it in the depends_on argument too."

    def __getstate__(self):
        d = super(HDDMRegressor, self).__getstate__()
        del d["wfpt_reg_class"]
        return d

    def __setstate__(self, d):
        d["wfpt_reg_class"] = deepcopy(wfpt_reg_like)
        super(HDDMRegressor, self).__setstate__(d)

    def _prepare_model_descriptors(
        self, data=None, models=None, group_only_regressors=True, group_only_nodes=None
    ):
        self.model_descrs = []
        group_only_nodes_tmp = deepcopy(group_only_nodes)
        # Cycle through list of regression models supplied
        for model in models:
            if isinstance(model, dict):
                try:
                    model_str = model["model"]
                    link_func = model["link_func"]
                except KeyError:
                    raise KeyError(
                        "HDDMRegressor requires a model specification either like {'model': 'v ~ 1 + C(your_variable)', 'link_func' lambda x: np.exp(x)} or just a model string"
                    )
            else:
                model_str = model
                link_func = lambda x: x

            # Find separator
            separator = model_str.find("~")
            assert separator != -1, "No outcome variable specified."

            # Separate model string into outcome and model
            outcome = model_str[:separator].strip(" ")
            model_stripped = model_str[(separator + 1) :]

            # Add outcome to reg_outcomes
            self.reg_outcomes.add(outcome)

            # Get covariate names from patsys dmatrix
            covariates = dmatrix(model_stripped, data).design_info.column_names

            # Build model descriptor
            model_descr = {
                "outcome": outcome,
                "model": model_stripped,
                "params": [
                    "{out}_{reg}".format(out=outcome, reg=reg) for reg in covariates
                ],
                "link_func": link_func,
            }
            self.model_descrs.append(model_descr)

            if group_only_regressors:
                group_only_nodes_tmp += model_descr["params"]
        return group_only_nodes_tmp

    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)
        return Knode(
            self.wfpt_reg_class,
            "wfpt",
            observed=True,
            col_name=["rt"],
            reg_outcomes=self.reg_outcomes,
            **wfpt_parents
        )

    def _create_stochastic_knodes(self, include):
        # Create all stochastic knodes except for the ones that we want to replace
        # with regressors.

        # Basic model parameters
        knodes = super(HDDMRegressor, self)._create_stochastic_knodes(
            include.difference(self.reg_outcomes)
        )

        # Indirect beta parameters
        if "indirect_betas" in self.model_config:
            for beta_tmp in self.model_config["indirect_betas"].keys():
                assert beta_tmp not in self.model_config["params"], (
                    "Naming conflict:"
                    + "beta_tmp "
                    + " is "
                    + beta_tmp
                    + ", but the parameter is already used as a"
                    + "basic model parameter!"
                )
                print("making indirect betas knode: ", beta_tmp)
                knodes.update(self._create_family_normal(beta_tmp, value=0.0))

                self.slice_widths[beta_tmp] = 0.05

        # This is in dire need of refactoring. Like any monster, it just grew over time.
        # The main problem is that it's not always clear which prior to use. For the intercept
        # we want to use the original parameters' prior. Also for categoricals that do not
        # have an intercept, but not when the categorical is part of an interaction....

        # create regressor params
        for reg in self.model_descrs:
            reg_parents = {}

            # Find intercept parameter
            intercept = (
                np.asarray([param.find("Intercept") for param in reg["params"]]) != -1
            )

            # If no intercept specified (via 0 + C()) assume all C() are different conditions
            # -> all are intercepts
            if not np.any(intercept):
                # Has categorical but no interaction
                intercept = np.asarray(
                    [
                        (param.find("C(") != -1) and (param.find(":") == -1)
                        for param in reg["params"]
                    ]
                )

            # check if link fuction is identity for current regressor
            # ------
            try:
                link_is_identity = np.array_equal(
                    reg["link_func"](np.arange(-10, 10, 0.01)),
                    np.arange(-10, 10, 0.01),
                )
            except:
                link_is_identity = False

            # print("Reg Model:")
            # print(reg)
            # ------

            for inter, param in zip(intercept, reg["params"]):
                trans = 0
                if inter:
                    # Intercept parameter should have original prior (not centered on 0)
                    param_lookup = param[: param.find("_")]

                    # Check if param_lookup is 'z' (for standard HDDM)
                    # Or more generally a parameter that should originally be transformed (for LAN extension --> self.nn is True)
                    is_indirect_regressor = 0
                    is_indirect_regressor_target = 0
                    if self.nn:
                        # If our current parameter is a indirect regressor we apply appropriate settings
                        if ("indirect_regressors" in self.model_config) and (
                            param_lookup in self.model_config["indirect_regressors"]
                        ):
                            is_indirect_regressor = 1
                            trans = 0
                            param_lower = self.model_config["indirect_regressors"][
                                param_lookup
                            ]["param_bounds"][0]
                            param_upper = self.model_config["indirect_regressors"][
                                param_lookup
                            ]["param_bounds"][1]
                            param_std_upper = 10

                            if (
                                "default_value"
                                in self.model_config["indirect_regressors"][
                                    param_lookup
                                ]
                            ):
                                default_val = self.model_config["indirect_regressors"][
                                    param_lookup
                                ]["default_value"]
                            else:
                                default_val = (param_upper - param_lower) / 2

                        # If our current parameter is a normal regressor we apply appropriate settings
                        else:
                            if ("indirect_regressors" in self.model_config) and (
                                param_lookup
                                in self.model_config["indirect_regressor_targets"]
                            ):
                                # If the current parameter is targeted by an indirect regressor,
                                # we need to apply special bounds, and it is not obvious which ones makes sense
                                # AF-TODO: Make this a bit more adaptive
                                is_indirect_regressor_target = 1
                                param_lower = -10
                                param_upper = 10
                                param_std_upper = 10
                                default_val = 0.0

                            param_id = self.model_config["params"].index(param_lookup)
                            trans = self.model_config["params_trans"][param_id]

                            if trans and (not is_indirect_regressor_target):
                                if not link_is_identity:
                                    default_val = 0.1

                                param_lower = self.model_config["param_bounds"][0][
                                    param_id
                                ]
                                param_upper = self.model_config["param_bounds"][1][
                                    param_id
                                ]

                                if not "params_std_upper" in self.model_config.keys():
                                    print(
                                        "Supplied model_config does not have a params_std_upper argument."
                                    )
                                    print("Set to a default of 10")
                                    param_std_upper = 10
                                elif (
                                    self.model_config["params_std_upper"][param_id]
                                    == None
                                ):
                                    print(
                                        "Supplied model_config specifies params_std_upper for ",
                                        param,
                                        "as ",
                                        "None.",
                                    )
                                    print("Changed to 10")
                                    param_std_upper = 10
                                else:
                                    param_std_upper = self.model_config[
                                        "params_std_upper"
                                    ][param_id]
                            elif trans and is_indirect_regressor_target:
                                raise ValueError(
                                    "Target to an indirect regressor is specified as parameter to transform for sampling. This is not allowed. \n"
                                    + 'Please make sure that the "trans" value for all targets to indirect regressors is set to 0'
                                )

                    elif param_lookup == "z":
                        trans = 1

                        # Used if link is not identity
                        default_val = 0.1
                        param_lower = 0
                        param_upper = 1
                        param_std_upper = 100

                    # If parameter is transformed and link function is identity --> apply usual prior
                    # If parameter is not transformed --> apply usual prior
                    # (Parameter should not be a indirect regressor, and not be the target of an indirect regressor for this)
                    if (
                        ((trans and link_is_identity) or (not trans))
                        and (not is_indirect_regressor)
                        and (not is_indirect_regressor_target)
                    ):
                        reg_family = super(
                            HDDMRegressor, self
                        )._create_stochastic_knodes([param_lookup])
                    else:  # Otherwise apply normal prior family
                        # If the parameter is listed as transformed, but non-identity link is assigned
                        # --> Actually do not transform parameter and apply truncated normal family prior
                        # There is no general strategy here, because we don't know the supplied link a priori
                        # AF-TD: Create Warning
                        # reg_family = self._create_family_trunc_normal(
                        #     param_lookup,
                        #     value=default_val,
                        #     lower=param_lower,
                        #     upper=param_upper,
                        #     std_upper=param_std_upper,
                        # )

                        reg_family = self._create_family_normal(
                            param_lookup,
                            value=default_val,
                            std_lower=1e-10,
                            std_upper=param_std_upper,
                            g_mu=0,
                            g_tau=15**-2,
                        )

                    # Rename nodes to avoid collissions
                    names = list(reg_family.keys())
                    for name in names:
                        knode = reg_family.pop(name)
                        knode.name = knode.name.replace(param_lookup, param, 1)
                        reg_family[name.replace(param_lookup, param, 1)] = knode
                    # switch param lookup to param
                    param_lookup = param

                else:
                    reg_family = self._create_family_normal(param)
                    param_lookup = param

                reg_parents[param] = reg_family["%s_bottom" % param_lookup]

                # AF-NOTE: BUG ?
                # reg will never be in self.group_only_nodes....
                if reg not in self.group_only_nodes:
                    reg_family["%s_subj_reg" % param] = reg_family.pop(
                        "%s_bottom" % param_lookup
                    )

                knodes.update(reg_family)

                self.slice_widths[param] = 0.05

            reg_knode = KnodeRegress(
                pm.Deterministic,
                "%s_reg" % reg["outcome"],
                regressor=reg,
                subj=self.is_group_model,
                plot=False,
                trace=False,
                hidden=True,
                keep_regressor_trace=self.keep_regressor_trace,
                **reg_parents
            )

            knodes["%s_bottom" % reg["outcome"]] = reg_knode

        return knodes


# Some standard link functions
def id_link(x):
    return x
