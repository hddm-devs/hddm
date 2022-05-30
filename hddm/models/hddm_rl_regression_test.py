from copy import deepcopy
import math
import numpy as np
import pymc as pm
import pandas as pd
from patsy import dmatrix
from functools import partial

import hddm
from hddm.models import HDDM
import kabuki
from kabuki import Knode
from kabuki.utils import stochastic_from_dist

try:
    from hddm.torch.mlp_inference_class import load_torch_mlp
except:
    print(
        "It seems that you do not have pytorch installed."
        + "The HDDMnn, HDDMnnRegressor, HDDMnnStimCoding, HDDMnnRL and HDDMnnRLRegressor"
        + "classes will not work"
    )


def generate_wfpt_rl_reg_stochastic_class(
    model=None, model_config=None, model_config_rl=None, wiener_params=None, **kwargs
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

    def wienerRL_multi_like(value, v, a, z, t, rl_alpha, reg_outcomes, p_outlier=0, w_outlier=0.1, **kwargs):
        
        """Log-likelihood for the full DDM using the interpolation method"""
        response = value["response"].values.astype(int)
        q = value["q_init"].iloc[0]
        feedback = value["feedback"].values.astype(float)
        split_by = value["split_by"].values.astype(int)
        ssm_params = {
            "v": v,
            "a": a,
            "z": z,
            "t": t
        }
        rl_params = {
            "rl_alpha": rl_alpha
        }

        data = np.zeros((len(response), len(ssm_params) + 2), dtype=np.float32)
        data[:, len(ssm_params):] = np.stack([np.absolute(value["rt"]).astype(np.float32), value["response"].astype(np.float32)], axis=1)

        data_rl = np.zeros((len(rl_params), len(response)), dtype=np.float32)

        param_str = ['v', 'a', 'z', 't', 'rl_alpha']
        cnt = 0
        for tmp in param_str:
            if tmp in reg_outcomes and tmp in ssm_params:
                data[:, cnt] = np.squeeze(ssm_params[tmp].loc[value["rt"].index].values)
            elif tmp not in reg_outcomes and tmp in ssm_params:
                data[:, cnt] = ssm_params[tmp]
            
            if tmp in reg_outcomes and tmp in rl_params:
                data_rl[:, cnt] = rl_params[tmp].loc[value["rt"].index].values
            elif tmp not in reg_outcomes and tmp in rl_params:
                data_rl[:, cnt] = rl_params[tmp]

            cnt += 1
        

        return hddm.wfpt.wiener_like_rlssm_nn_reg(data, data_rl, 
            value["rt"].values.astype(float), value["response"].values.astype(int), value["feedback"].values, value["split_by"].values.astype(int), value["q_init"].iloc[0], 
            params_bnds=np.array([[-3.0, 0.3, 0.1, 0.001, 0.0], [3.0, 2.5, 0.9, 2.0, 1.0]]), network=kwargs["network"], p_outlier=p_outlier, w_outlier=w_outlier)


    #stoch = stochastic_from_dist("wfpt_reg", wienerRL_multi_like)
    print("here kwargs -- ", kwargs.keys())
    stoch = stochastic_from_dist("wfpt_reg", partial(wienerRL_multi_like, **kwargs))

    return stoch


#wfpt_reg_like = generate_wfpt_rl_reg_stochastic_class(model=None, model_config=None, model_config_rl=None, wiener_params=None, sampling_method="drift", **kwargs)

################################################################################################


class KnodeRegress(kabuki.hierarchical.Knode):
    def __init__(self, *args, **kwargs):
        self.keep_regressor_trace = kwargs.pop("keep_regressor_trace", False)
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

        # Make sure design matrix is kosher
        dm = dmatrix(reg["model"], data=data)
        if math.isnan(dm.sum()):
            raise NotImplementedError("DesignMatrix contains NaNs.")

        def func(
            args,
            design_matrix=dmatrix(reg["model"], data=data),
            link_func=reg["link_func"],
        ):
            # convert parents to matrix
            params = np.matrix(args)
            # Apply design matrix to input data
            if design_matrix.shape[1] != params.shape[1]:
                raise NotImplementedError(
                    "Missing columns in design matrix. You need data for all conditions for all subjects."
                )
            predictor = link_func(
                pd.DataFrame((design_matrix * params).sum(axis=1), index=data.index)
            )

            return pd.DataFrame(predictor, index=data.index)

        return self.pymc_node(
            func, kwargs["doc"], name, parents=parents, trace=self.keep_regressor_trace
        )


class HDDMnnRLRegressorTest(HDDM):
    """HDDMrlRegressor allows estimation of the RLDDM where parameter
    values are linear models of a covariate (e.g. a brain measure like
    fMRI or different conditions).
    """

    def __init__(
        self,
        data,
        ssm_model,
        rl_rule,
        models,
        group_only_regressors=True,
        keep_regressor_trace=False,
        **kwargs
    ):
        """Instantiate a regression model.

        :Arguments:

            * data : pandas.DataFrame
                data containing 'rt', 'response','feedback','split_by' and 'q_init' column and any
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

            Internally, HDDMrlRegressor uses patsy which allows for
            simple yet powerful model specification. For more information see:
            http://patsy.readthedocs.org/en/latest/overview.html

        :Example:

            Consider you have a trial-by-trial brain measure
            (e.g. fMRI) as an extra column called 'BOLD' in your data
            frame. You want to estimate whether BOLD has an effect on
            drift-rate. The corresponding model might look like
            this:
                ```python
                HDDMrlRegressor(data, 'v ~ BOLD')
                ```

            This will estimate an v_Intercept and v_BOLD. If v_BOLD is
            positive it means that there is a positive correlation
            between BOLD and drift-rate on a trial-by-trial basis.

            This type of mechanism also allows within-subject
            effects. If you have two conditions, 'cond1' and 'cond2'
            in the 'conditions' column of your data you may
            specify:
                ```python
                HDDMrlRegressor(data, 'v ~ C(condition)')
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
        )  # holds all the parameters that are going to modeled as outcome

        self.model_descrs = []

        for model in models:
            if isinstance(model, dict):
                try:
                    model_str = model["model"]
                    link_func = model["link_func"]
                except KeyError:
                    raise KeyError(
                        "HDDMrlRegressor requires a model specification either like {'model': 'v ~ 1 + C(your_variable)', 'link_func' lambda x: np.exp(x)} or just a model string"
                    )
            else:
                model_str = model
                link_func = lambda x: x

            separator = model_str.find("~")
            assert separator != -1, "No outcome variable specified."
            outcome = model_str[:separator].strip(" ")
            model_stripped = model_str[(separator + 1) :]
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

            print("Adding these covariates:")
            print(model_descr["params"])
            if group_only_regressors:
                group_only_nodes += model_descr["params"]
                kwargs["group_only_nodes"] = group_only_nodes
            self.reg_outcomes.add(outcome)

        # set wfpt_reg
        #self.wfpt_reg_class = deepcopy(wfpt_reg_like)

        # super(HDDMrlRegressor, self).__init__(data, **kwargs)

        # # Sanity checks
        # for model_descr in self.model_descrs:
        #     for param in model_descr["params"]:
        #         assert (
        #             len(self.depends[param]) == 0
        #         ), "When using patsy, you can not use any model parameter in depends_on."
        
        # ===
        # 
        # ===

        self.nn = True
        self.rlssm_model = True
        self.nn_rl_reg = True

        self.network = kwargs.pop("network", None)
        self.non_centered = kwargs.pop("non_centered", False)

        self.w_outlier = kwargs.pop("w_outlier", 0.1)
        self.model = kwargs.pop("model", "ddm")
        self.rl_rule = kwargs.pop("rl_rule", "RWupdate")
        self.model_config = kwargs.pop("model_config", None)
        self.model_config_rl = kwargs.pop("model_config_rl", None)

        print("\nPrinting model specifications -- ")
        print("ssm: ", self.model)
        print("rl rule: ", self.rl_rule)
        print("using non-centered dist.: ", self.non_centered)

        if not "wiener_params" in kwargs.keys():
            kwargs["wiener_params"] = {
                "err": 1e-4,
                "n_st": 2,
                "n_sz": 2,
                "use_adaptive": 1,
                "simps_err": 1e-3,
                "w_outlier": 0.1,
            }

        # If no config was supplied try loading a config from
        # the configs included in the package
        if self.model_config == None:
            try:
                self.model_config = deepcopy(hddm.model_config.model_config[self.model])
            except:
                print(
                    "It seems that you supplied a model string that refers to an undefined model"
                )
        
        
        if self.model_config_rl == None:
            try:
                self.model_config_rl = deepcopy(
                    hddm.model_config_rl.model_config_rl[self.rl_rule]
                )
            except:
                print(
                    "It seems that you supplied a model string that refers to an undefined model."
                    + "This works only if you supply a custom model_config_rl dictionary."
                )


        if self.network is None:
            try:
                self.network = load_torch_mlp(model=self.model)
            except:
                print("Couldn't execute load_torch_mlp()...")
                print("Option 1: pytorch not installed or version older than 1.7?")
                print(
                    "Option 2: pytorch model for your model string is not yet available"
                )
                return None

        network_dict = {"network": self.network}

        self.wfpt_reg_class = generate_wfpt_rl_reg_stochastic_class(
            model=self.model,
            model_config=self.model_config,
            model_config_rl=self.model_config_rl,
            wiener_params=kwargs["wiener_params"],
            **network_dict
        )

        super(HDDMnnRLRegressorTest, self).__init__(
            data, **kwargs
        )

    def __getstate__(self):
        d = super(HDDMnnRLRegressorTest, self).__getstate__()
        del d["wfpt_reg_class"]
        for model in d["model_descrs"]:
            if "link_func" in model:
                # print("WARNING: Will not save custom link functions.")
                del model["link_func"]
        return d

    def __setstate__(self, d):
        d["wfpt_reg_class"] = deepcopy(wfpt_reg_like)
        # print("WARNING: Custom link functions will not be loaded.")
        for model in d["model_descrs"]:
            model["link_func"] = lambda x: x
        super(HDDMnnRLRegressorTest, self).__setstate__(d)

    def _create_stochastic_knodes_rl(self, include):
        knodes = super(HDDMnnRLRegressorTest, self)._create_stochastic_knodes(include)
        if "rl_alpha" in include:
            knodes.update(
                self._create_family_normal_normal_hnormal(
                    "rl_alpha", value=0, g_tau=50**-2, std_std=10
                )
            )
        return knodes

    def _create_wfpt_knode(self, knodes):
        wfpt_parents = super(HDDMnnRLRegressorTest, self)._create_wfpt_parents_dict(knodes)
        wfpt_parents["rl_alpha"] = knodes["rl_alpha_bottom"]
        return Knode(
            self.wfpt_reg_class,
            "wfpt",
            observed=True,
            col_name=["split_by", "feedback", "response", "rt", "q_init"],
            reg_outcomes=self.reg_outcomes,
            **wfpt_parents
        )

    def _create_stochastic_knodes(self, include):
        # Create all stochastic knodes except for the ones that we want to replace
        # with regressors.
        knodes = self._create_stochastic_knodes_rl(
            include.difference(self.reg_outcomes)
        )

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

            for inter, param in zip(intercept, reg["params"]):
                if inter:
                    # Intercept parameter should have original prior (not centered on 0)
                    param_lookup = param[: param.find("_")]
                    reg_family = self._create_stochastic_knodes_rl([param_lookup])
                    # Rename nodes to avoid collissions
                    names = list(reg_family.keys())
                    for name in names:
                        knode = reg_family.pop(name)
                        knode.name = knode.name.replace(param_lookup, param, 1)
                        reg_family[name.replace(param_lookup, param, 1)] = knode
                    param_lookup = param

                else:
                    # param_lookup = param[:param.find('_')]
                    reg_family = self._create_family_normal(param)
                    param_lookup = param

                reg_parents[param] = reg_family["%s_bottom" % param_lookup]
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
