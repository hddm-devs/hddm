import hddm
from hddm.models import HDDMRegressor
from copy import deepcopy
from kabuki import Knode

try:
    from hddm.torch.mlp_inference_class import load_torch_mlp
except:
    print(
        "It seems that you do not have pytorch installed."
        + "The HDDMnn, HDDMnnRegressor, HDDMnnStimCoding, HDDMnnRL and HDDMnnRLRegressor"
        + "classes will not work"
    )


class HDDMnnRLRegressor(HDDMRegressor):
    """HDDMnnRLRegressor allows estimation of the RLSSM where parameter
    values are linear models of a covariate (e.g. a brain measure like
    fMRI or different conditions).
    """

    def __init__(
        self,
        data,
        models,
        group_only_regressors=True,
        keep_regressor_trace=False,
        indirect_regressors=None,
        indirect_betas=None,
        **kwargs
    ):
        """Instantiate a regression model, with neural network based likelihoods.

        :Arguments:

            data : pandas.DataFrame
                data containing 'rt', 'response', column and any
                covariates you might want to use.
            models : str or list of str
                Patsy linear model specifier.
                E.g. 'v ~ cov'
                You can include multiple linear models that influence
                separate DDM parameters.

        :Optional:

            model: str <default='ddm'>
                String that determines which sequential sampling model you would like to fit your data to.
                Currently available models are: 'ddm', 'full_ddm', 'angle', 'weibull', 'ornstein', 'levy'
            rl_rule: str <default='RWupdate'>
                String that determines which reinforcement learning model you would like to fit your data to.
            include: list <default=None>
                A list with parameters we wish to include in the fitting procedure.
                Which parameters you can include depends on the model you specified under the model parameters.
            non_centered: bool <default=False>
                Denotes whether non-centered distributions (a form of re-parameterization) should be used for reinforcement learning parameters.
            group_only_regressors : bool (default = True)
                Do not estimate individual subject parameters for all regressors.
            keep_regressor_trace : bool (default = False)
                Whether to keep a trace of the regressor. This will use much more space,
                but needed for posterior predictive checks.
            Additional keyword args are passed on to HDDM.

        :Note:

            Internally, HDDMnnRLRegressor uses patsy which allows for
            simple yet powerful model specification. For more information see:
            http://patsy.readthedocs.org/en/latest/overview.html

        :Example:

            Consider you have a trial-by-trial brain measure
            (e.g. fMRI) as an extra column called 'BOLD' in your data
            frame. You want to estimate whether BOLD has an effect on
            drift-rate. The corresponding model might look like
            this:
                ```python
                HDDMnnRLRegressor(data, 'a ~ BOLD', model='angle', rl_rule='RWupdate', include=['v', 'a', 't', 'z', 'theta', 'rl_alpha'])
                ```

            This will estimate an v_Intercept and v_BOLD. If v_BOLD is
            positive it means that there is a positive correlation
            between BOLD and drift-rate on a trial-by-trial basis.

            This type of mechanism also allows within-subject
            effects. If you have two conditions, 'cond1' and 'cond2'
            in the 'conditions' column of your data you may
            specify:
                ```python
                HDDMnnRLRegressor(data, 'a ~ C(condition)', model='angle', rl_rule='RWupdate', include=['v', 'a', 't', 'z', 'theta', 'rl_alpha'])
                ```
            This will lead to estimation of 'v_Intercept' for cond1
            and v_C(condition)[T.cond2] for cond1 + cond2.

        """
        # Signify as neural net class for later super() inits
        self.nn = True
        self.rlssm_model = True
        self.nn_rl_reg = True

        if "informative" in kwargs.keys():
            pass
        else:
            print("Using default priors: Uninformative")
            kwargs["informative"] = False

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

        # Add indirect_regressors to model_config, if they were supplied
        self._add_indirect_regressors(indirect_regressors=indirect_regressors)

        # Add indirect betas to model_config, if they were supplied
        self._add_indirect_betas(indirect_betas=indirect_betas)

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

        self.wfpt_nn_rl_reg_class = hddm.likelihoods_mlp.make_mlp_likelihood_reg_nn_rl(
            model=self.model,
            model_config=self.model_config,
            model_config_rl=self.model_config_rl,
            wiener_params=kwargs["wiener_params"],
            **network_dict
        )

        super(HDDMnnRLRegressor, self).__init__(
            data, models, group_only_regressors, keep_regressor_trace, **kwargs
        )

    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)

        return Knode(
            self.wfpt_nn_rl_reg_class,
            "wfpt",
            observed=True,
            col_name=["split_by", "feedback", "response", "rt", "q_init"]
            + self.model_config["likelihood_relevant_covariates"],
            reg_outcomes=self.reg_outcomes,
            **wfpt_parents
        )

    def _add_indirect_betas(self, indirect_betas=None):
        self.model_config["likelihood_relevant_covariates"] = []

        if indirect_betas is not None:
            assert (
                type(indirect_betas) == dict
            ), "indirect parameters is supplied, but is not a dictionary"
            self.model_config["indirect_betas"] = indirect_betas

            relevant_covariates = []
            for indirect_beta_tmp in indirect_betas.keys():
                for linked_covariate_tmp in indirect_betas[indirect_beta_tmp][
                    "links_to"
                ].keys():
                    tmp = indirect_betas[indirect_beta_tmp]["links_to"][
                        linked_covariate_tmp
                    ]

                    if tmp in relevant_covariates:
                        pass
                    else:
                        relevant_covariates.append(tmp)

            self.model_config["likelihood_relevant_covariates"] = relevant_covariates

    def _add_indirect_regressors(self, indirect_regressors=None):
        if indirect_regressors is not None:
            assert (
                type(indirect_regressors) == dict
            ), "indirect_regressors is supplied, but not as a dictionary"
            self.model_config["indirect_regressors"] = indirect_regressors

            # Compute all indirect regressor targets
            indirect_regressor_targets = []
            for indirect_regressor in self.model_config["indirect_regressors"].keys():
                for target_tmp in self.model_config["indirect_regressors"][
                    indirect_regressor
                ]["links_to"]:
                    indirect_regressor_targets.append(target_tmp)

            self.model_config["indirect_regressor_targets"] = indirect_regressor_targets

    def __getstate__(self):
        d = super(HDDMnnRLRegressor, self).__getstate__()
        del d["wfpt_nn_rl_reg_class"]

        return d

    def __setstate__(self, d):
        network_dict = {"network": d["network"]}

        d["wfpt_nn_rl_reg_class"] = hddm.likelihoods_mlp.make_mlp_likelihood_reg_nn_rl(
            model=d["model"],
            model_config=d["model_config"],
            model_config_rl=d["model_config_rl"],
            wiener_params=d["wiener_params"],
            **network_dict
        )

        super(HDDMnnRLRegressor, self).__setstate__(d)
