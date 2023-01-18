"""
"""
import numpy as np
import hddm
from kabuki.hierarchical import (
    Knode,
)

# from kabuki.utils import stochastic_from_dist
from hddm.models import HDDMnn
from copy import deepcopy

try:
    from hddm.torch.mlp_inference_class import load_torch_mlp
except:
    print(
        "It seems that you do not have pytorch installed."
        + "The HDDMnn, HDDMnnRL, HDDMnnRegressor, HDDMnnStimCoding, HDDMnnRL and HDDMnnRLRegressor"
        + "classes will not work"
    )


class HDDMnnRL(HDDMnn):
    """HDDMnn model class that uses neural network based likelihoods in conjuction with RL rules.

    :Arguments:
        data: pandas.DataFrame
            Input data with a row for each trial.

            Must contain the following columns:
            * 'rt': Reaction time of trial in seconds.
            * 'response': Binary response (e.g. 0->error, 1->correct)
            * 'subj_idx': A unique ID (int) of each subject.
            * Other user-defined columns that can be used in depends_on keyword.

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

        informative : bool <default=True>
            Whether to use informative priors (True) or vague priors
            (False).  Informative priors are not yet implemented for neural network based
            models.

        is_group_model : bool
            If True, this results in a hierarchical
            model with separate parameter distributions for each
            subject. The subject parameter distributions are
            themselves distributed according to a group parameter
            distribution.

        p_outlier : double (default=0.05)
            The probability of outliers in the data. if p_outlier is passed in the
            'include' argument, then it is estimated from the data and the value passed
            using the p_outlier argument is ignored.



    :Example:
        >>> m = hddm.HDDMnnRL(data, model='angle', rl_rule='RWupdate', include=['v', 'a', 't', 'z', 'theta', 'rl_alpha'], p_outlier = 0.0)
        >>> m.sample(2000, burn=1000, dbname='traces.db', db='pickle')

    """

    def __init__(self, *args, **kwargs):
        self.rlssm_model = True

        self.model = kwargs.pop("model", "ddm")
        self.rl_rule = kwargs.pop("rl_rule", "RWupdate")
        self.non_centered = kwargs.pop("non_centered", False)
        self.model_config = kwargs.pop("model_config", None)
        self.model_config_rl = kwargs.pop("model_config_rl", None)

        self.network = kwargs.pop("network", None)

        print("\nPrinting model specifications -- ")
        print("ssm: ", self.model)
        print("rl rule: ", self.rl_rule)
        print("using non-centered dist.: ", self.non_centered)

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

        if self.model_config == None:
            try:
                self.model_config = deepcopy(hddm.model_config.model_config[self.model])
            except:
                print(
                    "It seems that you supplied a model string that refers to an undefined model."
                    + "This works only if you supply a custom model_config dictionary."
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

        kwargs_dict = {"network": self.network}

        if not "wiener_params" in kwargs.keys():
            kwargs["wiener_params"] = {
                "err": 1e-4,
                "n_st": 2,
                "n_sz": 2,
                "use_adaptive": 1,
                "simps_err": 1e-3,
                "w_outlier": 0.1,
            }

        self.wfpt_nn_rlssm = hddm.likelihoods_mlp.make_mlp_likelihood_rlssm(
            model=self.model,
            model_config=self.model_config,
            model_config_rl=self.model_config_rl,
            wiener_params=kwargs["wiener_params"],
            **kwargs_dict
        )

        # Initialize super class
        super(HDDMnnRL, self).__init__(
            model=self.model,
            network=self.network,
            non_centered=self.non_centered,
            *args,
            **kwargs
        )

    def _create_wfpt_parents_dict(self, knodes):
        wfpt_parents = super(HDDMnnRL, self)._create_wfpt_parents_dict(knodes)

        return wfpt_parents

    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)

        return Knode(
            self.wfpt_nn_rlssm,
            "wfpt",
            observed=True,
            col_name=["split_by", "feedback", "response", "rt", "q_init"],
            **wfpt_parents
        )

    def __getstate__(self):
        d = super(HDDMnnRL, self).__getstate__()
        del d["wfpt_nn_rlssm"]

        return d

    def __setstate__(self, d):
        network_dict = {"network": d["network"]}
        d["wfpt_nn_rlssm"] = hddm.likelihoods_mlp.make_mlp_likelihood_rlssm(
            model=d["model"],
            model_config=d["model_config"],
            model_config_rl=d["model_config_rl"],
            wiener_params=d["wiener_params"],
            **network_dict
        )

        super(HDDMnnRL, self).__setstate__(d)
