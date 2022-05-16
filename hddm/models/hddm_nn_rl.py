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
        + "The HDDMnn, HDDMnnRL, HDDMnnRegressor and HDDMnnStimCoding"
        + "classes will not work"
    )


class HDDMnnRL(HDDMnn):
    """HDDMnn model class that uses neural network based likelihoods in conjuction with RL rules."""

    def __init__(self, *args, **kwargs):
        self.rlssm_model = True

        self.model = kwargs.pop("model", None)
        self.rl_rule = kwargs.pop("rl_rule", "RWupdate")
        self.model_config = kwargs.pop("model_config", None)
        self.non_centered = kwargs.pop("non_centered", False)

        print("\nPrinting model specifications -- ")
        print("ssm: ", self.model)
        print("rl rule: ", self.rl_rule)
        print("using centered dist.: ", self.non_centered)

        self.model_config_rl = kwargs.pop("model_config_rl", None)
        if self.model_config_rl == None:
            try:
                self.model_config_rl = deepcopy(hddm.model_config_rl.model_config_rl[self.rl_rule])
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

        self.network = load_torch_mlp(model=self.model)

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
        super(HDDMnnRL, self).__init__(model=self.model, network=self.network, *args, **kwargs)

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
            model=d["model"], model_config=d["model_config"], 
            model_config_rl=d["model_config_rl"],
            wiener_params = d['wiener_params'], **network_dict
        )

        super(HDDMnnRL, self).__setstate__(d)