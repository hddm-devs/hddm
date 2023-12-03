from hddm.models import HDDMStimCoding
from hddm.models.hddm_stimcoding import KnodeWfptStimCoding
from copy import deepcopy

try:
    # print('HDDM: Trying import of pytorch related classes.')
    from hddm.torch.mlp_inference_class import load_torch_mlp
except:
    print(
        "It seems that you do not have pytorch installed."
        + "The HDDMnn, HDDMnnRegressor and HDDMnnStimCoding"
        + "classes will not work"
    )

import hddm


class HDDMnnStimCoding(HDDMStimCoding):
    """HDDMnn model that can be used when stimulus coding and estimation
    of bias (i.e. displacement of starting point z) is required.

    In that case, the 'resp' column in your data should contain 0 and
    1 for the chosen stimulus (or direction), not whether the response
    was correct or not as you would use in accuracy coding. You then
    have to provide another column (referred to as stim_col) which
    contains information about which the correct response was.

    HDDMnnStimCoding distinguishes itself from the HDDMStimCoding class by allowing you
    to specify a variety of generative models. Likelihoods are based on Neural Networks.

    :Arguments:
        model: str <default='ddm'>
            String that determines which model you would like to fit your data to.
            Currently available models are: 'ddm', 'full_ddm', 'angle', 'weibull', 'ornstein', 'levy'

        nbin: int <default=512>
            Relevant only if network type was chosen to be 'cnn'. CNNs can be trained on coarser or
            finer binnings of RT space. At this moment only networks with 512 bins are available.

        include: list <default=None>
            A list with parameters we wish to include in the fitting procedure. Generally, per default included
            in fitting are the drift parameter 'v', the boundary separation parameter 'a' and the non-decision-time 't'.
            Which parameters you can include depends on the model you specified under the model parameters.

        split_param : {'v', 'z'} <default='z'>
            There are two ways to model stimulus coding in the case where both stimuli
            have equal information (so that there can be no difference in drift):
            * 'z': Use z for stimulus A and 1-z for stimulus B
            * 'v': Use drift v for stimulus A and -v for stimulus B

        stim_col : str
            Column name for extracting the stimuli to use for splitting.

        drift_criterion : bool <default=False>
            Whether to estimate a constant factor added to the drift-rate.
            Requires split_param='v' to be set.

    """

    def __init__(self, *args, **kwargs):
        self.nn = True
        self.rlssm_model = kwargs.pop("rlssm_model", False)

        if "informative" in kwargs.keys():
            pass
        else:
            print("Setting priors to default: Uninformative")
            kwargs["informative"] = False

        self.network = kwargs.pop("network", None)
        self.non_centered = kwargs.pop("non_centered", False)
        self.w_outlier = kwargs.pop("w_outlier", 0.1)
        self.model = kwargs.pop("model", "ddm")
        self.model_config = kwargs.pop("model_config", None)

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
        self.wfpt_nn = hddm.likelihoods_mlp.make_mlp_likelihood(
            model=self.model,
            model_config=self.model_config,
            wiener_params=kwargs["wiener_params"],
            **network_dict
        )

        super(HDDMnnStimCoding, self).__init__(*args, **kwargs)

    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)
        # Here we use a special Knode (see below) that either inverts v or z
        # depending on what the correct stimulus was for that trial type.

        return KnodeWfptStimCoding(
            self.wfpt_nn,
            "wfpt",  # TD: ADD wfpt class we need
            observed=True,
            col_name=[
                "rt",
                "response",
            ],  # Note: This is different from hddm_base stimcoding class where it was set to col_name = 'rt',
            depends=[self.stim_col],
            split_param=self.split_param,
            stims=self.stims,
            stim_col=self.stim_col,
            **wfpt_parents
        )

    def __getstate__(self):
        d = super(HDDMnnStimCoding, self).__getstate__()
        # del d["network"]
        del d["wfpt_nn"]
        return d

    def __setstate__(self, d):
        # d["network"] = load_torch_mlp(model=d["model"])
        network_dict = {"network": d["network"]}
        d["wfpt_nn"] = hddm.likelihoods_mlp.make_mlp_likelihood(
            model=d["model"], **network_dict
        )

        super(HDDMnnStimCoding, self).__setstate__(d)
