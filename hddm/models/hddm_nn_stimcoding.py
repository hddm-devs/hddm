from hddm.models import HDDMStimCoding
from hddm.models.hddm_stimcoding import KnodeWfptStimCoding

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

        network_type: str <default='mlp'>
            String that defines which kind of network to use for the likelihoods. There are currently two
            options: 'mlp', 'cnn'. CNNs should be treated as experimental at this point.

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
        print(
            "Setting priors uninformative (LANs only work with uninformative priors for now)"
        )
        kwargs["informative"] = False
        self.network_type = kwargs.pop("network_type", "torch_mlp")
        self.network = kwargs.pop("network", None)
        self.non_centered = kwargs.pop("non_centered", False)
        self.w_outlier = kwargs.pop("w_outlier", 0.1)
        self.model = kwargs.pop("model", "ddm")

        if self.network_type == "torch_mlp":
            if self.network is None:
                try:
                    self.network = load_torch_mlp(model=self.model)
                except:
                    print("Couldn't find load_torch_mlp()... pytorch not installed?")
                    return None
            network_dict = {"network": self.network}
            self.wfpt_nn = hddm.likelihoods_mlp.make_mlp_likelihood(
                model=self.model, **network_dict
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
            ],  # Note: This is different from vanilla stimcoding class where it was set to col_name = 'rt',
            depends=[self.stim_col],
            split_param=self.split_param,
            stims=self.stims,
            stim_col=self.stim_col,
            **wfpt_parents
        )

    def __getstate__(self):
        d = super(HDDMnnStimCoding, self).__getstate__()
        del d["network"]
        del d["wfpt_nn"]
        return d

    def __setstate__(self, d):

        if d["network_type"] == "torch_mlp":
            d["network"] = load_torch_mlp(model=d["model"])
            network_dict = {"network": d["network"]}
            d["wfpt_nn"] = hddm.likelihoods_mlp.make_mlp_likelihood(
                model=d["model"], **network_dict
            )

        super(HDDMnnStimCoding, self).__setstate__(d)
