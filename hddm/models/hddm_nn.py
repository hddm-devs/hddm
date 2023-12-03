"""
"""
import hddm
from kabuki.hierarchical import (
    Knode,
)

# from kabuki.utils import stochastic_from_dist
from hddm.models import HDDM
from copy import deepcopy

try:
    from hddm.torch.mlp_inference_class import load_torch_mlp
except:
    print(
        "It seems that you do not have pytorch installed."
        + "The HDDMnn, HDDMnnRegressor, HDDMnnStimCoding, HDDMnnRL and HDDMnnRLRegressor"
        + "classes will not work"
    )


class HDDMnn(HDDM):
    """HDDM model class that uses neural network based likelihoods to include a variety of other models.

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
            String that determines which model you would like to fit your data to.
            Currently available models are: 'ddm', 'full_ddm', 'angle', 'weibull', 'ornstein', 'levy'

        nbin: int <default=512>
            Relevant only if network type was chosen to be 'cnn'. CNNs can be trained on coarser or
            finer binnings of RT space. At this moment only networks with 512 bins are available.

        include: list <default=None>
            A list with parameters we wish to include in the fitting procedure. Generally, per default included
            in fitting are the drift parameter 'v', the boundary separation parameter 'a' and the non-decision-time 't'.
            Which parameters you can include depends on the model you specified under the model parameters.

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

        depends_on : dict
            Specifies which parameter depends on data
            of a column in data. For each unique element in that
            column, a separate set of parameter distributions will be
            created and applied. Multiple columns can be specified in
            a sequential container (e.g. list)

            :Example:

                >>> hddm.HDDM(data, depends_on={'v': 'difficulty'})

                Separate drift-rate parameters will be estimated
                for each difficulty. Requires 'data' to have a
                column difficulty.

        bias : bool
            Whether to allow a bias to be estimated. This
            is normally used when the responses represent
            left/right and subjects could develop a bias towards
            responding right. This is normally never done,
            however, when the 'response' column codes
            correct/error.

        p_outlier : double (default=0.05)
            The probability of outliers in the data. if p_outlier is passed in the
            'include' argument, then it is estimated from the data and the value passed
            using the p_outlier argument is ignored.

        default_intervars : dict (default = {'sz': 0, 'st': 0, 'sv': 0})
            Fix intertrial variabilities to a certain value. Note that this will only
            have effect for variables not estimated from the data. This is relevant only when fitting the
            Full-DDM model, and not when fitting e.g. the weibull, angle, ornstein or levy models.

        plot_var : bool
             Plot group variability parameters when calling pymc.Matplot.plot()
             (i.e. variance of Normal distribution.)

        trace_subjs : bool
             Save trace for subjs (needed for many
             statistics so probably a good idea.)

        std_depends : bool (default=False)
             Should the depends_on keyword affect the group std node.
             If True it means that both, group mean and std will be split
             by condition.

    :Example:
        >>> data, params = hddm.generate.gen_rand_data() # gen data
        >>> model = hddm.HDDMnn(data, model = 'angle') # create object
        >>> mcmc.sample(500, burn=20) # Sample from posterior

    """

    def __init__(self, *args, **kwargs):
        self.nn = True

        if "informative" in kwargs.keys():
            pass
        else:
            print("Using default priors: Uninformative")
            kwargs["informative"] = False

        self.network = kwargs.pop("network", None)  # LAX
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

        network_dict = {"network": self.network}
        self.wfpt_nn = hddm.likelihoods_mlp.make_mlp_likelihood(
            model=self.model,
            model_config=self.model_config,
            wiener_params=kwargs["wiener_params"],
            **network_dict
        )

        # Initialize super class
        super(HDDMnn, self).__init__(*args, **kwargs)

    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)
        return Knode(
            self.wfpt_nn,
            "wfpt",
            observed=True,
            col_name=[
                "response",
                "rt",
            ],  # TODO: One could preprocess at initialization
            **wfpt_parents
        )

    def __getstate__(self):
        d = super(HDDMnn, self).__getstate__()
        # del d["network"] # del
        # temporary
        del d["wfpt_nn"]
        return d

    def __setstate__(self, d):
        # print(d)

        # print(d["network"]) # del
        # d["network"] = load_torch_mlp(model=d["model"]) # del

        # temporary
        network_dict = {"network": d["network"]}
        d["wfpt_nn"] = hddm.likelihoods_mlp.make_mlp_likelihood(
            model=d["model"],
            model_config=d["model_config"],
            wiener_params=d["wiener_params"],
            **network_dict
        )

        super(HDDMnn, self).__setstate__(d)
