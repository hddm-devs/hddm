import pandas as pd
import numpy as np
from scipy.stats import truncnorm
from patsy import dmatrix
from collections import OrderedDict
from hddm.simulators.basic_simulator import *
from hddm.model_config import model_config
from functools import partial


# Helper
def hddm_preprocess(
    simulator_data=None,
    subj_id="none",
    keep_negative_responses=False,
    add_model_parameters=False,
    keep_subj_idx=True,
):
    """Takes simulator data and turns it into HDDM ready format.

    :Arguments:
        simulator_data: tuple
            Output of e.g. the hddm.simulators.basic_simulator function.
        subj_id: str <default='none'>
            Subject id to attach to returned dataset
        keep_negative_responses: bool <default=False>
            Whether or not to turn negative responses into 0
        add_model_parameters: bool <default=False>
            Whether or not to add trial by trial model parameters to returned dataset
        keep_subj_idx: bool <default=True>
            Whether to keep subject id in the returned dataset

    """
    # Define dataframe if simulator output is normal (comes out as list tuple [rts, choices, metadata])
    if len(simulator_data) == 3:
        df = pd.DataFrame(simulator_data[0].astype(np.double), columns=["rt"])
        df["response"] = simulator_data[1].astype(int)

    if not keep_negative_responses:
        df.loc[df["response"] == -1.0, "response"] = 0.0
    if keep_subj_idx:
        df["subj_idx"] = subj_id

    # Add ground truth parameters to dataframe
    if add_model_parameters:
        for param in model_config[simulator_data[2]["model"]]["params"]:
            if len(simulator_data[2][param]) > 1:
                df[param] = simulator_data[2][param]
            else:
                # print(param)
                # print(simulator_data[2][param][0])
                df[param] = simulator_data[2][param][0]
    return df


def _add_outliers(
    sim_out=None,
    p_outlier=None,  # AF-comment: Redundant argument, can compute from sim_out !
    max_rt_outlier=10.0,
):
    """Add outliers to simulated data

    :Arguments:
        sim_out: tuple <default=None>
            Output of hddm.simulators.basic_simulator
        p_outlier: float <default=None>
            Probability of outliers
        max_rt_outlier: float
            Maximum reaction time that an outlier can take

    :Return:
        sim_out data with the appropriate number of samples exchanged by the samples
        from the outlier distribution.
    """

    if p_outlier == 0:
        return sim_out
    else:
        # Sample number of outliers from appropriate binomial
        n_outliers = np.random.binomial(n=sim_out[0].shape[0], p=p_outlier)

        # Only if the sampled number of outliers is above 0,
        # do we bother generating and storing them
        if n_outliers > 0:
            # Initialize the outlier data
            outlier_data = np.zeros((n_outliers, 2))

            # Generate outliers
            # Reaction times are uniform between 0 and 1/max_rt_outlier (default 1 / 0.1)
            # Choice are random with equal probability among the valid choice options
            outlier_data[:, 0] = np.random.uniform(
                low=0.0, high=max_rt_outlier, size=n_outliers
            )
            outlier_data[:, 1] = np.random.choice(
                sim_out[2]["possible_choices"], size=n_outliers
            )

            # Exchange the last parts of the simulator data for the outliers
            sim_out[0][-n_outliers:, 0] = outlier_data[:, 0]
            sim_out[1][-n_outliers:, 0] = outlier_data[:, 1]
    return sim_out


# -------------------------------------------------------------------------------------
# Parameter set generator
def make_parameter_vectors_nn(model="angle", param_dict=None, n_parameter_vectors=10):
    """Generates a (number of) parameter vector(s) for a given model.

    :Arguments:

        model: str <default='angle'>
            String that specifies the model to be simulated.
            Current options include, 'angle', 'ornstein', 'levy', 'weibull', 'full_ddm'
        param_dict: dict <default=None>
            Dictionary of parameter values that you would like to pre-specify. The dictionary takes the form (for the simple examples of the ddm),
            {'v': [0], 'a': [1.5]} etc.. For a given key supply either a list of length 1, or a list of
            length equal to the n_parameter_vectors argument supplied.
        n_parameter_vectors: int <default=10>
            Nuber of parameter vectors you want to generate

    Return: pandas.DataFrame
            Columns are parameter names and rows fill the parameter values.
    """

    parameter_data = np.zeros((n_parameter_vectors, len(model_config[model]["params"])))

    if param_dict is not None:
        cnt = 0
        for param in model_config[model]["params"]:
            if param in param_dict.keys():
                if (len(param_dict[param]) == n_parameter_vectors) or (
                    len(param_dict[param]) == 1
                ):
                    # Check if parameters are properly in bounds
                    if (
                        np.sum(
                            np.array(param_dict[param])
                            < model_config[model]["param_bounds"][0][cnt]
                        )
                        > 0
                        or np.sum(
                            np.array(param_dict[param])
                            > model_config[model]["param_bounds"][1][cnt]
                        )
                        > 0
                    ):
                        print(
                            "The parameter: ",
                            param,
                            ", is out of the accepted bounds [",
                            model_config[model]["param_bounds"][0][cnt],
                            ",",
                            model_config[model]["param_bounds"][1][cnt],
                            "]",
                        )
                        return
                    else:
                        parameter_data[:, cnt] = param_dict[param]
                else:
                    print(
                        "Param dict not specified correctly. Lengths of parameter lists needs to be 1 or equal to n_param_sets"
                    )

            else:
                parameter_data[:, cnt] = np.random.uniform(
                    low=model_config[model]["param_bounds"][0][cnt],
                    high=model_config[model]["param_bounds"][1][cnt],
                    size=n_parameter_vectors,
                )
            cnt += 1
    else:
        parameter_data = np.random.uniform(
            low=model_config[model]["param_bounds"][0],
            high=model_config[model]["param_bounds"][1],
            size=(n_parameter_vectors, len(model_config[model]["params"])),
        )

    return pd.DataFrame(parameter_data, columns=model_config[model]["params"])


# Dataset generators
def simulator_single_subject(
    parameters=(0, 0, 0),
    p_outlier=0.0,
    max_rt_outlier=10.0,
    model="angle",
    n_samples=1000,
    delta_t=0.001,
    max_t=20,
    bin_dim=None,
    bin_pointwise=False,
    verbose=0,
):
    """Generate a hddm-ready dataset from a single set of parameters

    :Arguments:
        parameters: dict, list or numpy array
            Model parameters with which to simulate. Dict is preferable for informative error messages.
            If you know the order of parameters for your model of choice, you can also directly supply a
            list or nump.array which needs to have the parameters in the correct order.
        p_outlier: float between 0 and 1 <default=0>
            Probability of generating outlier datapoints. An outlier is defined
            as a random choice from a uniform RT distribution
        max_rt_outlier: float > 0 <default=10.0>
            Using max_rt_outlier (which is commonly defined for hddm models) here as an imlicit maximum
            on the RT of outliers. Outlier RTs are sampled uniformly from [0, max_rt_outlier]
        model: str <default='angle'>
            String that specifies the model to be simulated.
            Current options include, 'angle', 'ornstein', 'levy', 'weibull', 'full_ddm'
        n_samples: int <default=1000>
            Number of samples to simulate.
        delta_t: float <default=0.001>
            Size fo timesteps in simulator (conceptually measured in seconds)
        max_t: float <default=20>
            Maximum reaction the simulator can reach
        bin_dim: int <default=None>
            If simulator output should be binned, this specifies the number of bins to use
        bin_pointwise: bool <default=False>
            Determines whether to bin simulator output pointwise. Pointwise here is in contrast to producing binned output
            in the form of a histogram. Binning pointwise gives each trial's RT and index which is the respective bin-number.
            This is expected when you are using the 'cnn' network to fit the dataset later. If pointwise is not chosen,
            then the takes the form of a histogram, with bin-wise frequencies.

    Return: tuple of (pandas.DataFrame, dict, list)
        The first part of the tuple holds a DataFrame with a 'reaction time' column and a 'response' column. Ready to be fit with hddm.
        The second part of the tuple hold a dict with parameter names as keys and parameter values as values.
        The third part gives back the parameters supplied in array form.
        This return is consistent with the returned objects in other data generators under hddm.simulators
    """

    # Sanity checks
    assert p_outlier >= 0 and p_outlier <= 1, "p_outlier is not between 0 and 1"
    assert max_rt_outlier > 0, "max_rt__outlier needs to be > 0"

    if verbose:
        print("Model: ", model)
        print("Parameters needed: ", model_config[model]["params"])

    if parameters is None:
        print("Proposing parameters and checking if in bounds")
        params_ok = 0
        while not params_ok:
            parameters = np.random.normal(
                loc=model_config[model]["param_bounds"][0]
                + (
                    (1 / 2)
                    * (
                        model_config[model]["param_bounds"][1]
                        - model_config[model]["param_bounds"][0]
                    )
                ),
                scale=(
                    (1 / 4)
                    * (
                        model_config[model]["param_bounds"][1]
                        - model_config[model]["param_bounds"][0]
                    )
                ),
                size=1,
            )
            if not bool(
                int(
                    np.sum(
                        parameters < np.array(model_config[model]["param_bounds"][0])
                    )
                    + np.sum(
                        parameters > np.array(model_config[model]["param_bounds"][1])
                    )
                )
            ):
                params_ok = 1

        gt = {}
        for param in model_config[model]["params"]:
            id_tmp = model_config[model]["params"].index(param)
            gt[param] = parameters[id_tmp]

    elif type(parameters) == list or type(parameters) == np.ndarray:
        gt = {}
        for param in model_config[model]["params"]:
            id_tmp = model_config[model]["params"].index(param)
            gt[param] = parameters[id_tmp]

    elif type(parameters) == dict:
        gt = parameters.copy()

        # Get max shape of parameter (in case it is supplied as part length-n vector, part length-1 vector)
        tmp_max = 0
        for key_ in gt.keys():
            tmp_ = len(gt[key_])
            if tmp_ > tmp_max:
                tmp_max = tmp_

        parameters = np.zeros((tmp_max, len(model_config[model]["params"])))

        for param in model_config[model]["params"]:
            idx = model_config[model]["params"].index(param)
            if param in gt.keys():
                parameters[:, idx] = gt[param]
            else:
                print("The parameter ", param, " was not supplied to the function.")
                print(
                    "Taking default ",
                    param,
                    " from hddm.model_config as",
                    model_config.model_config[model]["params_default"],
                )
                parameters[:, idx] = model_config[model]["params_default"][idx]
    else:
        return "parameters argument is not of type list, np.ndarray, dict"

    if verbose:
        print(parameters)

    x = simulator(
        theta=parameters,
        model=model,
        n_samples=n_samples,
        delta_t=delta_t,
        max_t=max_t,
        bin_dim=bin_dim,
        bin_pointwise=bin_pointwise,
    )

    # Add outliers
    # (Potentially 0 outliers)
    x = _add_outliers(
        sim_out=x,
        p_outlier=p_outlier,
        max_rt_outlier=max_rt_outlier,
    )

    data_out = hddm_preprocess(x, add_model_parameters=True)

    return (data_out, gt)


def simulator_stimcoding(
    model="angle",
    split_by="v",
    p_outlier=0.0,
    max_rt_outlier=10.0,
    drift_criterion=0.0,
    n_trials_per_condition=1000,
    delta_t=0.001,
    prespecified_params={},
    bin_pointwise=False,
    bin_dim=None,
    max_t=20.0,
):
    """Generate a dataset as expected by Hddmstimcoding. Essentially it is a specific way to parameterize two condition data.

    :Arguments:
        parameters: list or numpy array
            Model parameters with which to simulate.
        model: str <default='angle'>
            String that specifies the model to be simulated.
            Current options include, 'angle', 'ornstein', 'levy', 'weibull', 'full_ddm'
        split_by: str <default='v'>
            You can split by 'v' or 'z'. If splitting by 'v' one condition's v_0 = drift_criterion + 'v', the other
            condition's v_1 = drift_criterion - 'v'.
            Respectively for 'z', 'z_0' = 'z' and 'z_1' = 1 - 'z'.
        p_outlier: float between 0 and 1 <default=0>
            Probability of generating outlier datapoints. An outlier is defined
            as a random choice from a uniform RT distribution
        max_rt_outlier: float > 0 <default=10.0>
            Using max_rt_outlier (which is commonly defined for hddm models) here as an imlicit maximum
            on the RT of outliers. Outlier RTs are sampled uniformly from [0, max_rt_outlier]
        drift_criterion: float <default=0.0>
            Parameter that can be treated as the 'bias part' of the slope, in case we split_by 'v'.
        n_trials_per_condition: int <default=1000>
            Number of samples to simulate per condition (here 2 condition by design).
        delta_t: float <default=0.001>
            Size fo timesteps in simulator (conceptually measured in seconds)
        prespecified_params: dict <default = {}>
            A dictionary with parameter names keys. Values are list of either length 1, or length equal to the number of conditions (here 2).
        max_t: float <default=20>
            Maximum reaction the simulator can reach
        bin_dim: int <default=None>
            If simulator output should be binned, this specifies the number of bins to use
        bin_pointwise: bool <default=False>
            Determines whether to bin simulator output pointwise. Pointwise here is in contrast to producing binned output
            in the form of a histogram. Binning pointwise gives each trial's RT and index which is the respective bin-number.
            This is expected when you are using the 'cnn' network to fit the dataset later. If pointwise is not chosen,
            then the takes the form of a histogram, with bin-wise frequencies.

    Return: pandas.DataFrame holding a 'reaction time' column and a 'response' column. Ready to be fit with hddm.
    """

    param_base = np.tile(
        np.random.uniform(
            low=model_config[model]["param_bounds"][0],
            high=model_config[model]["param_bounds"][1],
            size=(1, len(model_config[model]["params"])),
        ),
        (2, 1),
    )

    # Fill in prespecified parameters if supplied
    if prespecified_params is not None:
        if type(prespecified_params) == dict:
            for param in prespecified_params:
                id_tmp = model_config[model]["params"].index(param)
                param_base[:, id_tmp] = prespecified_params[param]
        else:
            print(
                "prespecified_params is not supplied as a dictionary, please reformat the input"
            )
            return

    if type(split_by) == list:
        pass
    elif type(split_by) == str:
        split_by = [split_by]
    else:
        print(
            "Can not recognize data-type of argument: split_by, provided neither a list nor a string"
        )
        return

    gt = {}
    for i in range(len(model_config[model]["params"])):
        gt[model_config[model]["params"][i]] = param_base[0, i]

    for i in range(2):
        if i == 0:
            if "v" in split_by:
                id_tmp = model_config[model]["params"].index("v")
                param_base[i, id_tmp] = drift_criterion - param_base[i, id_tmp]
                gt["dc"] = drift_criterion

        if i == 1:
            if "v" in split_by:
                id_tmp = model_config[model]["params"].index("v")
                param_base[i, id_tmp] = drift_criterion + param_base[i, id_tmp]
            if "z" in split_by:
                id_tmp = model_config[model]["params"].index("z")
                param_base[i, id_tmp] = 1 - param_base[i, id_tmp]

    dataframes = []
    for i in range(2):
        sim_out = simulator(
            theta = param_base[i, :],
            model=model,
            n_samples=n_trials_per_condition,
            bin_dim=bin_dim,
            bin_pointwise=bin_pointwise,
            max_t=max_t,
            delta_t=delta_t,
        )

        sim_out = _add_outliers(
            sim_out=sim_out,
            p_outlier=p_outlier,
            max_rt_outlier=max_rt_outlier,
        )

        dataframes.append(
            hddm_preprocess(
                simulator_data=sim_out, subj_id=i + 1, add_model_parameters=True
            )
        )

    data_out = pd.concat(dataframes)
    data_out = data_out.rename(columns={"subj_idx": "stim"})
    data_out["subj_idx"] = "none"
    data_out = data_out.reset_index(drop=True)
    return (data_out, gt)


def simulator_h_c(
    data=None,
    n_subjects=10,
    n_trials_per_subject=100,
    model="ddm_hddm_base",
    conditions=None,
    depends_on=None,
    regression_models=None,
    regression_covariates=None,  # need this to make initial covariate matrix from which to use dmatrix (patsy)
    group_only_regressors=True,
    group_only=["z"],
    fixed_at_default=None,
    p_outlier=0.0,
    outlier_max_t=10.0,
    **kwargs,
):
    """Flexible simulator that allows specification of models very similar to the hddm model classes. Has two major modes. When data \n
    is supplied the function generates synthetic versions of the provided data. If no data is provided, you can supply
    a varied of options to create complicated synthetic datasets from scratch.

    :Arguments:
        data: pd.DataFrame <default=None>
            Actual covariate dataset. If data is supplied its covariates are used instead of generated.
        n_subjects: int <default=5>
            Number of subjects in the datasets
        n_trials_per_subject: int <default=500>
            Number of trials for each subject
        model: str <default = 'ddm_hddm_base'>
            Model to sample from. For traditional hddm supported models, append '_hddm_base' to the model. Omitting 'hddm_base'
            imposes constraints on the parameter sets to not violate the trained parameter space of our LANs.
        conditions: dict <default=None>
            Keys represent condition relevant columns, and values are lists of unique items for each condition relevant column.
            Example: {"c_one": ["high", "low"], "c_two": ["high", "low"], "c_three": ["high", "medium", "low"]}
        depends_on: dict <default=None>
            Keys specify model parameters that depend on the values --> lists of condition relevant columns.
            Follows the syntax in the HDDM model classes. Example: {"v": ["c_one", "c_two"]}
        regression_models: list or strings <default=None>
            Specify regression model formulas for one or more dependent parameters in a list.
            Follows syntax of HDDM model classes.
            Example: ["z ~ covariate_name"]
        regression_covariates: dict <default={'covariate_name': {'type': 'categorical', 'range': (0, 4)}}>
            Dictionary in dictionary. Specify the name of the covariate column as keys, and for each key supply the 'type' (categorical, continuous) and
            'range' ((lower bound, upper bound)) of the covariate.
            Example: {"covariate_name": {"type": "categorical", "range": (0, 4)}}
        group_only_regressors: bin <default=True>
            Should regressors only be specified at the group level? If true then only intercepts are specified subject wise.
            Other covariates act globally.
        group_only: list <default = ['z']>
            List of parameters that are specified only at the group level.
        fixed_at_default: list <default=None>
            List of parameters for which defaults are to be used.
            These defaults are specified in the model_config dictionary,
            which you can access via: hddm.simulators.model_config.
            Example: ['t']
        p_outlier: float <default = 0.0>
            Specifies the proportion of outliers in the data.
        outlier_max_t: float <default = 10.0>
            Outliers are generated from np.random.uniform(low = 0, high = outlier_max_t) with random choices.
    Returns:
        (pandas.DataFrame, dict): The Dataframe holds the generated dataset, ready for constuction of an hddm model. The dictionary holds the groundtruth parameter (values) and parameter names (keys). Keys match
                                  the names of traces when fitting the equivalent hddm model. The parameter dictionary is useful for some graphs, otherwise not neccessary.
    """

    # print('starting data generation')
    meta_params = {
        "group_param_dist": "normal",
        "gen_norm_std": 1 / 3,
        "uniform_buffer": 1 / 5,
        "gen_std_std": 1 / 8,
        "covariate_range": 1 / 4,
    }

    for key_ in kwargs.keys():
        meta_params[key_] = kwargs[key_]

    if depends_on is None:
        depends_on = {}

    def check_params(data=None, model=None, is_nn=True):
        """
        Function checks if parameters are within legal bounds
        """
        for key in data.keys():
            if key in model_config[model]["params"]:
                if (
                    np.sum(
                        data[key]
                        < model_config[model]["param_bounds"][0][
                            model_config[model]["params"].index(key)
                        ]
                    )
                    > 0
                ):
                    return 0
                elif (
                    np.sum(
                        data[key]
                        > model_config[model]["param_bounds"][1][
                            model_config[model]["params"].index(key)
                        ]
                    )
                    > 0
                ):
                    return 0
        return 1

    def get_parameter_remainder(
        regression_models=None, group_only=None, depends_on=None, fixed_at_default=None
    ):
        """
        The arguments supplied to the simulator implicitly specify how we should handle a bunch of model parameters.
        If there remain model parameters that did not receive implicit instructions, we call these 'remainder' parameters
        and sample them randomly for our simulations.
        """

        # Add subject parameters to full_parameter_dict
        total_param_list = model_config[model]["params"]
        params_utilized = []

        # Regression Part
        # reg_df = make_covariate_df(regression_covariates, n_trials_per_subject)
        if regression_models is not None:
            for regression_model in regression_models:
                separator = regression_model.find("~")
                assert separator != -1, "No outcome variable specified."
                params_utilized += regression_model[:separator].strip(" ")

        # Group only Part
        if group_only is not None:
            params_utilized += group_only

        # Fixed Part
        if fixed_at_default is not None:
            params_utilized += fixed_at_default

        # Depends on Part
        if depends_on is not None:
            for depends_on_key in depends_on.keys():
                params_utilized += [depends_on_key]

        params_utilized = list(set(params_utilized))

        # Rest of Params
        remainder = set(total_param_list) - set(params_utilized)

        return remainder

    def make_covariate_df(regression_covariates, n_trials_per_subject):
        """
        Goes through the supplied covariate data, and turns it into a dataframe, with randomly generated covariate values.
        Each column refers to one covariate.
        """

        cov_df = pd.DataFrame(
            np.zeros((n_trials_per_subject, len(list(regression_covariates.keys())))),
            columns=[key for key in regression_covariates.keys()],
        )

        for covariate in regression_covariates.keys():
            tmp = regression_covariates[covariate]
            if tmp["type"] == "categorical":
                cov_df[covariate] = np.random.choice(
                    np.arange(tmp["range"][0], tmp["range"][1] + 1, 1),
                    replace=True,
                    size=n_trials_per_subject,
                ) / (tmp["range"][1])
            else:
                cov_df[covariate] = np.random.uniform(
                    low=tmp["range"][0], high=tmp["range"][1], size=n_trials_per_subject
                ) / (tmp["range"][1] - tmp["range"][0])

        return cov_df

    def make_conditions_df(conditions=None):
        """
        Makes a dataframe out of the supplied condition dictionary, that stores each combination as a row.
        """
        arg_tuple = tuple([conditions[key] for key in conditions.keys()])
        condition_rows = np.meshgrid(*arg_tuple)
        return pd.DataFrame(
            np.column_stack([x_tmp.flatten() for x_tmp in condition_rows]),
            columns=[key for key in conditions.keys()],
        )

    def make_single_sub_cond_df_gen(
        conditions_df,
        depends_on,
        regression_models,
        regression_covariates,
        group_only_regressors,
        group_only,
        fixed_at_default,
        remainder,
        model,
        group_level_parameter_dict,
        n_subjects,
        n_trials_per_subject,
    ):
        # Construct subject data
        full_parameter_dict = group_level_parameter_dict.copy()

        # Subject part -----------------------
        full_data = []
        # Condition --------------------------
        if conditions_df is None:
            n_conditions = 1
        else:
            n_conditions = conditions_df.shape[0]

        for condition_id in range(n_conditions):
            # remainder_set = 0
            regressor_set = 0

            for subj_idx in range(n_subjects):
                # Parameter vector
                subj_data = pd.DataFrame(index=np.arange(0, n_trials_per_subject, 1))
                subj_data["subj_idx"] = str(subj_idx)

                # Fixed part
                if fixed_at_default is not None:
                    for fixed_tmp in fixed_at_default:
                        subj_data[fixed_tmp] = group_level_parameter_dict[fixed_tmp]

                # Group only part
                if group_only is not None:
                    for group_only_tmp in group_only:
                        if group_only_tmp in list(depends_on.keys()):
                            pass
                        else:
                            subj_data[group_only_tmp] = group_level_parameter_dict[
                                group_only_tmp
                            ]

                # Remainder part
                if remainder is not None:
                    for remainder_tmp in remainder:
                        tmp_mean = group_level_parameter_dict[remainder_tmp]
                        tmp_std = group_level_parameter_dict[remainder_tmp + "_std"]

                        # If the subject has been seen before, we use the parameters from
                        # the previous condition, since the remainder parameters do not change
                        # across conditions
                        if (
                            remainder_tmp + "_subj." + str(subj_idx)
                            in full_parameter_dict.keys()
                        ):
                            pass
                        else:
                            # Otherwise, generate new parameter for this subject (really only relevant first condition for remainder parameters)
                            full_parameter_dict[
                                remainder_tmp + "_subj." + str(subj_idx)
                            ] = np.random.normal(loc=tmp_mean, scale=tmp_std)

                        subj_data[remainder_tmp] = full_parameter_dict[
                            remainder_tmp + "_subj." + str(subj_idx)
                        ]

                # Depends on part
                if depends_on is not None:
                    # conditions_tmp = conditions_df.iloc[condition_id]
                    for depends_tmp in depends_on.keys():
                        conditions_df_tmp = conditions_df[depends_on[depends_tmp]].iloc[
                            condition_id
                        ]
                        condition_elem = ".".join(conditions_df_tmp)

                        # Add parameters to subject dataframe
                        if depends_tmp not in group_only:
                            tmp_mean = group_level_parameter_dict[
                                depends_tmp + "(" + condition_elem + ")"
                            ]
                            tmp_std = group_level_parameter_dict[depends_tmp + "_std"]
                            tmp_param_name = (
                                depends_tmp
                                + "_subj("
                                + condition_elem
                                + ")."
                                + str(subj_idx)
                            )

                            # If the subject / condition combination has been see before
                            # we do not reassign a new parameter here !

                            if tmp_param_name in full_parameter_dict.keys():
                                pass
                            else:  # Otherwise assign new parameter
                                full_parameter_dict[tmp_param_name] = np.random.normal(
                                    loc=tmp_mean, scale=tmp_std
                                )

                            # Assign the parameter to subject data
                            subj_data[depends_tmp] = full_parameter_dict[
                                depends_tmp
                                + "_subj("
                                + condition_elem
                                + ")."
                                + str(subj_idx)
                            ]
                        else:
                            subj_data[depends_tmp] = full_parameter_dict[
                                depends_tmp + "(" + condition_elem + ")"
                            ]

                        # Add the respective stimulus columns
                        for condition_key_tmp in conditions_df_tmp.keys():
                            subj_data[condition_key_tmp] = conditions_df_tmp[
                                condition_key_tmp
                            ]

                # Regressor part
                if regression_covariates is not None:
                    cov_df = make_covariate_df(
                        regression_covariates, n_trials_per_subject
                    )

                    # Add cov_df to subject data
                    for key_tmp in cov_df.keys():
                        subj_data[key_tmp] = cov_df[key_tmp].copy()

                if regression_models is not None:
                    for reg_model in regression_models:
                        # Make Design Matrix
                        separator = reg_model.find("~")
                        outcome = reg_model[:separator].strip(" ")
                        reg_model_stripped = reg_model[(separator + 1) :]
                        design_matrix = dmatrix(reg_model_stripped, cov_df)

                        reg_params_tmp = []
                        reg_param_names_tmp = []
                        for reg_param_key in group_level_parameter_dict[
                            outcome + "_reg"
                        ].keys():
                            if (
                                group_only_regressors and "Intercept" in reg_param_key
                            ) or (not group_only_regressors):
                                reg_params_tmp.append(
                                    np.random.normal(
                                        loc=group_level_parameter_dict[
                                            outcome + "_reg"
                                        ][reg_param_key],
                                        scale=group_level_parameter_dict[
                                            outcome + "_reg_std"
                                        ][reg_param_key + "_std"],
                                    )
                                )

                                reg_param_names_tmp.append(
                                    reg_param_key + "_subj." + str(subj_idx)
                                )
                            else:
                                reg_params_tmp.append(
                                    group_level_parameter_dict[outcome + "_reg"][
                                        reg_param_key
                                    ]
                                )
                                reg_param_names_tmp.append(reg_param_key)

                        reg_params_tmp = np.array(reg_params_tmp)

                        for key in group_level_parameter_dict[outcome + "_reg"].keys():
                            full_parameter_dict[key] = group_level_parameter_dict[
                                outcome + "_reg"
                            ][key]
                        for key in group_level_parameter_dict[
                            outcome + "_reg_std"
                        ].keys():
                            full_parameter_dict[key] = group_level_parameter_dict[
                                outcome + "_reg_std"
                            ][key]

                        if not regressor_set:
                            for k in range(len(reg_param_names_tmp)):
                                full_parameter_dict[
                                    reg_param_names_tmp[k]
                                ] = reg_params_tmp[k]

                        subj_data[outcome] = (design_matrix * reg_params_tmp).sum(
                            axis=1
                        )  # AF-TD: This should probably include a noise term here (parameter really defined as coming from a linear model + noise)

                # Append full data:
                full_data.append(subj_data.copy())

            remainder_set = 1
            regressor_set = 1

        full_data = pd.concat(full_data)
        parameters = full_data[model_config[model]["params"]]

        # Run the actual simulations
        # print(parameters)

        sim_data = simulator(
            theta=parameters.values,
            model=model,
            n_samples=1,
            delta_t=0.001,
            max_t=20,
            no_noise=False,
            bin_dim=None,
            bin_pointwise=False,
        )

        # Post-processing
        full_data["rt"] = sim_data[0].astype(np.float64)
        full_data["response"] = sim_data[1].astype(np.float64)
        full_data.loc[full_data["response"] < 0, ["response"]] = 0.0

        # Add in outliers
        if p_outlier > 0:
            # print('passing through outlier creation')
            outlier_idx = np.random.choice(
                list(data.index),
                replace=False,
                size=int(p_outlier * len(list(data.index))),
            )
            outlier_data = np.zeros((outlier_idx.shape[0], 2))

            # Outlier rts
            outlier_data[:, 0] = np.random.uniform(
                low=0.0, high=outlier_max_t, size=outlier_data.shape[0]
            )

            # Outlier choices
            outlier_data[:, 1] = np.random.choice(
                sim_data[2]["possible_choices"], size=outlier_data.shape[0]
            )

            # Exchange data for outliers
            full_data.iloc[
                outlier_idx,
                [
                    list(full_data.keys()).index("rt"),
                    list(full_data.keys()).index("response"),
                ],
            ] = outlier_data

            # Identify outliers in dataframe
            full_data["outlier"] = 0
            full_data[outlier_idx, [list(full_data.keys()).index("outlier")]] = 1

        full_data_cols = ["rt", "response", "subj_idx"]

        if regression_covariates is not None:
            full_data_cols += [key for key in regression_covariates.keys()]
        if conditions is not None:
            full_data_cols += [key for key in conditions.keys()]

        full_data_cols += model_config[model]["params"]
        full_data = full_data[full_data_cols]
        full_data.reset_index(drop=True, inplace=True)

        # AF-Comment: Does this cover all corner cases?
        # If n_subjects is 1 --> we overwrite the group parameters with the subj.0 parameters
        if n_subjects == 1:
            new_param_dict = {}
            for key, value in full_parameter_dict.items():
                if "subj" in key:
                    new_key = key
                    new_key = new_key.replace("_subj", "")
                    new_key = new_key[: new_key.find(".")]
                    new_param_dict[new_key] = value
                elif "_std" in key:
                    pass
                else:
                    new_param_dict[key] = value
            full_parameter_dict = new_param_dict

        return full_data, full_parameter_dict

    def make_single_sub_cond_df_from_gt(
        data,
        conditions_df,
        depends_on,
        regression_models,
        regression_covariates,
        group_only_regressors,
        group_only,
        fixed_at_default,
        remainder,
        model,
        group_level_parameter_dict,
    ):
        # Construct subject data
        full_parameter_dict = group_level_parameter_dict.copy()

        # Initialize parameter columns in data
        for param in model_config[model]["params"]:
            data[param] = 0

        for subj_idx in data["subj_idx"].unique():
            # Fixed part
            if fixed_at_default is not None:
                for fixed_tmp in fixed_at_default:
                    data.loc[
                        data["subj_idx"] == int(subj_idx), [fixed_tmp]
                    ] = group_level_parameter_dict[fixed_tmp]

            # Group only part
            if group_only is not None:
                for group_only_tmp in group_only:
                    if group_only_tmp in list(depends_on.keys()):
                        pass
                    else:
                        data.loc[
                            data["subj_idx"] == int(subj_idx), [group_only_tmp]
                        ] = group_level_parameter_dict[group_only_tmp]

            # Remainder part
            if remainder is not None:
                for remainder_tmp in remainder:
                    tmp_mean = group_level_parameter_dict[remainder_tmp]
                    tmp_std = group_level_parameter_dict[remainder_tmp + "_std"]
                    full_parameter_dict[
                        remainder_tmp + "_subj." + str(subj_idx)
                    ] = np.random.normal(loc=tmp_mean, scale=tmp_std)

                    data.loc[
                        data["subj_idx"] == int(subj_idx), [remainder_tmp]
                    ] = full_parameter_dict[remainder_tmp + "_subj." + str(subj_idx)]

            # Depends on part
            if depends_on is not None:
                # Go through depends_on variables:
                for depends_tmp in depends_on.keys():
                    conditions_df_tmp = conditions_df[
                        depends_on[depends_tmp]
                    ].drop_duplicates()

                    for condition_id in range(conditions_df_tmp.shape[0]):
                        condition_elem = ".".join(conditions_df_tmp.iloc[condition_id])
                        bool_ = data["subj_idx"] == int(subj_idx)

                        for key_ in conditions_df_tmp.keys():
                            bool_ = (bool_) & (
                                data[key_].astype(str)
                                == conditions_df_tmp.iloc[condition_id][key_]
                            )

                        # Check if there is data which adheres to the condition currently active
                        # Otherwise there is nothing to update
                        # AF COMMENT: This check should already be applied at the point of generating the condition_df dataframe
                        if np.sum(bool_) > 0:
                            if depends_tmp not in group_only:
                                tmp_mean = group_level_parameter_dict[
                                    depends_tmp + "(" + condition_elem + ")"
                                ]
                                tmp_std = group_level_parameter_dict[
                                    depends_tmp + "_std"
                                ]

                                full_parameter_dict[
                                    depends_tmp
                                    + "_subj("
                                    + condition_elem
                                    + ")."
                                    + str(subj_idx)
                                ] = np.random.normal(loc=tmp_mean, scale=tmp_std)

                                data.loc[bool_, depends_tmp] = full_parameter_dict[
                                    depends_tmp
                                    + "_subj("
                                    + condition_elem
                                    + ")."
                                    + str(subj_idx)
                                ]

                            else:
                                # print('passed here (group_only) with depends_tmp: ', depends_tmp)
                                data.loc[bool_, depends_tmp] = full_parameter_dict[
                                    depends_tmp + "(" + condition_elem + ")"
                                ]

            # Regressor part
            if regression_models is not None:
                for reg_model in regression_models:
                    # Make Design Matrix
                    separator = reg_model.find("~")
                    outcome = reg_model[:separator].strip(" ")
                    reg_model_stripped = reg_model[(separator + 1) :]
                    design_matrix = dmatrix(
                        reg_model_stripped,
                        data.loc[data["subj_idx"] == int(subj_idx), :],
                    )

                    reg_params_tmp = []
                    reg_param_names_tmp = []
                    for reg_param_key in group_level_parameter_dict[
                        outcome + "_reg"
                    ].keys():
                        if (group_only_regressors and "Intercept" in reg_param_key) or (
                            not group_only_regressors
                        ):
                            reg_params_tmp.append(
                                np.random.normal(
                                    loc=group_level_parameter_dict[outcome + "_reg"][
                                        reg_param_key
                                    ],
                                    scale=group_level_parameter_dict[
                                        outcome + "_reg_std"
                                    ][reg_param_key + "_std"],
                                )
                            )

                            reg_param_names_tmp.append(
                                reg_param_key + "_subj." + str(subj_idx)
                            )
                        else:
                            reg_params_tmp.append(
                                group_level_parameter_dict[outcome + "_reg"][
                                    reg_param_key
                                ]
                            )
                            reg_param_names_tmp.append(reg_param_key)

                    reg_params_tmp = np.array(reg_params_tmp)

                    for key in group_level_parameter_dict[outcome + "_reg"].keys():
                        full_parameter_dict[key] = group_level_parameter_dict[
                            outcome + "_reg"
                        ][key]

                    for key in group_level_parameter_dict[outcome + "_reg_std"].keys():
                        full_parameter_dict[key] = group_level_parameter_dict[
                            outcome + "_reg_std"
                        ][key]

                    for k in range(len(reg_param_names_tmp)):
                        full_parameter_dict[reg_param_names_tmp[k]] = reg_params_tmp[k]

                    data.loc[data["subj_idx"] == int(subj_idx), [outcome]] = (
                        design_matrix * reg_params_tmp
                    ).sum(
                        axis=1
                    )  # AF-TD: This should probably include a noise term here (parameter really defined as coming from a linear model + noise)

        parameters = data[model_config[model]["params"]]

        sim_data = simulator(
            theta=parameters.values,
            model=model,
            n_samples=1,
            delta_t=0.001,
            max_t=20,
            no_noise=False,
            bin_dim=None,
            bin_pointwise=False,
        )

        # Post-processing
        data["rt"] = sim_data[0].astype(np.float64)
        data["response"] = sim_data[1].astype(np.float64)
        data.loc[data["response"] < 0, ["response"]] = 0.0

        # Add in outliers
        if p_outlier > 0:
            outlier_idx = np.random.choice(
                list(data.index),
                replace=False,
                size=int(p_outlier * len(list(data.index))),
            )
            outlier_data = np.zeros((outlier_idx.shape[0], 2))

            # Outlier rts
            outlier_data[:, 0] = np.random.uniform(
                low=0.0, high=outlier_max_t, size=outlier_data.shape[0]
            )

            # Outlier choices
            outlier_data[:, 1] = np.random.choice(
                sim_data[2]["possible_choices"], size=outlier_data.shape[0]
            )

            # Exchange data for outliers
            data.loc[outlier_idx, ["rt", "response"]] = outlier_data

            # Identify outliers in dataframe
            data["outlier"] = 0
            data.loc[outlier_idx, [list(data.keys()).index("outlier")]] = 1

        # AF-Comment: Does this cover all corner cases?
        # If n_subjects is 1 --> we overwrite the group parameters with the subj.0 parameters
        if len(list(data["subj_idx"].unique())) == 1:
            new_param_dict = {}
            for key, value in full_parameter_dict.items():
                if "subj" in key:
                    new_key = key
                    new_key = new_key.replace("_subj", "")
                    new_key = new_key[: new_key.find(".")]
                    new_param_dict[new_key] = value
                elif "_std" in key:
                    pass
                else:
                    new_param_dict[key] = value
            full_parameter_dict = new_param_dict

        return data, full_parameter_dict

    def make_group_level_params(
        data,
        conditions_df,
        group_only,
        depends_on,
        model,
        fixed_at_default,
        remainder,
        group_only_regressors,
        regression_models,
        regression_covariates,
        group_param_dist="normal",
        gen_norm_std=1 / 4,
        uniform_buffer=1 / 5,
        gen_std_std=1 / 8,
        covariate_range=1
        / 4,  # multiplied by range of parameter bounds to give size of covariate
    ):
        """
        Make group level parameters from the information supplied.
        """

        # Some comments

        group_level_parameter_dict = {}

        # COLLECT PARAMETER WISE DATA AND ON CONSTRAINTS AND RV-GENERATORS ------
        param_gen_info = {}
        for param_name in model_config[model]["params"]:
            idx = model_config[model]["params"].index(param_name)

            param_gen_info[param_name] = {}
            # print(idx)
            # print(model_config[model]["param_bounds"])
            param_gen_info[param_name]["range"] = (
                model_config[model]["param_bounds"][1][idx]
                - model_config[model]["param_bounds"][0][idx]
            )

            param_gen_info[param_name]["mid"] = model_config[model]["param_bounds"][0][
                idx
            ] + (param_gen_info[param_name]["range"] / 2)
            param_gen_info[param_name]["gen_norm_std"] = gen_norm_std * (
                param_gen_info[param_name]["range"] / 2
            )
            param_gen_info[param_name]["uniform_buffer"] = uniform_buffer * (
                param_gen_info[param_name]["range"] / 2
            )
            param_gen_info[param_name]["std_gen_std"] = (
                gen_std_std * param_gen_info[param_name]["range"]
            )
            param_gen_info[param_name]["covariate_range"] = (
                covariate_range * param_gen_info[param_name]["range"]
            )

            if group_param_dist == "normal":
                param_gen_info[param_name]["rv"] = partial(
                    np.random.normal,
                    loc=param_gen_info[param_name]["mid"],
                    scale=param_gen_info[param_name]["gen_norm_std"],
                )
            elif group_param_dist == "uniform":
                param_gen_info[param_name]["rv"] = partial(
                    np.random.uniform,
                    low=model_config[model]["param_bounds"][0][param_name]
                    + param_gen_info[param_name]["uniform_buffer"],
                    high=model_config[model]["param_bounds"][1][param_name]
                    - param_gen_info[param_name]["uniform_buffer"],
                )

            param_gen_info[param_name]["std_rv"] = partial(
                np.random.uniform, low=0, high=param_gen_info[param_name]["std_gen_std"]
            )

            param_gen_info[param_name]["covariate_rv"] = partial(
                np.random.uniform,
                low=-param_gen_info[param_name]["covariate_range"],
                high=param_gen_info[param_name]["covariate_range"],
            )
        # -----------------------------------------------

        # Fixed part --------------------------------------------------------
        if fixed_at_default is not None:
            for fixed_tmp in fixed_at_default:
                group_level_parameter_dict[fixed_tmp] = model_config[model][
                    "params_default"
                ][model_config[model]["params"].index(fixed_tmp)]

        # Group only part (excluding depends on) ----------------------------
        if len(group_only) > 0:
            for group_only_tmp in group_only:
                if group_only_tmp in list(depends_on.keys()):
                    pass
                else:
                    group_level_parameter_dict[group_only_tmp] = param_gen_info[
                        group_only_tmp
                    ]["rv"]()

        # Remainder part -----------------------------------------------------
        if remainder is not None:
            for remainder_tmp in remainder:
                group_level_parameter_dict[remainder_tmp] = param_gen_info[
                    remainder_tmp
                ]["rv"]()
                group_level_parameter_dict[remainder_tmp + "_std"] = param_gen_info[
                    remainder_tmp
                ]["std_rv"]()

        # Depends on part ----------------------------------------------------
        if depends_on is not None:
            for depends_tmp in depends_on.keys():
                conditions_df_tmp = conditions_df[depends_on[depends_tmp]]

                # Get unique elements:
                unique_elems = []
                for i in range(conditions_df_tmp.shape[0]):
                    unique_elems.append(".".join(conditions_df_tmp.iloc[i]))
                unique_elems = np.unique(np.array(unique_elems))

                for unique_elem in unique_elems:
                    group_level_parameter_dict[
                        depends_tmp + "(" + unique_elem + ")"
                    ] = param_gen_info[depends_tmp]["rv"]()

                if depends_tmp not in group_only:
                    group_level_parameter_dict[depends_tmp + "_std"] = param_gen_info[
                        remainder_tmp
                    ]["std_rv"]()

        # Regressor part ------------------------------------------------------
        if regression_covariates is not None:
            # AF ADDED:
            # IF covariates supplied: skip generation
            if data is None:
                cov_df = make_covariate_df(regression_covariates, n_trials_per_subject)
            else:
                cov_df = data

        if regression_models is not None:
            for reg_model in regression_models:
                separator = reg_model.find("~")
                outcome = reg_model[:separator].strip(" ")
                reg_model_stripped = reg_model[(separator + 1) :]

                # Run through patsy dmatrix to get the covariate names
                # that patsy assigns !
                covariate_names = dmatrix(
                    reg_model_stripped, cov_df
                ).design_info.column_names

                reg_trace_dict = OrderedDict()
                reg_std_trace_dict = OrderedDict()

                for covariate in covariate_names:
                    if ("Intercept" in covariate) or (covariate == "1"):
                        # AF-COMMENT: Here instead of covariate_rv --> just use
                        # print(reg_trace_dict)
                        reg_trace_dict[outcome + "_" + covariate] = param_gen_info[
                            outcome
                        ]["rv"]()

                        # Intercept is always fit subject wise
                        reg_std_trace_dict[
                            outcome + "_" + covariate + "_" + "std"
                        ] = param_gen_info[outcome]["std_rv"]()

                    else:
                        reg_trace_dict[outcome + "_" + covariate] = param_gen_info[
                            outcome
                        ]["covariate_rv"]()

                        if not group_only_regressors:
                            reg_std_trace_dict[
                                outcome + "_" + covariate + "_" + "std"
                            ] = param_gen_info[outcome]["std_rv"]()

                group_level_parameter_dict[outcome + "_reg"] = reg_trace_dict.copy()

                # AF-COMMENT: Is this necessary ?
                # if not group_only_regressors:
                group_level_parameter_dict[
                    outcome + "_reg" + "_std"
                ] = reg_std_trace_dict.copy()

        return group_level_parameter_dict

    # MAIN PART OF THE FUNCTION -----------------------------------------------------------------

    # Some checks
    if group_only is None:
        group_only = []

    # Specify 'remainder' parameters --> will be sampled randomly from the allowed range
    remainder = get_parameter_remainder(
        regression_models=regression_models,
        group_only=group_only,
        depends_on=depends_on,
        fixed_at_default=fixed_at_default,
    )

    # Make conditions df
    if depends_on is not None:
        # print("depends_on is: ", depends_on)
        if type(depends_on) == dict:
            if len(list(depends_on.keys())) > 0:
                # If data is None then conditions were supplied as an argument
                if data is None:
                    conditions_df = make_conditions_df(conditions=conditions)
                else:  # Otherwise we have covariate data, so we can deduce conditions
                    conditions = dict()
                    for key_ in depends_on.keys():
                        for col in depends_on[key_]:
                            conditions[col] = np.sort(data[col].unique()).astype(str)
                    conditions_df = make_conditions_df(conditions=conditions)
            else:
                conditions_df = None
        else:
            conditions_df = None
    else:
        conditions_df = None

    params_ok_all = 0
    cnt = 0
    while params_ok_all == 0:
        if cnt > 0:
            print(
                "new round of data simulation because parameter bounds where violated"
            )

        group_level_param_dict = make_group_level_params(
            data=data,
            conditions_df=conditions_df,
            group_only=group_only,
            depends_on=depends_on,
            model=model,
            fixed_at_default=fixed_at_default,
            remainder=remainder,
            group_only_regressors=group_only_regressors,
            regression_models=regression_models,
            regression_covariates=regression_covariates,
            group_param_dist=meta_params["group_param_dist"],
            gen_norm_std=meta_params["gen_norm_std"],
            uniform_buffer=meta_params["uniform_buffer"],
            gen_std_std=meta_params["gen_std_std"],
            covariate_range=meta_params["covariate_range"],
        )

        if data is None:
            data_, full_parameter_dict = make_single_sub_cond_df_gen(
                conditions_df=conditions_df,
                group_only=group_only,
                depends_on=depends_on,
                model=model,
                fixed_at_default=fixed_at_default,
                remainder=remainder,
                regression_models=regression_models,
                regression_covariates=regression_covariates,
                group_only_regressors=group_only_regressors,
                group_level_parameter_dict=group_level_param_dict,
                n_trials_per_subject=n_trials_per_subject,
                n_subjects=n_subjects,
            )
        else:
            data_, full_parameter_dict = make_single_sub_cond_df_from_gt(
                data=data,
                conditions_df=conditions_df,
                group_only=group_only,
                depends_on=depends_on,
                model=model,
                fixed_at_default=fixed_at_default,
                remainder=remainder,
                regression_models=regression_models,
                regression_covariates=regression_covariates,
                group_only_regressors=group_only_regressors,
                group_level_parameter_dict=group_level_param_dict,
            )
        # params_ok_all = 1
        params_ok_all = check_params(data=data_, model=model)
        cnt += 1

    return data_, full_parameter_dict
