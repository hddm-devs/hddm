from hddm.simulators import *

# from hddm.simulators import boundary_functions
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import os
import hddm

# import sys
import kabuki
import pandas as pd
import seaborn as sns

from kabuki.analyze import post_pred_gen, post_pred_compare_stats
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import truncnorm
from scipy.stats import scoreatpercentile
from scipy.stats.mstats import mquantiles
from copy import deepcopy

from hddm.model_config import model_config


# PREPROCESSING FUNCTIONS ------------------------------------------------------------------------------------------------------------------
def untransform_traces(traces=None, model=None, is_nn=False):
    """
    Check which traces are transformed and apply inverse logit transform to them. (This works only if the transformed variable was
    originally transformed via a logistic function. In hddm this is the standard transform applied to for example the z, parameter.)
    """
    # Check if any traces have the 'trans' label and apply inverse logit transform to get the trace back in original parameterization
    for key in traces.keys():
        if "_trans" in key:
            param_idx = key.find("_")
            param_id = key[:param_idx]

            if not model_config[model]["params_trans"][
                model_config[model]["params"].index(param_id)
            ]:
                print(
                    "You are applying the sigmoid transformation with the ",
                    param_id,
                    " parameter",
                )
                print("Is this intended ?")

            if is_nn:
                lower_lim = model_config[model]["param_bounds"][0][
                    model_config[model]["params"].index(param_id)
                ]
                upper_lim = model_config[model]["param_bounds"][1][
                    model_config[model]["params"].index(param_id)
                ]
            else:
                lower_lim = 0
                upper_lim = 1

            val_tmp = lower_lim + (upper_lim - lower_lim) * (
                1 / (1 + np.exp(-traces[key]))
            )
            traces[key] = val_tmp
            traces.rename(columns={key: key.replace("_trans", "")}, inplace=True)
    return traces


def get_subj_ids(data):
    """
    Get unique subject ids from the supplied panda DataFrame. (expect to find a 'subj_idx' column.)
    """
    # get unique subject ids corresponding to a data subset
    return data["subj_idx"].unique()


def subset_data(data, row_tmp):
    """
    Subset data based on the values in row_tmp (pandas series)
    """
    # note row_tmp is expected as a pandas series (shape (n,) DataFrame)
    data_subset = data
    for key in row_tmp.keys():
        data_subset = data_subset.loc[data_subset[key] == row_tmp[key], :]
    return data_subset


def make_trace_plotready_h_c(
    trace_dict=None,
    model="",
    is_group_model=None,  # Not used ?
    model_ground_truth=None,
):
    """
    Brings the supplied trace dictionary into the final form. A dictionary of dictionaries, with first-level key indicating the
    condition number and second-level key indicating the subject id.
    Does not yet work for Regression based models.
    """
    # This should make traces plotready for the scenarios -->  condition, hierarchical (single model is dealt with trivially by using the traces as is and simply reordering the parmeters to match simulator inputs)
    # Function returns enough data to be flexibly usable across a variety of graphs. One just has to fish out the relevant parts.

    dat_h_c = (
        {}
    )  # will store a dictionary of dictionaries of dictionaries. Level-one keys specify conditions, level-two keys specify subject level data

    for key in trace_dict.keys():
        dat_h_c[key] = {}  # intialize condition level dict
        unique_subj_ids = trace_dict[key]["data"][
            "subj_idx"
        ].unique()  # get unique subject ids for conditions (number and id of subjects may differ across conditions)

        for subj_id in unique_subj_ids:
            dat_h_c[key][subj_id] = {}  # initialize subject level dict
            dat_h_c[key][subj_id]["traces"] = np.zeros(
                (trace_dict[key]["traces"].shape[0], len(model_config[model]["params"]))
            )  # initialize trace container
            dat_h_c[key][subj_id]["data"] = trace_dict[key]["data"].loc[
                trace_dict[key]["data"]["subj_idx"] == subj_id, :
            ]  # add data from the given subject in given condition

            # Check if data contains ground truth parameters (once for the case where we supply 'model_ground_truth' and once where we didn't)
            # AF-TODO: Reorganize this so that we supply trial by trial parameters separately

            test_passed = 1
            if model_ground_truth is None:
                for check_param in model_config[model]["params"]:
                    if check_param in list(dat_h_c[key][subj_id]["data"].keys()):
                        pass
                    else:
                        test_passed = 0
            else:
                for check_param in model_config[model_ground_truth]["params"]:
                    if check_param in list(dat_h_c[key][subj_id]["data"].keys()):
                        pass
                    else:
                        test_passed = 0

            # If the data contain the ground truth parameters for the fitted model, but no ground truth model was specified
            # we can savely assume that the ground truth model was the fitted model.
            if test_passed and (model_ground_truth is None):
                model_ground_truth = model

            # Dat gt_parameter_vector to dat_h_c dict
            # If parameters not in the dataframe --> set to None
            if test_passed:
                dat_h_c[key][subj_id]["gt_parameter_vector"] = (
                    dat_h_c[key][subj_id]["data"]
                    .iloc[0, :][
                        [param for param in model_config[model_ground_truth]["params"]]
                    ]
                    .values
                )
            else:
                dat_h_c[key][subj_id]["gt_parameter_vector"] = None

            # We want to store two version of the trace_names in lists
            # One that isolates the model parameter name, and the original
            trace_names_tmp = []
            trace_names_param_only_tmp = []

            # Go through trace dictionary and add relevant traces to dat_h_c
            for trace_key in trace_dict[key]["traces"].keys():
                # If the current subj_id is not in the trace name --> skip
                if ("subj" in trace_key) and (not (str(subj_id) in trace_key)):
                    continue
                else:
                    trace_names_tmp.append(trace_key)
                    key_param_only = trace_key

                    if "subj" in trace_key and (str(subj_id) in trace_key):
                        key_param_only = trace_key.split("_")[0]
                    if not ("subj" in trace_key) and ("(" in trace_key):
                        key_param_only = trace_key.split("(")[0]

                    trace_names_param_only_tmp.append(key_param_only)

                    dat_h_c[key][subj_id]["traces"][
                        :, model_config[model]["params"].index(key_param_only)
                    ] = trace_dict[key]["traces"][trace_key]
                    full_condition_subj_label = trace_dict[key]["condition_label"]

                    if trace_dict[key]["condition_label"] is not None:
                        full_condition_subj_label["subj_idx"] = subj_id
                    else:
                        full_condition_subj_label = pd.DataFrame(
                            [subj_id], columns=["subj_idx"]
                        )

                    dat_h_c[key][subj_id]["cond_subj_label"] = full_condition_subj_label
                    dat_h_c[key][subj_id]["condition_label"] = trace_dict[key][
                        "condition_label"
                    ]

            dat_h_c[key][subj_id]["trace_names"] = trace_names_tmp

    return dat_h_c


def pick_out_params_h_c(
    condition_dataframe=None,
    data=None,
    params_default_fixed=None,
    params_subj_only=None,
    params_depends=None,
    params_group_only=None,
    is_group_model=True,
):
    """
    Function goes through the conditions defined implicitly in the dataset and picks out the condition relevant parameters.
    Returns a dictionary which stores condition relevant data, parameters and condition labels.
    """
    # params_default_fixed
    param_ids = list(params_default_fixed)

    if not is_group_model:
        for param_tmp in params_subj_only:
            param_str = param_tmp
            param_ids.append(param_str)
    else:
        ids = get_subj_ids(data=data)
        # Parameters which do NOT depend on covariates --> consider only subject level
        for param_tmp in params_subj_only:
            for id_tmp in ids:
                # make str
                param_str = param_tmp + "_subj." + str(id_tmp)
                param_ids.append(param_str)

        # There might be more group only parameters than the ones we find in params_depends
        # --> we add these to our parameter ids directly
        # Their traces carry simply the respective model parameter name

        # Make sure that parms_group only is actually supplied
        # If not we skip
        if not params_group_only == None:
            if params_depends == None:
                params_depends_tmp = []
            else:
                params_depends_tmp = params_depends.copy()

            if len(set(params_group_only) - set(params_depends_tmp)) > 0:
                for param_tmp in set(params_group_only) - set(params_depends):
                    param_str = str(param_tmp)
                    param_ids.append(param_str)

    out_dict = {}
    # We now pick out all the parameters that are relevant for a particular group's data

    # NOTE: The intended use of this is to get subject level posterior predictives, which leaves out group level parameters for which subj level
    # parameters are specified. Group level posteriors (samples from the posterior on the group level, without using subject level traces) are not considered here.
    if condition_dataframe is not None:
        n_ = condition_dataframe.shape[0]
        for i in range(n_):
            param_ids_by_condition = param_ids.copy()

            for param_tmp in params_depends.keys():
                depend_cols = params_depends[param_tmp]
                depend_cols_sorted = np.array(depend_cols)[
                    np.argsort(
                        np.array(
                            [
                                list(condition_dataframe.keys()).index(col)
                                for col in depend_cols
                            ]
                        )
                    )
                ]
                row_tmp = condition_dataframe.iloc[i][depend_cols]
                data_subset = subset_data(data=data, row_tmp=row_tmp)
                # If group model --> consider subject level
                if is_group_model:
                    if param_tmp in params_group_only:
                        param_str = (
                            param_tmp
                            + "("
                            + ".".join(
                                [
                                    str(row_tmp[col_tmp])
                                    for col_tmp in depend_cols_sorted
                                ]
                            )
                            + ")"
                        )
                        param_ids_by_condition.append(param_str)
                    else:
                        ids = get_subj_ids(data=data_subset).astype(str)
                        for id_tmp in ids:
                            # make str
                            param_str = (
                                param_tmp
                                + "_subj"
                                + "("
                                + ".".join(
                                    [
                                        str(row_tmp[col_tmp])
                                        for col_tmp in depend_cols_sorted
                                    ]
                                )
                                + ")."
                                + id_tmp
                            )
                            param_ids_by_condition.append(param_str)
                else:  # otherwise --> no need for subject level
                    param_str = (
                        param_tmp
                        + "("
                        + ".".join(
                            [str(row_tmp[col_tmp]) for col_tmp in depend_cols_sorted]
                        )
                        + ")"
                    )
                    param_ids_by_condition.append(param_str)

            # Out dict should now include all parameters that are relevant for the particular condition i,
            # in param_ids_by_condition.
            out_dict[i] = {
                "data": data_subset.copy(),
                "params": param_ids_by_condition.copy(),
                "condition_label": condition_dataframe.iloc[i],
            }

    else:
        out_dict[0] = {"data": data, "params": param_ids, "condition_label": None}

    return out_dict


def filter_subject_condition_traces(
    hddm_model,
    model_ground_truth=None,  # None, 'model_name'
):
    data = hddm_model.data

    # TODO-AF: Take into account 'group-only-knodes'

    # Since hddm asks only for parameters in addition to 'a', 'v', 't' in the include statement
    # for the logic applied here we add those back in to get the full set of parameters which where fit

    # This works for all models thus far (since they follow the 'a', 'v', 't' parameterization)

    # AF-TODO: If adding in other models to HDDM --> we might need a condition here in case some models do not include ['a', 'v', 't'] in the parameters

    includes_full = list(hddm_model.include) + ["a", "v", "t"]
    is_group_model = hddm_model.is_group_model
    depends = hddm_model.depends_on
    group_only_nodes = list(hddm_model.group_only_nodes)

    # If hddmnn get model attribute from arguments
    if hddm_model.nn:
        tmp_cfg = hddm.simulators.model_config[hddm_model.model]
        model = (
            hddm_model.model
        )  # AF-TODO --> Make 'model' part of args for all HDDM classes

    # If hddm hddm_base --> more labor
    else:
        if "sv" in includes_full or "st" in includes_full or "sz" in includes_full:
            model = "full_ddm"
            tmp_cfg = hddm.simulators.model_config["full_ddm"]
        else:
            tmp_cfg = hddm.simulators.model_config["ddm"]
            model = "ddm"

    # identify the parameters that are supposed to be fixed at default levels
    included_params = set(includes_full)
    if depends is not None:
        included_params = included_params.union(set(list(depends.keys())))
    if group_only_nodes is not None:
        included_params = included_params.union(set(group_only_nodes))

    includes_diff = set(tmp_cfg["params"]) - included_params  # - set(group_only_nodes))

    # Here we care about subject wise posterior predictives

    # Scenario 1: We have multiple conditions and / or a group model
    # Use Hierarchical DataFrame
    if (depends is not None) or (depends is None and is_group_model):
        # Get parameters that have condition dependence (finally condition + subj)
        if depends is not None:
            params_depends = list(depends.keys())
            condition_list = []

            for key in depends.keys():
                if type(depends[key]) == str and not (depends[key] in condition_list):
                    condition_list.append(depends[key])
                elif type(depends[key]) == list:
                    for tmp_depend in depends[key]:
                        if not (tmp_depend in condition_list):
                            condition_list.append(tmp_depend)
                else:
                    pass

            # Make dataframe that holds all the unique conditions (via groupby and summarizing with size)
            condition_dataframe = (
                data.groupby(condition_list)
                .size()
                .reset_index()
                .drop(labels=[0], axis=1)
            )
        else:
            params_depends = []
            condition_dataframe = None

        # Get parameters that have no condition dependence (only subj) (but were fit)
        params_subj_only = list(
            set(includes_full) - (set(params_depends).union(set(group_only_nodes)))
        )

        # Get parameters that were not even fit
        # Have to add these parameters to the final trace objects
        params_default_fixed = list(
            includes_diff
        )  # - set(group_only_nodes)) # was computed above
        traces = untransform_traces(
            hddm_model.get_traces(), model=model, is_nn=hddm_model.nn
        )  # untransform_traces(hddm_model.get_traces())

        if not hddm_model.nn:
            traces = _convert_params(traces)

        # Now for each 'frame' define the trace columns which we want to keep !
        # condition_wise_params_dict defines a dictionary which holds the necessary trace data for each 'condition'
        condition_wise_params_dict = pick_out_params_h_c(
            condition_dataframe,
            data=data,
            params_default_fixed=params_default_fixed,
            params_subj_only=params_subj_only,
            params_depends=depends,
            params_group_only=group_only_nodes,
            is_group_model=is_group_model,
        )

        for key_tmp in condition_wise_params_dict.keys():
            # Condition wise params carries all expected parameter names for a given condition
            # Some of these might not have been fit so for the tracees we want to set those to the 'default' as specified by the model config

            # --> Copy parameter names and add traces for included parameters (for which we HAVE traces)
            included_params = condition_wise_params_dict[key_tmp]["params"].copy()
            for param_not_included in list(includes_diff):
                included_params.remove(param_not_included)

            condition_wise_params_dict[key_tmp]["traces"] = traces[
                included_params
            ].copy()

            # --> Add in 'fake' traces for parameters that where fixed as 'default value' as specified by model config
            for param_not_included in list(includes_diff):
                condition_wise_params_dict[key_tmp]["traces"][
                    param_not_included
                ] = model_config[model]["params_default"][
                    model_config[model]["params"].index(param_not_included)
                ]

        plotready_traces = make_trace_plotready_h_c(
            trace_dict=condition_wise_params_dict,
            model=model,
            is_group_model=is_group_model,
            model_ground_truth=model_ground_truth,
        )

        return plotready_traces

    # Scenario 2: Single condition single subject model (or data collapsed across subjects)
    else:
        traces = untransform_traces(
            hddm_model.get_traces(), model=model, is_nn=hddm_model.nn
        )  # untransform_traces(hddm_model.get_traces())

        if not hddm_model.nn:
            traces = _convert_params(traces)

        # Traces transformed into plot-expected format:
        # dim 1: plot number
        # dim 2: subplot number
        # dim 3: trace row
        # dim 4: trace col
        traces_array = np.zeros((traces.shape[0], traces.shape[1]))
        for trace_key in traces.keys():
            traces_array[:, model_config[model]["params"].index(trace_key)] = traces[
                trace_key
            ].copy()

        ######
        # Check if data contais ground truth parameters
        test_passed = 1
        if model_ground_truth is None:
            for check_param in model_config[model]["params"]:
                if check_param in list(data.keys()):
                    pass
                else:
                    test_passed = 0
        else:
            for check_param in model_config[model_ground_truth]["params"]:
                if check_param in list(data.keys()):
                    pass
                else:
                    test_passed = 0

        # Dat gt_parameter_vector to dat_h_c dict
        # If parameters not in the dataframe --> set to None
        if test_passed:
            gt_parameters = data.iloc[0, :][
                [param for param in model_config[model_ground_truth]["params"]]
            ].values
        else:
            gt_parameters = None
        ######

        out_dict = {
            0: {
                0: {
                    "data": data,
                    "params": hddm.simulators.model_config[model]["params"],
                    "trace_names": hddm.simulators.model_config[model]["params"],
                    "traces": traces_array,
                    "gt_parameter_vector": gt_parameters,
                }
            }
        }
        return out_dict


def extract_multi_cond_subj_plot_n(data=None):
    """
    Function expects as data the output of a call to filter_subject_condition_traces().
    We then deduce if we have multiple-conditions, multiple-subjects and the total number of plots we need to produce.
    """
    # Classify plot type:
    # Check if dataset has multiple conditions
    multi_condition = 0
    if len(data.keys()) > 1:
        multi_condition = 1

    # Check if any condition has more than one subject
    multi_subject = 0
    max_len = 0
    for key in data.keys():
        tmp_len = len(data[key].keys())
        if tmp_len > max_len:
            max_len = tmp_len
    if max_len > 1:
        multi_subject = 1

    # Define number of plots we need:
    if multi_subject and multi_condition:
        n_plots = len(data.keys())
    else:
        n_plots = 1

    return multi_condition, multi_subject, n_plots


def _make_plot_sub_data(
    data=None, plot_n=None, multi_subject=None, multi_condition=None, grouped=False
):
    """
    Takes in the outputs of calls to filter_subject_condition_traces() and extract_multi_cond_subj_plot() and returns a dictionary
    that contains the information needed for one plot/figure (subplot).
    """

    if multi_subject and multi_condition:
        # Condition one
        sub_data = data[list(data.keys())[plot_n]]

        # We might want to return grouped data (collapse across subject)
        if grouped:
            grouped_sub_data = {}
            grouped_sub_data[0] = {}
            grouped_sub_data[0]["traces"] = np.vstack(
                [sub_data[key]["traces"] for key in sub_data.keys()]
            )
            grouped_sub_data[0]["data"] = pd.concat(
                [sub_data[key]["data"] for key in sub_data.keys()], axis=0
            )
            # We don't have a real ground truth parameter vector for grouped data
            grouped_sub_data[0]["gt_parameter_vector"] = None
            grouped_sub_data[0]["cond_subj_label"] = sub_data[list(sub_data.keys())[0]][
                "cond_subj_label"
            ]
            # grouped_sub_data[0]['condition_label'] = sub_data['condition_label']
            return grouped_sub_data

    if multi_condition and not multi_subject:
        # Condition two
        sub_data = {}
        for key in data.keys():
            subj_key = list(data[key].keys())[0]
            sub_data[key] = data[key][subj_key]
    if multi_subject and not multi_condition:
        # Condition three
        sub_data = data[list(data.keys())[0]]
    if not multi_subject and not multi_condition:
        # Condition four
        sub_data = data[0]
    return sub_data


def _convert_params(data=None):
    """
    This is called if we supply a hddm_base hddm model to the plot listed below.
    We need to execute some parameter conversion to allow the new simulators to generate
    the correct rt distributions.
    Specifically any (subject wise etc.) 'a' parameter needs to be converted to a / 2.
    """
    for key in data.keys():
        if "a" == key.split("_")[0]:
            data[key] = data[key] / 2
    return data


# --------------------------------------------------------------------------------------------


# PLOTS --------------------------------------------------------------------------------------
# Plot bound
# Mean posterior predictives
def model_plot(
    hddm_model=None,
    dataset=None,
    model_ground_truth=None,
    grouped=False,
    n_posterior_parameters=500,  # optional / styling
    n_simulations_per_parameter=10,  # optional / stiling
    cols=3,  # styling
    max_t=5,  # styling
    show_model=True,  # styling
    show_trajectories=False,  # styling
    n_trajectories=10,  # styling
    color_trajectories="blue",  # styling
    alpha_trajectories=0.2,  # styling
    linewidth_trajectories=1.0,  # styling
    ylimit=2,  # styling
    posterior_linewidth=3,  # styling
    ground_truth_linewidth=3,  # styling
    hist_linewidth=3,  # styling
    bin_size=0.025,  # styling
    save=False,
    scale_x=1.0,  # styling
    scale_y=1.0,  # styling
    delta_t_graph=0.01,  # styling
    legend_fontsize="medium",  # styling
    legend_shadow=False,  # styling
):
    """The model plot is useful to illustrate model behavior graphically. It is quite a flexible
       plot allowing you to show path trajectories and embedded reaction time histograms etc..
       The main feature is the graphical illustration of a given model
       (this works for 'ddm', 'ornstein', 'levy', 'weibull', 'angle') separately colored for the ground truth parameterization
       and the parameterizations supplied as posterior samples from a hddm sampling run.

    Arguments:
        hddm_model: hddm model object <default=None>
            If you supply a ground truth model, the data you supplied to the hddm model should include trial by trial parameters.
        dataset: panda.DataFrame object <default=None>
            Instead of a hddm_model you might supply a dataset which may include multiple subjects. This is mostly useful to investigate
            model behavior. Note, the plot considers only the 'subj_idx' column of the dataset to create subplots.
        model_ground_truth: str <default=None>
            Specify the ground truth model (mostly useful for parameter recovery studies). If you specify a ground truth model, make sure that the dataset
            you supplied to your hddm model included trial by trial parameters.
        n_posterior_parameters: int <default=500>
            Number of posterior samples to draw for plotting. This needs to be smaller or equal to the number
            of posterior samples supplied to the plot.
        n_simulations_per_parameter: int <default=10>
            How many simulations to perform for each posterior parameter vector drawn.
        cols: int <default=3>
            Number of columns to split the plot into.
        max_t: float <default=10>
            Maximim reaction time to allow for internal simulations for the plot.
        show_model: bool <default=True>
            Whether or not to show the model in the final output (other option is to just show reaction time histograms)
        show_trajectories: bool <default=False>
            Whether or not to show some example trajectories of the simulators.
        n_trajectories: int <default=10>
            Number of trajectories to show if the show_trajectories argument is set to True,
        color_trajectories: str <default='blue'>
            Color of the trajectories if the show_trajectories arguent is set to True.
        alpha_trajectories: float <default=0.2>
            Sets transparency level of trajectories if the show_trajectories argument is set to True.
        linewidth_trajectories: float <default=1.0>
            Sets the linewidth of trajectories if the show_trajectories argument is set to True.
        ylimit: float <default=2>
            Sets the limit on the y-axis
        posterior_linewidth: float <default=3>
            Linewidth of the model visualizations corresponding to posterior samples.
        ground_truth_linewidth: float <default=3>
            Linewidth of the model visualization corresponding to the ground truth model
        hist_linewidth: float <default=3>
            Linewidth of the reaction time histograms (for gorund truth and posterior samples).
            To hide the reaction time histograms, set it to 0.
        bin_size: float <default=0.025>
            Bin size for the reaction time histograms.
        save: bool <default=False>
            Whether to save the plot
        scale_x: float <default=1.0>
            Scales the x axis of the graph
        scale_y: float <default=1.0>
            Salces the y axes o the graph
        delta_t_graph: float <default=0.01>
            Timesteps to use for the simulation runs performed for plotting.
        legend_fontsize: str <default='medium>
            Specify the fontsize of legends. Allowed are for example: 'x-small', 'small', 'medium', 'large', 'x-large'.
            See matplotlib documentation for further possible values.
        legend_shadow: bool <default=False>
            Should the legend have a box with shadow around it?
    Return: plot object
    """

    # Model-plot specific functions ---------
    def split_dataset_model_plot(data=None, model_ground_truth=None):
        data_dict = {}
        data_dict[0] = {}
        for subj_id in data["subj_idx"].unique():
            data_dict[0][subj_id] = {}
            data_dict[0][subj_id]["data"] = data.loc[data["subj_idx"] == subj_id, :][
                ["rt", "response"]
            ]
            if model_ground_truth is not None:
                data_dict[0][subj_id]["gt_parameter_vector"] = (
                    data.loc[data["subj_idx"] == subj_id, :]
                    .iloc[0][
                        [param for param in model_config[model_ground_truth]["params"]]
                    ]
                    .values
                )
        return data_dict

    # ----------------------------------------

    if hddm_model is None and model_ground_truth is None:
        return "Please provide either posterior samples, \n or a ground truth model and parameter set to plot something here. \n Currently you are requesting an empty plot"
    if hddm_model is not None and dataset is not None:
        return "Plesae provide either a dataset directly OR a fitted hddm_model object. You seem to have supplied both"

    if hddm_model is not None:
        assert (
            len(model_config[hddm_model.model]["choices"]) == 2
        ), "The model_plot() currently only works for 2-choice models"

    # AF-TODO: Shape checks
    if hddm_model is not None:
        data = filter_subject_condition_traces(
            hddm_model, model_ground_truth=model_ground_truth
        )
        multi_condition, multi_subject, n_plots = extract_multi_cond_subj_plot_n(
            data=data
        )

    if dataset is not None:
        data = split_dataset_model_plot(
            data=dataset, model_ground_truth=model_ground_truth
        )
        multi_condition, multi_subject, n_plots = extract_multi_cond_subj_plot_n(
            data=data
        )

    # Some style settings
    sns.set(style="white", palette="muted", color_codes=True, font_scale=2)

    # Outer for loop loops over Figures
    for plot_n in range(n_plots):
        sns.despine(right=True)

        t_s = np.arange(0, max_t, delta_t_graph)
        nbins = int((max_t) / bin_size)

        # Create data specifically for the current figure
        sub_data = _make_plot_sub_data(
            data=data,
            plot_n=plot_n,
            multi_subject=multi_subject,
            multi_condition=multi_condition,
            grouped=grouped,
        )

        # Define number of rows we need for display
        n_subplots = len(list(sub_data.keys()))
        if n_subplots > 1:
            rows = int(np.ceil(n_subplots / cols))
        else:
            rows = 1

        # If the data is grouped, by definition we will have a single plot for each figure --> rows and cols set to 1
        if grouped:
            rows = 1
            cols = 1

        # set up figure
        fig, ax = plt.subplots(
            rows,
            cols,
            figsize=(20 * scale_x, 20 * rows * scale_y),
            sharex=False,
            sharey=False,
        )

        # Inner for loop:
        # Loops over subplots (could be 1)
        subplot_cnt = 0
        for i in sub_data.keys():
            if grouped and subplot_cnt > 0:
                continue

            row_tmp = int(np.floor(subplot_cnt / cols))
            col_tmp = subplot_cnt - (cols * row_tmp)

            # SET BASIC AXIS OBJECTS
            if rows > 1 and cols > 1:
                ax[row_tmp, col_tmp].set_xlim(0, max_t)
                ax[row_tmp, col_tmp].set_ylim(-ylimit, ylimit)
            elif (rows == 1 and cols > 1) or (rows > 1 and cols == 1):
                ax[subplot_cnt].set_xlim(0, max_t)
                ax[subplot_cnt].set_ylim(-ylimit, ylimit)
            else:
                ax.set_xlim(0, max_t)
                ax.set_ylim(-ylimit, ylimit)

            if rows > 1 and cols > 1:
                ax_tmp = ax[row_tmp, col_tmp]
                ax_tmp_twin_up = ax[row_tmp, col_tmp].twinx()
                ax_tmp_twin_down = ax[row_tmp, col_tmp].twinx()
            elif (rows == 1 and cols > 1) or (rows > 1 and cols == 1):
                ax_tmp = ax[subplot_cnt]
                ax_tmp_twin_up = ax[subplot_cnt].twinx()
                ax_tmp_twin_down = ax[subplot_cnt].twinx()
            else:
                ax_tmp = ax
                ax_tmp_twin_up = ax.twinx()
                ax_tmp_twin_down = ax.twinx()

            ax_tmp_twin_up.set_ylim(-ylimit, ylimit)
            ax_tmp_twin_up.set_yticks([])

            ax_tmp_twin_down.set_ylim(ylimit, -ylimit)
            ax_tmp_twin_down.set_yticks([])

            # ADD TRAJECTORIES OF GROUND TRUTH VECTOR
            if (
                (show_trajectories == True)
                and (model_ground_truth is not None)
                and (not grouped)
            ):
                for k in range(n_trajectories):
                    out = simulator(
                        theta=sub_data[i][
                            "gt_parameter_vector"
                        ],  # ground_truth_parameters[i, :],
                        model=model_ground_truth,
                        n_samples=1,
                        bin_dim=None,
                    )
                    ax_tmp.plot(
                        out[2]["t"]
                        + np.arange(
                            0, out[2]["max_t"] + out[2]["delta_t"], out[2]["delta_t"]
                        )[out[2]["trajectory"][:, 0] > -999],
                        out[2]["trajectory"][out[2]["trajectory"] > -999],
                        color=color_trajectories,
                        alpha=alpha_trajectories,
                        linewidth=linewidth_trajectories,
                    )

                    # ax_ins = ax.inset_axes([1, 0.5, 0.2, 0.2]) --> important for levy ! AF TODO
                    # ax_ins.plot([0, 1, 2, 3])

            # RUN SIMULATIONS: POSTERIOR SAMPLES
            # AF - NOT NEEDED IF JUST DATASET / NO INTERFERENCE
            if hddm_model is not None:
                tmp_post = np.zeros(
                    (n_posterior_parameters * n_simulations_per_parameter, 2)
                )
                idx = np.random.choice(
                    sub_data[i]["traces"].shape[0],
                    size=n_posterior_parameters,
                    replace=False,
                )
                out = simulator(
                    theta=sub_data[i]["traces"][idx, :],
                    model=hddm_model.model,
                    n_samples=n_simulations_per_parameter,
                    bin_dim=None,
                )

                tmp_post = np.column_stack(
                    [out[0].squeeze().flatten(), out[1].squeeze().flatten()]
                )
                # post_dict[i] = np.column_stack([out[0].squeeze().flatten(), out[1].squeeze().flatten()])
                # tmp_post[(n_simulations_per_parameter * j):(n_simulations_per_parameter * (j + 1)), :] = np.concatenate([out[0], out[1]], axis = 1)

                # ADD HISTOGRAMS
                # POSTERIOR BASED HISTOGRAM
                choice_p_up_post = np.sum(tmp_post[:, 1] == 1) / tmp_post.shape[0]

                counts_2_up, bins = np.histogram(
                    tmp_post[tmp_post[:, 1] == 1, 0],
                    bins=np.linspace(0, max_t, nbins),
                    density=True,
                )

                counts_2_down, _ = np.histogram(
                    tmp_post[(tmp_post[:, 1] == -1) + (tmp_post[:, 1] == 0), 0],
                    bins=np.linspace(0, max_t, nbins),
                    density=True,
                )

                # if j == (n_posterior_parameters - 1) and row_tmp == 0 and col_tmp == 0:
                tmp_label = "Posterior Predictive"
                # else:
                # tmp_label = None

                ax_tmp_twin_up.hist(
                    bins[:-1],
                    bins,
                    weights=choice_p_up_post * counts_2_up,
                    histtype="step",
                    alpha=0.5,
                    color="black",
                    edgecolor="black",
                    zorder=-1,
                    label=tmp_label,
                    linewidth=hist_linewidth,
                )

                ax_tmp_twin_down.hist(
                    bins[:-1],
                    bins,
                    weights=(1 - choice_p_up_post) * counts_2_down,
                    histtype="step",
                    alpha=0.5,
                    color="black",
                    edgecolor="black",
                    linewidth=hist_linewidth,
                    zorder=-1,
                )

            # DATASET BASED HISTOGRAMS
            if sub_data[i]["data"] is not None:
                # These splits here is neither elegant nor necessary --> can represent ground_truth_data simply as a dict !
                # Wiser because either way we can have varying numbers of trials for each subject !
                counts_2_up, bins = np.histogram(
                    sub_data[i]["data"]
                    .loc[sub_data[i]["data"]["response"] == 1.0, :]["rt"]
                    .values,
                    bins=np.linspace(0, max_t, nbins),
                    density=True,
                )

                counts_2_down, _ = np.histogram(
                    np.abs(
                        sub_data[i]["data"]
                        .loc[
                            (sub_data[i]["data"]["response"] == -1.0)
                            | (sub_data[i]["data"]["response"] == 0.0),
                            :,
                        ]["rt"]
                        .values
                    ),
                    bins=np.linspace(0, max_t, nbins),
                    density=True,
                )

                choice_p_up_true_dat = (
                    np.sum(sub_data[i]["data"]["response"].values == 1)
                    / sub_data[i]["data"].values.shape[0]
                )

                if row_tmp == 0 and col_tmp == 0:
                    tmp_label = "Dataset"
                else:
                    tmp_label = None

                ax_tmp_twin_up.hist(
                    bins[:-1],
                    bins,
                    weights=choice_p_up_true_dat * counts_2_up,
                    histtype="step",
                    alpha=0.5,
                    color="blue",
                    edgecolor="blue",
                    zorder=-1,
                    linewidth=hist_linewidth,
                    label=tmp_label,
                )

                ax_tmp_twin_down.hist(
                    bins[:-1],
                    bins,
                    weights=(1 - choice_p_up_true_dat) * counts_2_down,
                    histtype="step",
                    alpha=0.5,
                    color="blue",
                    edgecolor="blue",
                    linewidth=hist_linewidth,
                    zorder=-1,
                )

                if row_tmp == 0 and col_tmp == 0:
                    ax_tmp_twin_up.legend(
                        loc="lower right",
                        fontsize=legend_fontsize,
                        shadow=legend_shadow,
                    )

            # BOUNDS AND SLOPES (model)
            if show_model:
                # This is a little tricky, basically because I ran into trouble with labels being drawn multiple times
                # We have draw most bounds without adding labels to the plot
                # Once we add bounds from ground truth parameters (if supplied) --> no label
                # Once we add a label to our posterior trace based bounds

                if hddm_model is None:
                    # If we didn't supply posterior_samples but want to show model
                    # we set n_posterior_parameters to 0
                    n_posterior_parameters = 0

                for j in range(n_posterior_parameters + 1):
                    tmp_label = ""
                    if j == (
                        n_posterior_parameters - 1
                    ):  # attach label to one posterior samples
                        tmp_label = "Model Samples"
                        tmp_model = hddm_model.model
                        tmp_samples = sub_data[i]["traces"][
                            idx[j], :
                        ]  # posterior_samples[i, idx[j], :]
                        # tmp_samples = posterior_samples[i, idx[j], :]
                        tmp_alpha = 0.05
                        tmp_color = "black"
                        tmp_linewidth = posterior_linewidth
                    elif (
                        (j == n_posterior_parameters)
                        and (model_ground_truth is not None)
                        and (not grouped)
                    ):  # ground truth parameters
                        tmp_samples = sub_data[i][
                            "gt_parameter_vector"
                        ]  # ground_truth_parameters[i, :]
                        tmp_model = model_ground_truth

                        # If we supplied ground truth data --> make ground truth model blue, otherwise red
                        tmp_colors = ["red", "blue"]
                        tmp_bool = sub_data[i]["data"] is not None
                        tmp_color = tmp_colors[int(tmp_bool)]
                        tmp_alpha = 1
                        tmp_label = "Ground Truth Model"
                        tmp_linewidth = ground_truth_linewidth
                    elif (
                        j == n_posterior_parameters and model_ground_truth == None
                    ):  # If no ground truth model and we are at j = n_posterior samples --> break out of loop
                        break
                    elif j < (
                        n_posterior_parameters - 1
                    ):  # run through posterior samples, but don't include label
                        tmp_model = hddm_model.model
                        tmp_samples = sub_data[i]["traces"][
                            idx[j], :
                        ]  # posterior_samples[i, idx[j], :]
                        tmp_alpha = 0.05
                        tmp_color = "black"
                        tmp_label = None
                        tmp_linewidth = posterior_linewidth
                    else:
                        continue

                    # MAKE BOUNDS (FROM MODEL CONFIG) !
                    if tmp_model == "weibull":
                        b = np.maximum(
                            tmp_samples[1]
                            * model_config[tmp_model]["boundary"](
                                t=t_s, alpha=tmp_samples[4], beta=tmp_samples[5]
                            ),
                            0,
                        )

                    if tmp_model == "angle" or tmp_model == "angle2":
                        b = np.maximum(
                            tmp_samples[1]
                            + model_config[tmp_model]["boundary"](
                                t=t_s, theta=tmp_samples[4]
                            ),
                            0,
                        )

                    if (
                        tmp_model == "ddm"
                        or tmp_model == "ornstein"
                        or tmp_model == "levy"
                        or tmp_model == "full_ddm"
                    ):
                        b = tmp_samples[1] * np.ones(t_s.shape[0])

                    # MAKE SLOPES (VIA TRAJECTORIES) !
                    # AF-TODO: Can probably be optimized for speed by plotting only a few points to draw the slope line
                    out = simulator(
                        theta=tmp_samples,
                        model=tmp_model,
                        n_samples=1,
                        no_noise=True,
                        delta_t=delta_t_graph,
                        bin_dim=None,
                    )

                    tmp_traj = out[2]["trajectory"]
                    maxid = np.minimum(
                        np.argmax(np.where(tmp_traj > -999)), t_s.shape[0]
                    )

                    # Upper bound
                    ax_tmp.plot(
                        t_s + tmp_samples[model_config[tmp_model]["params"].index("t")],
                        b,
                        tmp_color,
                        alpha=tmp_alpha,
                        zorder=1000 + j,
                        linewidth=tmp_linewidth,
                        label=tmp_label,
                    )

                    # Lower bound
                    ax_tmp.plot(
                        t_s + tmp_samples[model_config[tmp_model]["params"].index("t")],
                        -b,
                        tmp_color,
                        alpha=tmp_alpha,
                        zorder=1000 + j,
                        linewidth=tmp_linewidth,
                    )

                    # Slope
                    ax_tmp.plot(
                        t_s[:maxid]
                        + tmp_samples[model_config[tmp_model]["params"].index("t")],
                        tmp_traj[:maxid],
                        c=tmp_color,
                        alpha=tmp_alpha,
                        zorder=1000 + j,
                        linewidth=tmp_linewidth,
                    )  # TOOK AWAY LABEL

                    # Starting point
                    ax_tmp.axvline(
                        x=tmp_samples[
                            model_config[tmp_model]["params"].index("t")
                        ],  # this should identify the index of ndt directly via model config !
                        ymin=-ylimit,
                        ymax=ylimit,
                        c=tmp_color,
                        linestyle="--",
                        linewidth=tmp_linewidth,
                        alpha=tmp_alpha,
                    )

                    if (
                        tmp_label == "Ground Truth Model"
                        and row_tmp == 0
                        and col_tmp == 0
                    ):
                        ax_tmp.legend(
                            loc="upper right",
                            fontsize=legend_fontsize,
                            shadow=legend_shadow,
                        )

                    if rows == 1 and cols == 1:
                        ax_tmp.patch.set_visible(False)

            subplot_cnt += 1

            # Make condition label
            condition_label = ""
            fig_title_tmp = ""
            if hddm_model is not None:
                if (multi_condition and multi_subject) or (
                    not multi_condition and multi_subject
                ):
                    for label_key in sub_data[i]["cond_subj_label"].keys():
                        if "subj_idx" not in label_key:
                            condition_label += str(label_key) + ": "
                            condition_label += (
                                str(
                                    sub_data[i]["cond_subj_label"][[label_key]].values[
                                        0
                                    ]
                                )
                                + ", "
                            )
                    condition_label = condition_label[:-2]

            title_size = 24
            if (multi_condition and multi_subject) or (
                not multi_condition and multi_subject
            ):
                title_tmp = "Subject: " + str(i)
                fig_title_tmp = condition_label
            elif multi_condition and not multi_subject:
                # NOTE: We should never end up here if only a dataset was supplied
                title_tmp = condition_label
                title_size = title_size / (
                    0.5 * len(list(sub_data[i]["cond_subj_label"].keys()))
                )
            elif not multi_condition and not multi_subject:
                # No extra title needed for simple single subject plot
                title_tmp = ""

            # Set plot-global title
            if "cond_subj_label" in list(sub_data[i].keys()):
                fig.suptitle(
                    fig_title_tmp,
                    fontsize=26
                    / (0.5 * len(list(sub_data[i]["cond_subj_label"].keys()))),
                    y=0.9,
                )
            else:
                pass

            # Set x and y axis labels
            if row_tmp == (rows - 1):
                ax_tmp.set_xlabel(
                    "rt", fontdict={"fontsize": 24 * ((scale_x + scale_y) / 2)}
                )
            ax_tmp.set_ylabel("", fontdict={"fontsize": 24 * ((scale_x + scale_y) / 2)})

            ax_tmp.set_title(
                title_tmp, fontdict={"fontsize": 24 * ((scale_x + scale_y) / 2)}
            )
            ax_tmp.tick_params(axis="y", labelsize=20 * ((scale_x + scale_y) / 2))
            ax_tmp.tick_params(axis="x", labelsize=20 * ((scale_x + scale_y) / 2))

            # Some extra styling:
            if (model_ground_truth is not None) and (not grouped):
                if show_model:
                    # vertical dashed line at the gorund truth non-decision time
                    ax_tmp.axvline(
                        x=sub_data[i]["gt_parameter_vector"][
                            model_config[model_ground_truth]["params"].index("t")
                        ],
                        ymin=-ylimit,
                        ymax=ylimit,
                        c="blue",
                        linestyle="--",
                    )
                # horizontal (y = 0) line connecting y-axis with the first datapoints
                ax_tmp.axhline(
                    y=0,
                    xmin=0,
                    xmax=sub_data[i]["gt_parameter_vector"][
                        model_config[model_ground_truth]["params"].index("t")
                    ]
                    / max_t,
                    c="blue",
                    linestyle="--",
                )

        # Turn off subplots which were not needed in a given display
        if rows > 1 and cols > 1:
            for i in range(n_subplots, rows * cols, 1):
                row_tmp = int(np.floor(i / cols))
                col_tmp = i - (cols * row_tmp)
                ax[row_tmp, col_tmp].axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.9])

        if save == True:
            plt.savefig(
                "figures/" + "model_plot_" + str(plot_n) + "_" + str(i) + ".png",
                format="png",
                transparent=True,
                frameon=False,
            )
            plt.close()
        else:
            plt.show()

    return plt.show()


# Posterior Pair Plot
def plot_posterior_pair(
    hddm_model=None,
    axes_limits="samples",  # 'samples' or dict({'parameter_name': [lower bound, upper bound]})
    font_scale=1.5,
    height=2,
    aspect_ratio=1,
    n_subsample=1000,
    kde_levels=50,
    model_ground_truth=None,
    save=False,
    save_path=None,
    show=True,
):
    """Basic pair plot useful for inspecting posterior parameters.
       At this point can be used only for single subject data.
       Works for all models listed in hddm (e.g. 'ddm', 'angle', 'weibull', 'levy', 'ornstein')

    Arguments:
        hddm_model: hddm model object <default=None>
            If you supply a ground truth model, the data you supplied to the hddm model should include trial by trial parameters.
        axes_limits: str or dict <default='samples'>
            Either a string that says 'samples', which makes axes limits depends on the posterior sample values directly,
            (separately for each parameter). Or a dictionary with keys parameter names, and values a 2-list with a lower
            and an upper bound ([lower, upper]).
        height: float <default=10>
            Figure height in inches.
        aspect_ratio: float <default=1>
            Aspect ratio of figure
        n_subsample: int <default=1000>
            Number of posterior samples to use for figure finally. Subsamples from the provided traces.
        ground_truth_parameters: dict <default=None>
            Ground truth parameters (will be shown in the plot if supplied). Supplied as a dict of the form (e.g. DDM)
            {'v': 1, 'a': 2, 'z': 0.5, 't': 2}
        model_fitted: str <default=None>
            String that supplies which model was fitted to the data.
        font_scale: float <default= 1.5>
            Scale of fonts. Not always predictable which is the best value, so
            the setting is left to the user.
        save: bool <default= False>
            Whether or not to save the figure.
    Return: plot object
    """

    if hddm_model == None:
        return "No data supplied --> please supply a HDDM model (including traces)"

    model_fitted = hddm_model.model

    data = filter_subject_condition_traces(
        hddm_model=hddm_model, model_ground_truth=model_ground_truth
    )

    # print(data)

    sns.set()
    sns.set_theme(
        style="ticks", rc={"axes.spines.right": False, "axes.spines.top": False}
    )
    sns.set_context("notebook", font_scale=font_scale)
    plot_cnt = 0
    for c_tmp in data.keys():
        for s_tmp in data[c_tmp].keys():
            # Turn traces into dataframe:
            # Get all ground truths
            gt_dict = {}

            # AF COMMENT: Is this sorting of trace names necessrary ?
            sorted_trace_names_tmp = data[c_tmp][s_tmp]["trace_names"].copy()
            for trace_name_tmp in data[c_tmp][s_tmp]["trace_names"]:
                if (
                    trace_name_tmp.split("_")[0].split("(")[0]
                    in model_config[model_fitted]["params"]
                ):
                    tmp_param = trace_name_tmp.split("_")[0].split("(")[0]
                    idx_tmp = model_config[model_fitted]["params"].index(tmp_param)
                    sorted_trace_names_tmp[idx_tmp] = trace_name_tmp
                    if model_ground_truth is not None:
                        gt_dict[trace_name_tmp] = data[c_tmp][s_tmp][
                            "gt_parameter_vector"
                        ][idx_tmp]
                else:
                    pass

            data[c_tmp][s_tmp]["trace_names"] = sorted_trace_names_tmp.copy()

            data[c_tmp][s_tmp]["traces"] = pd.DataFrame(
                data[c_tmp][s_tmp]["traces"], columns=data[c_tmp][s_tmp]["trace_names"]
            )

            g = sns.PairGrid(
                data[c_tmp][s_tmp]["traces"].sample(n_subsample),
                height=height,
                aspect=aspect_ratio,
                diag_sharey=False,
            )

            g = g.map_diag(sns.kdeplot, color="black", shade=False)  # shade = True,

            g = g.map_lower(
                sns.kdeplot,
                thresh=0.01,
                n_levels=kde_levels,
                shade=False,
                cmap="Purples_d",
            )  # 'Greys'

            # Hide upper triangular part
            for i, j in zip(*np.triu_indices_from(g.axes, 1)):
                g.axes[i, j].set_visible(False)

            # Get x and y labels of graph as determined by the posterior_samples panda
            xlabels, ylabels = [], []

            for ax in g.axes[-1, :]:
                xlabel = ax.xaxis.get_label_text()
                ax.set_xlabel(ax.get_xlabel(), rotation=45)
                xlabels.append(xlabel)

            for ax in g.axes[:, 0]:
                ylabel = ax.yaxis.get_label_text()
                ylabels.append(ylabel)
                # ax.yaxis.set_label_text('')
                ax.set_ylabel("")

            if axes_limits == "model":
                for i in range(len(xlabels)):
                    for j in range(len(ylabels)):
                        try:
                            g.axes[j, i].set_xlim(
                                model_config[model_fitted]["param_bounds"][0][
                                    model_config[model_fitted]["params"].index(
                                        xlabels[i]
                                    )
                                ],
                                model_config[model_fitted]["param_bounds"][1][
                                    model_config[model_fitted]["params"].index(
                                        xlabels[i]
                                    )
                                ],
                            )
                            g.axes[j, i].set_ylim(
                                model_config[model_fitted]["param_bounds"][0][
                                    model_config[model_fitted]["params"].index(
                                        ylabels[j]
                                    )
                                ],
                                model_config[model_fitted]["param_bounds"][1][
                                    model_config[model_fitted]["params"].index(
                                        ylabels[j]
                                    )
                                ],
                            )
                        except:
                            print(
                                "ERROR: It looks like you are trying to make axis limits dependend on model specific parameters, but the column-names of your posterior traces do not align with the requested model's parameters"
                            )

            elif type(axes_limits) == dict:
                for i in range(len(xlabels)):
                    for j in range(len(ylabels)):
                        try:
                            g.axes[j, i].set_xlim(
                                axes_limits[xlabels[i]][0], axes_limits[xlabels[i]][1]
                            )
                            g.axes[j, i].set_ylim(
                                axes_limits[ylabels[j]][0], axes_limits[ylabels[j]][1]
                            )
                        except:
                            print(
                                "ERROR: Does your axes_limits dictionary match the column names of your posterior_samples DataFrame?"
                            )
                            return

            for ax in g.axes.flat:
                plt.setp(ax.get_xticklabels(), rotation=45)

            g.fig.suptitle(
                model_fitted.upper()
                + " , condition: "
                + str(c_tmp)
                + ", subject: "
                + str(s_tmp),
                y=1.03,
                fontsize=24,
            )

            # posterior_samples_key_set = np.sort(posterior_samples.keys())
            # If ground truth is available add it in:
            if model_ground_truth is not None:
                for i in range(g.axes.shape[0]):
                    for j in range(i + 1, g.axes.shape[0], 1):
                        g.axes[j, i].plot(
                            data[c_tmp][s_tmp]["gt_parameter_vector"][
                                i
                            ],  # [xlabels[i]],
                            data[c_tmp][s_tmp]["gt_parameter_vector"][
                                j
                            ],  # [ylabels[j]],
                            ".",
                            color="red",
                            markersize=10,
                        )

                for i in range(g.axes.shape[0]):
                    if i == 0:
                        y_lims_tmp = g.axes[i, i].get_ylim()
                        g.axes[i, i].set_ylim(0, y_lims_tmp[1])

                    g.axes[i, i].plot(
                        data[c_tmp][s_tmp]["gt_parameter_vector"][
                            i
                        ],  # [xlabels[i]], # ground_truth_parameters[xlabels[i]],
                        g.axes[i, i].get_ylim()[0],
                        ".",
                        color="red",
                        markersize=10,
                    )

                    # AF-COMMENT: The yaxis ticks are supposed to be turned off only for the
                    # diagonal, but seemingly this is applied across the board....
                    g.axes[i, i].yaxis.set_ticks([])

            if save == True:
                if save_path is None:
                    save_path = "figures/"
                    if os.path.exists("figures"):
                        pass
                    else:
                        os.mkdir("figures")
                elif type(save_path) == str:
                    pass
                else:
                    return "Error: please specify a save_path as a string"

                plt.savefig(
                    save_path + "posterior_pair_plot_" + str(plot_cnt) + ".png",
                    format="png",
                    transparent=True,
                )
            if show:
                plt.show()

            if save == True:
                plt.savefig(
                    "figures/" + "pair_plot_" + model_fitted + "_" + datatype + ".png",
                    format="png",
                    transparent=True,
                    frameon=False,
                )
                plt.close()
            plot_cnt += 1

    # Show
    return


def caterpillar_plot(
    hddm_model=None,
    ground_truth_parameter_dict=None,
    drop_sd=True,
    keep_key=None,
    x_limits=[-2, 2],
    aspect_ratio=2,
    figure_scale=1.0,
    save=False,
    show=True,
    tick_label_size_x=22,
    tick_label_size_y=14,
):
    """An alternative posterior predictive plot. Works for all models listed in hddm (e.g. 'ddm', 'angle', 'weibull', 'levy', 'ornstein')

    Arguments:
        hddm_model: hddm model object <default=None>
            If you supply a ground truth model, the data you supplied to the hddm model should include trial by trial parameters.
        model_ground_truth: str <default=None>
            Specify the ground truth model (mostly useful for parameter recovery studies). If you specify a ground truth model, make sure that the dataset
            you supplied to your hddm model included trial by trial parameters.
        drop_sd: bool <default=True>
            Whether or not to drop group level standard deviations from the caterpillar plot.
            This is sometimes useful because scales can be off if included.
        keep_key: list <default=None>
            If you want to keep only a specific list of parameters in the caterpillar plot, supply those here as
            a list. All other parameters for which you supply traces in the posterior samples are going to be ignored.
        x_limits: float <default=2>
            Sets the limit on the x-axis
        aspect_ratio: float <default=2>
            Aspect ratio of plot.
        figure_scale: float <default=1.0>
            Figure scaling. 1.0 refers to 10 inches as a baseline size.
        tick_label_size_x: int <default=22>
            Basic plot styling. Tick label size.
        tick_label_size_y: int <default=14>
            Basic plot styling. Tick label size.
        save: bool <default=False>
            Whether to save the plot

    Return: plot object
    """

    if hddm_model is None:
        return "No HDDM object supplied"

    sns.set(style="white", palette="muted", color_codes=True, font_scale=2)

    trace = untransform_traces(
        traces=hddm_model.get_traces(), model=hddm_model.model, is_nn=hddm_model.nn
    )

    if not hddm_model.nn:
        trace = _convert_params(trace)

    # TODO: Probably have to add parameter adjustments to account for LAN/no LAN case
    if keep_key is None:
        trace_cnt_tmp = trace.shape[1]
    else:
        trace_cnt_tmp = len(keep_key)

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(
            10 * figure_scale,
            0.0333 * trace_cnt_tmp * aspect_ratio * 10 * figure_scale,
        ),
        sharex=False,
        sharey=False,
    )

    sns.despine(right=True)
    ecdfs = {}
    plot_vals = {}  # [0.01, 0.9], [0.01, 0.99], [mean]

    for k in trace.keys():
        # If we want to keep only a specific parameter we skip all traces which don't include it in
        # their names !
        if keep_key is not None and keep_key not in k:
            continue

        # Deal with
        if "std" in k and drop_sd:
            pass

        else:
            ok_ = 1
            k_old = k  # keep original key around for indexing
            k = k.replace("_", "-")  # assign new prettier key for plotting

            if drop_sd == True:
                if "sd" in k:
                    ok_ = 0
            if ok_:
                # Make empirical CDFs and extract the 10th, 1th / 99th, 90th percentiles
                ecdfs[k] = ECDF(trace[k_old])
                tmp_sorted = sorted(trace[k_old])
                _p01 = tmp_sorted[np.sum(ecdfs[k](tmp_sorted) <= 0.01) - 1]
                _p99 = tmp_sorted[np.sum(ecdfs[k](tmp_sorted) <= 0.99) - 1]
                _p1 = tmp_sorted[np.sum(ecdfs[k](tmp_sorted) <= 0.1) - 1]
                _p9 = tmp_sorted[np.sum(ecdfs[k](tmp_sorted) <= 0.9) - 1]
                _pmean = trace[k_old].mean()
                plot_vals[k] = [[_p01, _p99], [_p1, _p9], _pmean]

    x = [plot_vals[k][2] for k in plot_vals.keys()]

    # Create y-axis labels first
    ax.scatter(x, plot_vals.keys(), c="black", marker="s", alpha=0)

    # Plot the actual cdf-based data
    for k in plot_vals.keys():
        k = k.replace("_", "-")
        ax.plot(plot_vals[k][1], [k, k], c="grey", zorder=-1, linewidth=5)
        ax.plot(plot_vals[k][0], [k, k], c="black", zorder=-1)

        # Add in ground truth if supplied
        if ground_truth_parameter_dict is not None:
            ax.scatter(
                ground_truth_parameter_dict[k.replace("-", "_")], k, c="red", marker="|"
            )

    # Styling
    ax.set_xlim(x_limits[0], x_limits[1])
    ax.tick_params(axis="y", size=tick_label_size_y)
    ax.tick_params(axis="x", size=tick_label_size_x)

    if save == True:
        plt.savefig(
            "figures/" + "caterpillar_plot_" + model + "_" + datatype + ".png",
            format="png",
            transparent=True,
            frameon=False,
        )

    if show:
        plt.show()
        return

    return


def posterior_predictive_plot(
    hddm_model=None,
    model_ground_truth="angle",
    grouped=False,
    cols=3,
    n_posterior_parameters=100,
    max_t=20,
    n_simulations_per_parameter=10,
    xlimit=10,
    bin_size=0.025,
    hist_linewidth=3,
    scale_x=0.5,
    scale_y=0.5,
    legend_fontsize="medium",
    legend_shadow=False,
    legend_pos="best",
    save=False,
    save_path=None,
    show=True,
):
    """An alternative posterior predictive plot. Works for all models listed in hddm (e.g. 'ddm', 'angle', 'weibull', 'levy', 'ornstein')

    Arguments:
        hddm_model: hddm model object <default=None>
            If you supply a ground truth model, the data you supplied to the hddm model should include trial by trial parameters.
        model_ground_truth: str <default=None>
            Specify the ground truth model (mostly useful for parameter recovery studies). If you specify a ground truth model, make sure that the dataset
            you supplied to your hddm model included trial by trial parameters.
        grouped: bool <default=False>
            If grouped is True, the graph will group over subjects and generate one plot per condition.
        n_plots: int <default=4>
            The plot attempts to be smart in trying to figure out how many plots are desired, however choosing it manual is a
            save option.
        n_posterior_parameters: int <default=500>
            Number of posterior samples to draw for plotting. This needs to be smaller or equal to the number
            of posterior samples supplied to the plot.
        n_simulations_per_parameter: int <default=10>
            How many simulations to perform for each posterior parameter vector drawn.
        cols: int <default=3>
            Number of columns to split the plot into.
        max_t: float <default=10>
            Maximim reaction time to allow for internal simulations for the plot.
        xlimit: float <default=2>
            Sets the limit on the x-axis
        hist_linewidth: float <default=3>
            Linewidth of the reaction time histograms (for gorund truth and posterior samples).
            To hide the reaction time histograms, set it to 0.
        bin_size: float <default=0.025>
            Bin size for the reaction time histograms.
        save: bool <default=False>
            Whether to save the plot
        scale_x: float <default=1.0>
            Scales the x axis of the graph
        scale_y: float <default=1.0>
            Scales the y axes o the graph
        legend_fontsize: str <default='medium'>
            Fontsize of legend items. Amongst others (see matplotlib documentation for a full list) you can use,
            'x-small', 'small', 'medium', 'large', 'x-large'
        legend_shadow: bool <default=False>
            Should the legend have a box with shadown around it?
        legend_position: str <default='best'>
            Position of the legend item. Amonst others (see matplotlib documentation for a full list) you can use,
            'center', 'upper right', 'lower right', 'upper left', 'lower left'.
        delta_t_graph: float <default=0.01>
            Timesteps to use for the simulation runs performed for plotting.
    Return: plot object
    """
    if hddm_model is not None:
        assert (
            len(model_config[hddm_model.model]["choices"]) == 2
        ), "The posterior_predictive() plot currently only works for 2-choice models"

    if n_posterior_parameters <= 1:
        print("ERROR: n_posterior_parameters needs to be larger than 1")
        return

    # AF-TODO: Shape checks
    if hddm_model is not None:
        data = filter_subject_condition_traces(
            hddm_model, model_ground_truth=model_ground_truth
        )
        multi_condition, multi_subject, n_plots = extract_multi_cond_subj_plot_n(
            data=data
        )

    # Taking care of special case with 1 plot
    if (not multi_condition and not multi_subject) or (grouped):
        cols = 1

    # General plot parameters
    nbins = int((2 * max_t) / bin_size)

    sns.set(style="white", palette="muted", color_codes=True, font_scale=2)

    sns.despine(right=True)

    # Cycle through plots
    fig_title_tmp = ""
    title_size = 24
    for plot_n in range(n_plots):
        sub_data = _make_plot_sub_data(
            data=data,
            plot_n=plot_n,
            multi_subject=multi_subject,
            multi_condition=multi_condition,
        )

        if grouped:
            n_subplots = 1
        else:
            n_subplots = len(list(sub_data.keys()))

        if n_subplots > 1:
            rows = int(np.ceil(n_subplots / cols))
        else:
            rows = 1

        fig, ax = plt.subplots(
            rows,
            cols,
            figsize=(20 * scale_x, 20 * rows * scale_y),
            sharex=False,
            sharey=False,
        )

        # Make condition label (and set global figure title depending on what kind of data we are dealing with)
        condition_label = ""

        if (
            (multi_condition and multi_subject)
            or (not multi_condition and multi_subject)
        ) and not grouped:
            for label_key in sub_data[list(sub_data.keys())[0]][
                "cond_subj_label"
            ].keys():
                if "subj_idx" not in label_key:
                    condition_label += str(label_key) + ": "
                    condition_label += (
                        str(
                            sub_data[list(sub_data.keys())[0]]["cond_subj_label"][
                                [label_key]
                            ].values[0]
                        )
                        + ", "
                    )
            condition_label = condition_label[:-2]

        if (
            (multi_condition and multi_subject)
            or (not multi_condition and multi_subject)
        ) and not grouped:
            fig_title_tmp = condition_label

        # Plot global title
        fig.suptitle(fig_title_tmp, fontsize=title_size)

        # GET SIMULATIONS AND COLLECT GROUND TRUTHS FOR CONDITON
        subplot_cnt = 0
        gt_dict = {}
        post_dict = {}
        for i in sub_data.keys():
            idx = np.random.choice(
                sub_data[i]["traces"].shape[0],
                size=n_posterior_parameters,
                replace=False,
            )

            out = simulator(
                theta=sub_data[i]["traces"][idx, :],  # posterior_samples[i, idx[j], :],
                model=hddm_model.model,
                n_samples=n_simulations_per_parameter,
                bin_dim=None,
            )

            post_dict[i] = np.stack([out[0].flatten(), out[1].flatten()], axis=1)
            gt_dict[i] = sub_data[i]["data"][["rt", "response"]].values
            # Make sure that zero responses are turned negative
            gt_dict[i][:, 1][gt_dict[i][:, 1] == 0] = -1

            subplot_cnt += 1

        # PLOTTING
        subplot_cnt = 0
        gt_color = "blue"
        for i in sub_data.keys():
            # If data is grouped the inner loop has to be passed just once
            # to set the styling.
            if grouped and (subplot_cnt > 0):
                break

            row_tmp = int(np.floor(subplot_cnt / cols))
            col_tmp = subplot_cnt - (cols * row_tmp)

            # Target the correct subplot
            if rows > 1 and cols > 1:
                ax_tmp = ax[row_tmp, col_tmp]

            elif (rows == 1 and cols > 1) or (rows > 1 and cols == 1):
                ax_tmp = ax[subplot_cnt]

            else:
                ax_tmp = ax

            # SUBPLOT LEVEL STYLING
            # Make Titles:
            if (
                (multi_condition and multi_subject)
                or (not multi_condition and multi_subject)
            ) and not grouped:
                title_tmp = "Subject: " + str(i)
            elif (multi_condition and not multi_subject) or grouped:
                # condition label is subplot title if we are dealing with grouped data or datasets
                # which are simply split by condition
                title_tmp = condition_label
                title_size = title_size / (
                    0.5
                    * len(
                        list(
                            sub_data[list(sub_data.keys())[0]]["cond_subj_label"].keys()
                        )
                    )
                )
            elif not multi_condition and not multi_subject:
                # No extra title needed for simple single subject plot
                title_tmp = ""

            # subplot title
            ax_tmp.set_title(title_tmp, fontsize=title_size)

            # subplot x-axis limits
            ax_tmp.set_xlim(-xlimit, xlimit)

            # ground-truth and data labels if we are dealing with the upper right sub-plot in a figure
            if row_tmp == 0 and col_tmp == 0:
                if model_ground_truth is not None:
                    label_0 = "Ground Truth"
                else:
                    label_0 = "DATA"
                ax_tmp.legend(
                    labels=["Posterior Predictive", label_0],
                    fontsize=legend_fontsize,
                    shadow=legend_shadow,
                    loc=legend_pos,  # "upper right",
                )

            # rt x-axis label if we are dealing with the last row of a figure
            if row_tmp == (rows - 1):
                ax_tmp.set_xlabel("rt", fontsize=24)

            # unset ylabel if first column
            if col_tmp == 0:
                ax_tmp.set_ylabel("", fontsize=24)

            # set ticks-size for x and y axis
            ax_tmp.tick_params(axis="y", size=22)
            ax_tmp.tick_params(axis="x", size=22)

            # PLOT DATA
            if grouped:
                post_tmp = np.vstack([post_dict[i] for i in post_dict.keys()])
                gt_tmp = np.vstack([gt_dict[i] for i in gt_dict.keys()])
            else:
                post_tmp = post_dict[i]
                gt_tmp = gt_dict[i]

            # Posterior Predictive
            ax_tmp.hist(
                post_tmp[:, 0] * post_tmp[:, 1],
                bins=np.linspace(
                    -max_t, max_t, nbins
                ),  # 50, # kde = False, # rug = False,
                alpha=1,
                color="black",
                histtype="step",
                density=1,
                edgecolor="black",
                linewidth=hist_linewidth,
            )

            # Ground Truth
            ax_tmp.hist(
                gt_tmp[:, 0] * gt_tmp[:, 1],
                alpha=0.5,
                color=gt_color,
                density=1,
                edgecolor=gt_color,
                histtype="step",
                linewidth=hist_linewidth,
                bins=np.linspace(-max_t, max_t, nbins),  # 50,
                # kde = False, #rug = False,
            )

            # Increment subplot counter
            subplot_cnt += 1

        # Turn off redundant subplots
        if rows > 1 and cols > 1:
            for i in range(n_plots, rows * cols, 1):
                row_tmp = int(np.floor(i / cols))
                col_tmp = i - (cols * row_tmp)
                ax[row_tmp, col_tmp].axis("off")

        # save and return
        if save == True:
            if save_path is None:
                save_path = "figures/"
                if os.path.exists("figures"):
                    pass
                else:
                    os.mkdir("figures")
            elif type(save_path) == str:
                pass
            else:
                return "Error: please specify a save_path as a string"

            plt.savefig(
                save_path
                + "posterior_predictive_plot_"
                + "subplot_"
                + str(plot_n)
                + ".png",
                format="png",
            )
        if show:
            plt.show()
        plt.close()
    return
