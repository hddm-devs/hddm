from hddm.simulators import *
from hddm.generate import *
from hddm.utils import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import arviz as az

import os
import warnings

# import pymc as pm
# import hddm

import pandas as pd
from tqdm import tqdm
import pymc

from kabuki.analyze import _post_pred_generate, _parents_to_random_posterior_sample
from statsmodels.distributions.empirical_distribution import ECDF

from hddm.model_config import model_config
from hddm.model_config_rl import model_config_rl


# Basic utility
def prettier_tag(tag):
    len_tag = len(tag)
    if len_tag == 1:
        return tag[0]
    else:
        return "(" + ", ".join([str(t) for t in tag]) + ")"


# Plot Composer Functions
def plot_posterior_pair(
    model,
    plot_func=None,
    save=False,
    path=None,
    figsize=(8, 6),
    format="png",
    samples=100,
    parameter_recovery_mode=False,
    **kwargs,
):
    """Generate posterior pair plots for each observed node.

    Arguments:

        model: kabuki.Hierarchical
            The (constructed and sampled) kabuki hierarchical model to
            create the posterior preditive from.

    Optional:

        samples: int <default=10>
            How many posterior samples to use.

        columns: int <default=3>
            How many columns to use for plotting the subjects.

        bins: int <default=100>
            How many bins to compute the data histogram over.

        figsize: (int, int) <default=(8, 6)>

        save: bool <default=False>
            Whether to save the figure to a file.

        path: str <default=None>
            Save figure into directory prefix

        format: str or list of strings <default='png'>
            Save figure to a image file of type 'format'. If more then one format is
            given, multiple files are created

        parameter_recovery_mode: bool <default=False>
            If the data attached to the model supplied under the model argument
            has the format expected of the simulator_h_c() function from the simulators.hddm_dataset_generators
            module, then parameter_recovery_mode = True can be use to supply ground truth parameterizations to the
            plot_func argument describes below.

        plot_func: function <default=_plot_posterior_pdf_node>
            Plotting function to use for each observed node
            (see default function for an example).

    Note:

        This function changes the current value and logp of the nodes.

    """
    if hasattr(model, "reg_outcomes"):
        return "Note: The posterior pair plot does not support regression models at this point! Aborting..."

    if hasattr(model, "model"):
        kwargs["model_"] = model.model
    else:
        kwargs["model_"] = "ddm_hddm_base"

    if plot_func is None:
        plot_func = _plot_func_pair

    observeds = model.get_observeds()

    kwargs["figsize"] = figsize
    kwargs["n_samples"] = samples

    # Plot different conditions (new figure for each)
    for tag, nodes in observeds.groupby("tag"):
        # Plot individual subjects (if present)
        for subj_i, (node_name, bottom_node) in enumerate(nodes.iterrows()):
            if "subj_idx" in bottom_node:
                if str(node_name) == "wfpt":
                    kwargs["title"] = str(subj_i)
                else:
                    kwargs["title"] = str(node_name)

            if parameter_recovery_mode:
                kwargs["node_data"] = model.data.loc[bottom_node["node"].value.index]

            g = plot_func(bottom_node["node"], **kwargs)
            plt.show()

            # Save figure if necessary
            if save:
                if len(tag) == 0:
                    fname = "ppq_subject_" + str(subj_i)
                else:
                    fname = (
                        "ppq_"
                        + ".".join([str(t) for t in tag])
                        + "_subject_"
                        + str(subj_i)
                    )

                if path is None:
                    path = "."
                if isinstance(format, str):
                    format = [format]
                print(["%s.%s" % (os.path.join(path, fname), x) for x in format])
                [
                    g.fig.savefig("%s.%s" % (os.path.join(path, fname), x), format=x)
                    for x in format
                ]


def plot_from_data(
    df,
    generative_model="ddm_hddm_base",
    plot_func=None,
    columns=None,
    save=False,
    save_name=None,
    make_transparent=False,
    show=True,
    path=None,
    groupby="subj_idx",
    figsize=(8, 6),
    format="png",
    keep_frame=True,
    keep_title=True,
    **kwargs,
):
    """Plot data from a hddm ready DataFrame.

    Arguments:

        df : pd.DataFrame
            HDDM ready dataframe.

        value_range : numpy.ndarray
            Array to evaluate the likelihood over.

    Optional:
        columns : int <default=3>
            How many columns to use for plotting the subjects.

        bins : int <default=100>
            How many bins to compute the data histogram over.

        figsize : (int, int) <default=(8, 6)>

        save : bool <default=False>
            Whether to save the figure to a file.

        path : str <default=None>
            Save figure into directory prefix

        format : str or list of strings
            Save figure to a image file of type 'format'. If more then one format is
            given, multiple files are created

        plot_func : function <default=_plot_func_posterior_pdf_node_nn>
            Plotting function to use for each observed node
            (see default function for an example).

    Note:
        This function changes the current value and logp of the nodes.
    """

    # Flip argument names to make the compatible with downstream expectations
    # of the plot_func() function
    if "add_data_model" in kwargs.keys():
        kwargs["add_posterior_mean_model"] = kwargs["add_data_model"]
    if "add_data_rts" in kwargs.keys():
        kwargs["add_posterior_mean_rts"] = kwargs["add_data_rts"]
    if "data_color" in kwargs.keys():
        kwargs["posterior_mean_color"] = kwargs["data_color"]
    else:
        kwargs["posterior_mean_color"] = "blue"

    kwargs["model_"] = generative_model
    title_ = kwargs.pop("title", "")
    ax_title_size = kwargs.pop("ax_title_fontsize", 10)

    if type(groupby) == str:
        groupby = [groupby]

    if plot_func is None:
        plot_func = _plot_func_posterior_pdf_node_nn

    if columns is None:
        # If there are less than 3 items to plot per figure,
        # only use as many columns as there are items.
        max_items = max([len(i[1]) for i in df.groupby(groupby).groups.items()])
        columns = min(3, max_items)

    n_plots = len(df.groupby(groupby))

    # Plot different conditions (new figure for each)
    fig = plt.figure(figsize=figsize)

    if make_transparent:
        fig.patch.set_facecolor("None")
        fig.patch.set_alpha(0.0)

    fig.suptitle(title_, fontsize=12)
    fig.subplots_adjust(top=0.9, hspace=0.4, wspace=0.3)

    i = 1
    for group_id, df_tmp in df.groupby(groupby):
        nrows = int(np.ceil(n_plots / columns))

        # Plot individual subjects (if present)
        ax = fig.add_subplot(nrows, columns, i)

        # Allow kwargs to pass to the plot_func, whether this is the first plot
        # (useful to generate legends only for the first subplot)
        if i == 1:
            kwargs["add_legend"] = True
        else:
            kwargs["add_legend"] = False

        # Make axis title
        tag = ""
        for j in range(len(groupby)):
            tag += groupby[j] + "(" + str(group_id[j]) + ")"
            if j < (len(groupby) - 1):
                tag += "_"
        # print(tag)
        if keep_title:
            ax.set_title(tag, fontsize=ax_title_size)
        # ax.set(frame_on=False)
        if not keep_frame:
            ax.set_axis_off()

        # Call plot function on ax
        # This function should manipulate the ax object, and is expected to not return anything.
        plot_func(df_tmp, ax, **kwargs)
        i += 1

        # Save figure if desired
        if save:
            if save_name is None:
                fname = "ppq_" + prettier_tag(tag)
            else:
                fname = save_name

            if path is None:
                path = "."
            if isinstance(format, str):
                format = [format]
            [
                fig.savefig(
                    "%s.%s" % (os.path.join(path, fname), x),
                    facecolor=fig.get_facecolor(),
                    format=x,
                )
                for x in format
            ]

        # Todo take care of plot closing etc.
        if show:
            pass


def plot_posterior_predictive(
    model,
    plot_func=None,
    required_method="pdf",
    columns=None,
    save=False,
    path=None,
    figsize=(8, 6),
    format="png",
    num_subjs=None,
    parameter_recovery_mode=False,
    subplots_adjust={"top": 0.85, "hspace": 0.4, "wspace": 0.3},
    **kwargs,
):
    """Plot the posterior predictive distribution of a kabuki hierarchical model.

    Arguments:

        model : kabuki.Hierarchical
            The (constructed and sampled) kabuki hierarchical model to
            create the posterior preditive from.

        value_range : numpy.ndarray
            Array to evaluate the likelihood over.

    Optional:

        samples : int <default=10>
            How many posterior samples to generate the posterior predictive over.

        columns : int <default=3>
            How many columns to use for plotting the subjects.

        bins : int <default=100>
            How many bins to compute the data histogram over.

        figsize : (int, int) <default=(8, 6)>

        save : bool <default=False>
            Whether to save the figure to a file.

        path : str <default=None>
            Save figure into directory prefix

        format : str or list of strings
            Save figure to a image file of type 'format'. If more then one format is
            given, multiple files are created

        subplots_adjust : dict <default={'top': 0.85, 'hspace': 0.4, 'wspace': 0.3}>
            Spacing rules for subplot organization. See Matplotlib documentation for details.

        parameter_recovery_mode: bool <default=False>
            If the data attached to the model supplied under the model argument
            has the format expected of the simulator_h_c() function from the simulators.hddm_dataset_generators
            module, then parameter_recovery_mode = True can be use to supply ground truth parameterizations to the
            plot_func argument describes below.

        plot_func : function <default=_plot_func_posterior_pdf_node_nn>
            Plotting function to use for each observed node
            (see default function for an example).

    Note:

        This function changes the current value and logp of the nodes.

    """

    if hasattr(model, "model"):
        kwargs["model_"] = model.model
    else:
        kwargs["model_"] = "ddm_hddm_base"

    if plot_func is None:
        plot_func = _plot_func_posterior_pdf_node_nn

    observeds = model.get_observeds()

    if columns is None:
        # If there are less than 3 items to plot per figure,
        # only use as many columns as there are items.
        max_items = max([len(i[1]) for i in observeds.groupby("tag").groups.items()])
        columns = min(3, max_items)

    # Plot different conditions (new figure for each)
    for tag, nodes in observeds.groupby("tag"):
        fig = plt.figure(figsize=figsize)  # prev utils.pretty_tag
        fig.suptitle(prettier_tag(tag), fontsize=12)
        fig.subplots_adjust(
            top=subplots_adjust["top"],
            hspace=subplots_adjust["hspace"],
            wspace=subplots_adjust["wspace"],
        )

        nrows = num_subjs or int(np.ceil(len(nodes) / columns))

        if len(nodes) - (nrows * columns) > 0:
            nrows += 1

        # Plot individual subjects (if present)
        i = 0
        for subj_i, (node_name, bottom_node) in enumerate(nodes.iterrows()):
            i += 1
            if not hasattr(bottom_node["node"], required_method):
                continue  # skip nodes that do not define the required_method

            ax = fig.add_subplot(nrows, columns, subj_i + 1)
            if "subj_idx" in bottom_node:
                ax.set_title(str(bottom_node["subj_idx"]))

            # Allow kwargs to pass to the plot_func, whether this is the first plot
            # (useful to generate legends only for the first subplot)
            if i == 1:
                kwargs["add_legend"] = True
            else:
                kwargs["add_legend"] = False

            if parameter_recovery_mode:
                kwargs["parameter_recovery_mode"] = True
                kwargs["node_data"] = model.data.loc[bottom_node["node"].value.index]

            # Call plot function on ax
            # This function should manipulate the ax object, and is expected to not return anything.
            plot_func(bottom_node["node"], ax, **kwargs)

            if i > (nrows * columns):
                warnings.warn("Too many nodes. Consider increasing number of columns.")
                break

            if num_subjs is not None and i >= num_subjs:
                break

        # Save figure if necessary
        if save:
            fname = "ppq_" + prettier_tag(tag)  # ".".join(tag)
            if path is None:
                path = "."
            if isinstance(format, str):
                format = [format]
            [
                fig.savefig("%s.%s" % (os.path.join(path, fname), x), format=x)
                for x in format
            ]
        plt.show()


# AXIS MANIPULATORS ---------------
def _plot_func_posterior_pdf_node_nn(
    bottom_node,
    axis,
    value_range=None,
    samples=10,
    bin_size=0.05,
    plot_likelihood_raw=False,
    linewidth=0.5,
    data_color="blue",
    posterior_color="red",
    add_legend=True,
    alpha=0.05,
    **kwargs,
):
    """Calculate posterior predictives from raw likelihood values and plot it on top of a histogram of the real data.
       The function does not define a figure, but manipulates an axis object.


    Arguments:
        bottom_node : pymc.stochastic
            Bottom node to compute posterior over.

        axis : matplotlib.axis
            Axis to plot into.

        value_range : numpy.ndarray
            Range over which to evaluate the likelihood.

    Optional:
        model : str <default='ddm_hddm_base'>
            str that defines the generative model underlying the kabuki model from which the bottom_node
            argument derives.

        samples : int <default=10>
            Number of posterior samples to use.

        bin_size: float <default=0.05>
            Size of bins for the data histogram.

        plot_likelihood_raw : bool <default=False>
            Whether or not to plot likelihoods sample wise.

        add_legend : bool <default=True>
            Whether or not to add a legend to the plot

        linewidth : float <default=0.5>
            Linewidth of histogram outlines.

        data_color : str <default="blue">
            Color of the data part of the plot.

        posterior_color : str <default="red">
            Color of the posterior part of the plot.

    """

    # Setup -----
    color_dict = {
        -1: "black",
        0: "black",
        1: "green",
        2: "blue",
        3: "red",
        4: "orange",
        5: "purple",
        6: "brown",
    }

    model_ = kwargs.pop("model_", "ddm_hddm_base")
    choices = model_config[model_]["choices"]
    n_choices = len(model_config[model_]["choices"])

    bins = np.arange(value_range[0], value_range[-1], bin_size)

    if value_range is None:
        # Infer from data by finding the min and max from the nodes
        raise NotImplementedError("value_range keyword argument must be supplied.")

    if n_choices == 2:
        like = np.empty((samples, len(value_range)), dtype=np.float32)
        pdf_in = value_range
    else:
        like = np.empty((samples, len(value_range), n_choices), dtype=np.float32)
        pdf_in = np.zeros((len(value_range), 2))
        pdf_in[:, 0] = value_range
    # -----

    # Get posterior parameters and plot corresponding likelihoods (if desired) ---
    for sample in range(samples):
        # Get random posterior sample
        _parents_to_random_posterior_sample(bottom_node)

        # Generate likelihood for parents parameters
        if n_choices == 2:
            like[sample, :] = bottom_node.pdf(pdf_in)
            if plot_likelihood_raw:
                axis.plot(
                    value_range,
                    like[sample, :],
                    color=posterior_color,
                    lw=1.0,
                    alpha=alpha,
                )
        else:
            c_cnt = 0
            for choice in choices:
                pdf_in[:, 1] = choice
                like[sample, :, c_cnt] = bottom_node.pdf(pdf_in)
                if plot_likelihood_raw:
                    like[sample, :, c_cnt] = bottom_node.pdf(pdf_in)
                    axis.plot(
                        pdf_in[:, 0],
                        like[sample, :, c_cnt],
                        color=color_dict[choice],
                        lw=1.0,
                        alpha=alpha,
                    )
                c_cnt += 1
    # -------

    # If we don't plot raw likelihoods, we generate a mean likelihood from the samples above
    # and plot it as a line with uncertainty bars
    if not plot_likelihood_raw:
        y = like.mean(axis=0)
        try:
            y_std = like.std(axis=0)
        except FloatingPointError:
            print(
                "WARNING! %s threw FloatingPointError over std computation. Setting to 0 and continuing."
                % bottom_node.__name__
            )
            y_std = np.zeros_like(y)

        if n_choices == 2:
            axis.plot(value_range, y, label="post pred", color=posterior_color)
            axis.fill_between(
                value_range, y - y_std, y + y_std, color=posterior_color, alpha=0.5
            )
        else:
            c_cnt = 0
            for choice in choices:
                axis.plot(
                    value_range,
                    y[:, c_cnt],
                    label="post pred",
                    color=color_dict[choice],
                )
                axis.fill_between(
                    value_range,
                    y[:, c_cnt] - y_std[:, c_cnt],
                    y[:, c_cnt] + y_std[:, c_cnt],
                    color=color_dict[choice],
                    alpha=0.5,
                )
                c_cnt += 1

    # Plot data
    if len(bottom_node.value) != 0:
        if n_choices == 2:
            rt_dat = bottom_node.value.copy()
            if np.sum(rt_dat.rt < 0) == 0:
                rt_dat.loc[rt_dat.response != 1, "rt"] = (-1) * rt_dat.rt[
                    rt_dat.response != 1
                ].values

            axis.hist(
                rt_dat.rt.values,
                density=True,
                color=data_color,
                label="data",
                bins=bins,
                linestyle="-",
                histtype="step",
                lw=linewidth,
            )
        else:
            for choice in choices:
                weights = np.tile(
                    (1 / bin_size) / bottom_node.value.shape[0],
                    reps=bottom_node.value[bottom_node.value.response == choice].shape[
                        0
                    ],
                )
                if np.sum(bottom_node.value.response == choice) > 0:
                    axis.hist(
                        bottom_node.value.rt[bottom_node.value.response == choice],
                        bins=np.arange(value_range[0], value_range[-1], bin_size),
                        weights=weights,
                        color=color_dict[choice],
                        label="data",
                        linestyle="dashed",
                        histtype="step",
                        lw=linewidth,
                    )

    axis.set_ylim(bottom=0)  # Likelihood and histogram can only be positive

    # Add a custom legend
    if add_legend:
        # If two choices only --> show data in blue, posterior samples in black
        if n_choices == 2:
            custom_elems = []
            custom_titles = []
            custom_elems.append(
                Line2D([0], [0], color=data_color, lw=1.0, linestyle="-")
            )
            custom_elems.append(
                Line2D([0], [0], color=posterior_color, lw=1.0, linestyle="-")
            )

            custom_titles.append("Data")
            custom_titles.append("Posterior")
        # If more than two choices --> more styling
        else:
            custom_elems = [
                Line2D([0], [0], color=color_dict[choice], lw=1) for choice in choices
            ]
            custom_titles = ["response: " + str(choice) for choice in choices]
            custom_elems.append(
                Line2D([0], [0], color=posterior_color, lw=1.0, linestyle="dashed")
            )
            custom_elems.append(Line2D([0], [0], color="black", lw=1.0, linestyle="-"))
            custom_titles.append("Data")
            custom_titles.append("Posterior")

        axis.legend(custom_elems, custom_titles, loc="upper right")


def _plot_func_posterior_node_from_sim(
    bottom_node,
    axis,
    value_range=None,
    samples=10,
    bin_size=0.05,
    add_posterior_uncertainty_rts=True,
    add_posterior_mean_rts=True,
    legend_location="upper right",
    legend_fontsize=12,
    legend_shadow=True,
    alpha=0.05,
    linewidth=0.5,
    add_legend=True,
    data_color="blue",
    posterior_mean_color="red",
    posterior_uncertainty_color="black",
    **kwargs,
):
    """Calculate posterior predictive for a certain bottom node and plot a histogram using the supplied axis element.

    :Arguments:
        bottom_node : pymc.stochastic
            Bottom node to compute posterior over.

        axis : matplotlib.axis
            Axis to plot into.

        value_range : numpy.ndarray
            Range over which to evaluate the likelihood.

    :Optional:
        samples : int (default=10)
            Number of posterior samples to use.

        bin_size : int (default=0.05)
            Number of bins to compute histogram over.

        add_posterior_uncertainty_rts: bool (default=True)
            Plot individual posterior samples or not.

        add_posterior_mean_rts: bool (default=True)
            Whether to add a mean posterior (histogram from a dataset collapsed across posterior samples)

        alpha: float (default=0.05)
            alpha (transparency) level for plot elements from single posterior samples.

        linewidth: float (default=0.5)
            linewidth used for histograms

        add_legend: bool (default=True)
            whether or not to add a legend to the current axis.

        legend_loc: str <default='upper right'>
            string defining legend position. Find the rest of the options in the matplotlib documentation.

        legend_shadow: bool <default=True>
            Add shadow to legend box?

        legend_fontsize: float <default=12>
            Fontsize of legend.

        data_color : str <default="blue">
            Color for the data part of the plot.

        posterior_mean_color : str <default="red">
            Color for the posterior mean part of the plot.

        posterior_uncertainty_color : str <default="black">
            Color for the posterior uncertainty part of the plot.

        model_: str (default='lca_no_bias_4')
            string that the defines generative models used (e.g. 'ddm', 'ornstein' etc.).

    """

    color_dict = {
        -1: "black",
        0: "black",
        1: "green",
        2: "blue",
        3: "red",
        4: "orange",
        5: "purple",
        6: "brown",
    }

    if value_range is None:
        # Infer from data by finding the min and max from the nodes
        raise NotImplementedError("value_range keyword argument must be supplied.")
    if len(value_range) == 1:
        value_range = (-value_range[0], value_range[0])
    else:
        value_range = (value_range[0], value_range[-1])

    # Extract some parameters from kwargs
    bins = np.arange(value_range[0], value_range[1], bin_size)

    model_ = kwargs.pop("model_", "lca_no_bias_4")
    choices = model_config[model_]["choices"]
    n_choices = len(model_config[model_]["choices"])

    if type(bottom_node) == pd.DataFrame:
        samples = None
        data_tmp = bottom_node
        data_only = 1
    else:
        samples = _post_pred_generate(
            bottom_node,
            samples=samples,
            data=None,
            append_data=False,
            add_model_parameters=False,
        )
        data_tmp = bottom_node.value
        data_only = 0

    # Go sample by sample (to show uncertainty)
    if add_posterior_uncertainty_rts and not data_only:
        for sample in samples:
            if n_choices == 2:
                if np.sum(sample.rt < 0) == 0:
                    sample.loc[sample.response != 1, "rt"] = (-1) * sample.rt[
                        sample.response != 1
                    ].values

                axis.hist(
                    sample.rt,
                    bins=bins,
                    density=True,
                    color=posterior_uncertainty_color,
                    label="posterior",
                    histtype="step",
                    lw=linewidth,
                    alpha=alpha,
                )

            else:
                for choice in choices:
                    weights = np.tile(
                        (1 / bin_size) / sample.shape[0],
                        reps=sample.loc[sample.response == choice, :].shape[0],
                    )

                    axis.hist(
                        sample.rt[sample.response == choice],
                        bins=bins,
                        weights=weights,
                        color=color_dict[choice],
                        label="posterior",
                        histtype="step",
                        lw=linewidth,
                        alpha=alpha,
                    )

    # Add a 'mean' line
    if add_posterior_mean_rts and not data_only:
        concat_data = pd.concat(samples)

        if n_choices == 2:
            if np.sum(concat_data.rt < 0) == 0:
                concat_data.loc[concat_data.response != 1, "rt"] = (
                    -1
                ) * concat_data.rt[concat_data.response != 1].values

            axis.hist(
                concat_data.rt,
                bins=bins,
                density=True,
                color=posterior_mean_color,
                label="posterior",
                histtype="step",
                lw=linewidth,
                alpha=1.0,
            )
        else:
            for choice in choices:
                weights = np.tile(
                    (1 / bin_size) / concat_data.shape[0],
                    reps=concat_data.loc[concat_data.response == choice, :].shape[0],
                )

                axis.hist(
                    concat_data.rt[concat_data.response == choice],
                    bins=bins,
                    weights=weights,
                    color=color_dict[choice],
                    label="posterior",
                    histtype="step",
                    lw=linewidth,
                    alpha=1.0,
                )

    # Plot data
    if len(data_tmp) != 0:
        if n_choices == 2:
            rt_dat = data_tmp.copy()
            if np.sum(rt_dat.rt < 0) == 0:
                if "response" in rt_dat.columns:
                    rt_dat.loc[rt_dat.response != 1, "rt"] = (-1) * rt_dat.rt[
                        rt_dat.response != 1
                    ].values
                else:
                    pass

            axis.hist(
                rt_dat.rt,
                bins=bins,
                density=True,
                color=data_color,
                label="data",
                linestyle="-",
                histtype="step",
                lw=linewidth,
            )
        else:
            for choice in choices:
                weights = np.tile(
                    (1 / bin_size) / data_tmp.shape[0],
                    reps=data_tmp[data_tmp.response == choice].shape[0],
                )

                axis.hist(
                    data_tmp.rt[data_tmp.response == choice],
                    bins=bins,
                    weights=weights,
                    color=color_dict[choice],
                    label="data",
                    linestyle="dashed",
                    histtype="step",
                    lw=linewidth,
                )

    axis.set_ylim(bottom=0)  # Likelihood and histogram can only be positive

    # Adding legend:
    if add_legend:
        if n_choices == 2:
            custom_elems = []
            custom_titles = []
            custom_elems.append(
                Line2D([0], [0], color=data_color, lw=1.0, linestyle="-")
            )
            custom_elems.append(
                Line2D([0], [0], color=posterior_mean_color, lw=1.0, linestyle="-")
            )
            custom_titles.append("Data")
            custom_titles.append("Posterior")
        else:
            custom_elems = [
                Line2D([0], [0], color=color_dict[choice], lw=1) for choice in choices
            ]
            custom_titles = ["response: " + str(choice) for choice in choices]
            custom_elems.append(
                Line2D([0], [0], color="black", lw=1.0, linestyle="dashed")
            )
            custom_elems.append(Line2D([0], [0], color="black", lw=1.0, linestyle="-"))
            custom_titles.append("Data")
            custom_titles.append("Posterior")
        if not data_only:
            axis.legend(
                custom_elems,
                custom_titles,
                loc=legend_location,
                fontsize=legend_fontsize,
                shadow=legend_shadow,
            )


def _plot_func_model(
    bottom_node,
    axis,
    value_range=None,
    samples=10,
    bin_size=0.05,
    add_data_rts=True,
    add_data_model=True,
    add_data_model_keep_slope=True,
    add_data_model_keep_boundary=True,
    add_data_model_keep_ndt=True,
    add_data_model_keep_starting_point=True,
    add_data_model_markersize_starting_point=50,
    add_data_model_markertype_starting_point=0,
    add_data_model_markershift_starting_point=0,
    add_posterior_uncertainty_model=False,
    add_posterior_uncertainty_rts=False,
    add_posterior_mean_model=True,
    add_posterior_mean_rts=True,
    add_trajectories=False,
    data_label="Data",
    secondary_data=None,
    secondary_data_label=None,
    secondary_data_color="blue",
    linewidth_histogram=0.5,
    linewidth_model=0.5,
    legend_fontsize=12,
    legend_shadow=True,
    legend_location="upper right",
    data_color="blue",
    posterior_mean_color="red",
    posterior_uncertainty_color="black",
    alpha=0.05,
    delta_t_model=0.01,
    add_legend=True,  # keep_frame=False,
    **kwargs,
):
    """Calculate posterior predictive for a certain bottom node.

    Arguments:
        bottom_node: pymc.stochastic
            Bottom node to compute posterior over.

        axis: matplotlib.axis
            Axis to plot into.

        value_range: numpy.ndarray
            Range over which to evaluate the likelihood.

    Optional:
        samples: int <default=10>
            Number of posterior samples to use.

        bin_size: float <default=0.05>
            Size of bins used for histograms

        alpha: float <default=0.05>
            alpha (transparency) level for the sample-wise elements of the plot

        add_data_rts: bool <default=True>
            Add data histogram of rts ?

        add_data_model: bool <default=True>
            Add model cartoon for data

        add_posterior_uncertainty_rts: bool <default=True>
            Add sample by sample histograms?

        add_posterior_mean_rts: bool <default=True>
            Add a mean posterior?

        add_model: bool <default=True>
            Whether to add model cartoons to the plot.

        linewidth_histogram: float <default=0.5>
            linewdith of histrogram plot elements.

        linewidth_model: float <default=0.5>
            linewidth of plot elements concerning the model cartoons.

        legend_location: str <default='upper right'>
            string defining legend position. Find the rest of the options in the matplotlib documentation.

        legend_shadow: bool <default=True>
            Add shadow to legend box?

        legend_fontsize: float <default=12>
            Fontsize of legend.

        data_color : str <default="blue">
            Color for the data part of the plot.

        posterior_mean_color : str <default="red">
            Color for the posterior mean part of the plot.

        posterior_uncertainty_color : str <default="black">
            Color for the posterior uncertainty part of the plot.

        delta_t_model:
            specifies plotting intervals for model cartoon elements of the graphs.
    """

    # AF-TODO: Add a mean version of this!
    if value_range is None:
        # Infer from data by finding the min and max from the nodes
        raise NotImplementedError("value_range keyword argument must be supplied.")

    if len(value_range) > 2:
        value_range = (value_range[0], value_range[-1])

    # Extract some parameters from kwargs
    bins = np.arange(value_range[0], value_range[-1], bin_size)

    # If bottom_node is a DataFrame we know that we are just plotting real data
    if type(bottom_node) == pd.DataFrame:
        samples_tmp = [bottom_node]
        data_tmp = None
    else:
        samples_tmp = _post_pred_generate(
            bottom_node,
            samples=samples,
            data=None,
            append_data=False,
            add_model_parameters=True,
        )
        data_tmp = bottom_node.value.copy()

    # Relevant for recovery mode
    node_data_full = kwargs.pop("node_data", None)

    tmp_model = kwargs.pop("model_", "angle")
    if len(model_config[tmp_model]["choices"]) > 2:
        raise ValueError("The model plot works only for 2 choice models at the moment")

    # ---------------------------

    ylim = kwargs.pop("ylim", 3)
    hist_bottom = kwargs.pop("hist_bottom", 2)
    hist_histtype = kwargs.pop("hist_histtype", "step")

    if ("ylim_high" in kwargs) and ("ylim_low" in kwargs):
        ylim_high = kwargs["ylim_high"]
        ylim_low = kwargs["ylim_low"]
    else:
        ylim_high = ylim
        ylim_low = -ylim

    if ("hist_bottom_high" in kwargs) and ("hist_bottom_low" in kwargs):
        hist_bottom_high = kwargs["hist_bottom_high"]
        hist_bottom_low = kwargs["hist_bottom_low"]
    else:
        hist_bottom_high = hist_bottom
        hist_bottom_low = hist_bottom

    axis.set_xlim(value_range[0], value_range[-1])
    axis.set_ylim(ylim_low, ylim_high)
    axis_twin_up = axis.twinx()
    axis_twin_down = axis.twinx()
    axis_twin_up.set_ylim(ylim_low, ylim_high)
    axis_twin_up.set_yticks([])
    axis_twin_down.set_ylim(ylim_high, ylim_low)
    axis_twin_down.set_yticks([])
    axis_twin_down.set_axis_off()
    axis_twin_up.set_axis_off()

    # ADD HISTOGRAMS
    # -------------------------------
    # POSTERIOR BASED HISTOGRAM
    if add_posterior_uncertainty_rts:  # add_uc_rts:
        j = 0
        for sample in samples_tmp:
            tmp_label = None

            if add_legend and j == 0:
                tmp_label = "PostPred"

            weights_up = np.tile(
                (1 / bin_size) / sample.shape[0],
                reps=sample.loc[sample.response == 1, :].shape[0],
            )
            weights_down = np.tile(
                (1 / bin_size) / sample.shape[0],
                reps=sample.loc[(sample.response != 1), :].shape[0],
            )

            axis_twin_up.hist(
                np.abs(sample.rt[sample.response == 1]),
                bins=bins,
                weights=weights_up,
                histtype=hist_histtype,
                bottom=hist_bottom_high,
                alpha=alpha,
                color=posterior_uncertainty_color,
                edgecolor=posterior_uncertainty_color,
                zorder=-1,
                label=tmp_label,
                linewidth=linewidth_histogram,
            )

            axis_twin_down.hist(
                np.abs(sample.loc[(sample.response != 1), :].rt),
                bins=bins,
                weights=weights_down,
                histtype=hist_histtype,
                bottom=hist_bottom_low,
                alpha=alpha,
                color=posterior_uncertainty_color,
                edgecolor=posterior_uncertainty_color,
                linewidth=linewidth_histogram,
                zorder=-1,
            )
            j += 1

    if add_posterior_mean_rts:  # add_mean_rts:
        concat_data = pd.concat(samples_tmp)
        tmp_label = None

        if add_legend:
            tmp_label = "PostPred Mean"

        weights_up = np.tile(
            (1 / bin_size) / concat_data.shape[0],
            reps=concat_data.loc[concat_data.response == 1, :].shape[0],
        )
        weights_down = np.tile(
            (1 / bin_size) / concat_data.shape[0],
            reps=concat_data.loc[(concat_data.response != 1), :].shape[0],
        )

        axis_twin_up.hist(
            np.abs(concat_data.rt[concat_data.response == 1]),
            bins=bins,
            weights=weights_up,
            histtype=hist_histtype,
            bottom=hist_bottom_high,
            alpha=1.0,
            color=posterior_mean_color,
            edgecolor=posterior_mean_color,
            zorder=-1,
            label=tmp_label,
            linewidth=linewidth_histogram,
        )

        axis_twin_down.hist(
            np.abs(concat_data.loc[(concat_data.response != 1), :].rt),
            bins=bins,
            weights=weights_down,
            histtype=hist_histtype,
            bottom=hist_bottom_low,
            alpha=1.0,
            color=posterior_mean_color,
            edgecolor=posterior_mean_color,
            linewidth=linewidth_histogram,
            zorder=-1,
        )

    # DATA HISTOGRAM
    if (data_tmp is not None) and add_data_rts:
        tmp_label = None
        if add_legend:
            tmp_label = data_label

        weights_up = np.tile(
            (1 / bin_size) / data_tmp.shape[0],
            reps=data_tmp[data_tmp.response == 1].shape[0],
        )
        weights_down = np.tile(
            (1 / bin_size) / data_tmp.shape[0],
            reps=data_tmp[(data_tmp.response != 1)].shape[0],
        )

        axis_twin_up.hist(
            np.abs(data_tmp[data_tmp.response == 1].rt),
            bins=bins,
            weights=weights_up,
            histtype=hist_histtype,
            bottom=hist_bottom_high,
            alpha=1,
            color=data_color,
            edgecolor=data_color,
            label=tmp_label,
            zorder=-1,
            linewidth=linewidth_histogram,
        )

        axis_twin_down.hist(
            np.abs(data_tmp[(data_tmp.response != 1)].rt),
            bins=bins,
            weights=weights_down,
            histtype=hist_histtype,
            bottom=hist_bottom_low,
            alpha=1,
            color=data_color,
            edgecolor=data_color,
            linewidth=linewidth_histogram,
            zorder=-1,
        )

    # SECONDARY DATA HISTOGRAM
    if secondary_data is not None:
        tmp_label = None
        if add_legend:
            if secondary_data_label is not None:
                tmp_label = secondary_data_label

        weights_up = np.tile(
            (1 / bin_size) / secondary_data.shape[0],
            reps=secondary_data[secondary_data.response == 1].shape[0],
        )
        weights_down = np.tile(
            (1 / bin_size) / secondary_data.shape[0],
            reps=secondary_data[(secondary_data.response != 1)].shape[0],
        )

        axis_twin_up.hist(
            np.abs(secondary_data[secondary_data.response == 1].rt),
            bins=bins,
            weights=weights_up,
            histtype=hist_histtype,
            bottom=hist_bottom_high,
            alpha=1,
            color=secondary_data_color,
            edgecolor=secondary_data_color,
            label=tmp_label,
            zorder=-100,
            linewidth=linewidth_histogram,
        )

        axis_twin_down.hist(
            np.abs(secondary_data[(secondary_data.response != 1)].rt),
            bins=bins,
            weights=weights_down,
            histtype=hist_histtype,
            bottom=hist_bottom_low,
            alpha=1,
            color=secondary_data_color,
            edgecolor=secondary_data_color,
            linewidth=linewidth_histogram,
            zorder=-100,
        )
    # -------------------------------

    if add_legend:
        if data_tmp is not None:
            axis_twin_up.legend(
                fontsize=legend_fontsize, shadow=legend_shadow, loc=legend_location
            )

    # ADD MODEL:
    j = 0
    t_s = np.arange(0, value_range[-1], delta_t_model)

    # MAKE BOUNDS (FROM MODEL CONFIG) !
    if add_posterior_uncertainty_model:  # add_uc_model:
        for sample in samples_tmp:
            _add_model_cartoon_to_ax(
                sample=sample,
                axis=axis,
                tmp_model=tmp_model,
                keep_slope=add_data_model_keep_slope,
                keep_boundary=add_data_model_keep_boundary,
                keep_ndt=add_data_model_keep_ndt,
                keep_starting_point=add_data_model_keep_starting_point,
                markersize_starting_point=add_data_model_markersize_starting_point,
                markertype_starting_point=add_data_model_markertype_starting_point,
                markershift_starting_point=add_data_model_markershift_starting_point,
                delta_t_graph=delta_t_model,
                sample_hist_alpha=alpha,
                lw_m=linewidth_model,
                tmp_label=tmp_label,
                ylim_low=ylim_low,
                ylim_high=ylim_high,
                t_s=t_s,
                color=posterior_uncertainty_color,
                zorder_cnt=j,
            )

    if (node_data_full is not None) and add_data_model:
        _add_model_cartoon_to_ax(
            sample=node_data_full,
            axis=axis,
            tmp_model=tmp_model,
            keep_slope=add_data_model_keep_slope,
            keep_boundary=add_data_model_keep_boundary,
            keep_ndt=add_data_model_keep_ndt,
            keep_starting_point=add_data_model_keep_starting_point,
            markersize_starting_point=add_data_model_markersize_starting_point,
            markertype_starting_point=add_data_model_markertype_starting_point,
            markershift_starting_point=add_data_model_markershift_starting_point,
            delta_t_graph=delta_t_model,
            sample_hist_alpha=1.0,
            lw_m=linewidth_model + 0.5,
            tmp_label=None,
            ylim_low=ylim_low,
            ylim_high=ylim_high,
            t_s=t_s,
            color=data_color,
            zorder_cnt=j + 1,
        )

    if add_posterior_mean_model:  # add_mean_model:
        tmp_label = None
        if add_legend:
            tmp_label = "PostPred Mean"

        _add_model_cartoon_to_ax(
            sample=pd.DataFrame(pd.concat(samples_tmp).mean().astype(np.float32)).T,
            axis=axis,
            tmp_model=tmp_model,
            keep_slope=add_data_model_keep_slope,
            keep_boundary=add_data_model_keep_boundary,
            keep_ndt=add_data_model_keep_ndt,
            keep_starting_point=add_data_model_keep_starting_point,
            markersize_starting_point=add_data_model_markersize_starting_point,
            markertype_starting_point=add_data_model_markertype_starting_point,
            markershift_starting_point=add_data_model_markershift_starting_point,
            delta_t_graph=delta_t_model,
            sample_hist_alpha=1.0,
            lw_m=linewidth_model + 0.5,
            tmp_label=None,
            ylim_low=ylim_low,
            ylim_high=ylim_high,
            t_s=t_s,
            color=posterior_mean_color,
            zorder_cnt=j + 2,
        )

    if add_trajectories:
        _add_trajectories(
            axis=axis,
            sample=samples_tmp[0],
            tmp_model=tmp_model,
            t_s=t_s,
            delta_t_graph=delta_t_model,
            **kwargs,
        )


# AF-TODO: Add documentation for this function
def _add_trajectories(
    axis=None,
    sample=None,
    t_s=None,
    delta_t_graph=0.01,
    tmp_model=None,
    n_trajectories=10,
    supplied_trajectory=None,
    maxid_supplied_trajectory=1,  # useful for gifs
    highlight_trajectory_rt_choice=True,
    markersize_trajectory_rt_choice=50,
    markertype_trajectory_rt_choice="*",
    markercolor_trajectory_rt_choice="red",
    linewidth_trajectories=1,
    alpha_trajectories=0.5,
    color_trajectories="black",
    **kwargs,
):
    # Check markercolor type
    if type(markercolor_trajectory_rt_choice) == str:
        markercolor_trajectory_rt_choice_dict = {}
        for value_ in model_config[tmp_model]["choices"]:
            markercolor_trajectory_rt_choice_dict[
                value_
            ] = markercolor_trajectory_rt_choice
    elif type(markercolor_trajectory_rt_choice) == list:
        cnt = 0
        for value_ in model_config[tmp_model]["choices"]:
            markercolor_trajectory_rt_choice_dict[
                value_
            ] = markercolor_trajectory_rt_choice[cnt]
            cnt += 1
    elif type(markercolor_trajectory_rt_choice) == dict:
        markercolor_trajectory_rt_choice_dict = markercolor_trajectory_rt_choice
    else:
        pass

    # Check trajectory color type
    if type(color_trajectories) == str:
        color_trajectories_dict = {}
        for value_ in model_config[tmp_model]["choices"]:
            color_trajectories_dict[value_] = color_trajectories
    elif type(color_trajectories) == list:
        cnt = 0
        for value_ in model_config[tmp_model]["choices"]:
            color_trajectories_dict[value_] = color_trajectories[cnt]
            cnt += 1
    elif type(color_trajectories) == dict:
        color_trajectories_dict = color_trajectories
    else:
        pass

    # Make bounds
    (b_low, b_high) = _make_bounds(
        tmp_model=tmp_model,
        sample=sample,
        delta_t_graph=delta_t_graph,
        t_s=t_s,
        return_shifted_by_ndt=False,
    )

    # Trajectories
    if supplied_trajectory is None:
        for i in range(n_trajectories):
            rand_int = np.random.choice(400000000)
            out_traj = simulator(
                theta=sample[model_config[tmp_model]["params"]].values[0],
                model=tmp_model,
                n_samples=1,
                no_noise=False,
                delta_t=delta_t_graph,
                bin_dim=None,
                random_state=rand_int,
            )

            tmp_traj = out_traj[2]["trajectory"]
            tmp_traj_choice = float(out_traj[1].flatten())
            maxid = np.minimum(np.argmax(np.where(tmp_traj > -999)), t_s.shape[0])

            # Identify boundary value at timepoint of crossing
            b_tmp = b_high[maxid] if tmp_traj_choice > 0 else b_low[maxid]

            axis.plot(
                t_s[:maxid] + sample.t.values[0],
                tmp_traj[:maxid],
                color=color_trajectories_dict[tmp_traj_choice],
                alpha=alpha_trajectories,
                linewidth=linewidth_trajectories,
                zorder=2000 + i,
            )

            if highlight_trajectory_rt_choice:
                axis.scatter(
                    t_s[maxid] + sample.t.values[0],
                    b_tmp,
                    # tmp_traj[maxid],
                    markersize_trajectory_rt_choice,
                    color=markercolor_trajectory_rt_choice_dict[tmp_traj_choice],
                    alpha=1,
                    marker=markertype_trajectory_rt_choice,
                    zorder=2000 + i,
                )

    else:
        if len(supplied_trajectory["trajectories"].shape) == 1:
            supplied_trajectory["trajectories"] = np.expand_dims(
                supplied_trajectory["trajectories"], axis=0
            )

        for j in range(supplied_trajectory["trajectories"].shape[0]):
            maxid = np.minimum(
                np.argmax(np.where(supplied_trajectory["trajectories"][j, :] > -999)),
                t_s.shape[0],
            )
            if j == (supplied_trajectory["trajectories"].shape[0] - 1):
                maxid_traj = min(maxid, maxid_supplied_trajectory)
            else:
                maxid_traj = maxid

            axis.plot(
                t_s[:maxid_traj] + sample.t.values[0],
                supplied_trajectory["trajectories"][j, :maxid_traj],
                color=color_trajectories_dict[
                    supplied_trajectory["trajectory_choices"][j]
                ],  # color_trajectories,
                alpha=alpha_trajectories,
                linewidth=linewidth_trajectories,
                zorder=2000 + j,
            )

            # Identify boundary value at timepoint of crossing
            b_tmp = (
                b_high[maxid_traj]
                if supplied_trajectory["trajectory_choices"][j] > 0
                else b_low[maxid_traj]
            )

            if maxid_traj == maxid:
                if highlight_trajectory_rt_choice:
                    axis.scatter(
                        t_s[maxid_traj] + sample.t.values[0],
                        b_tmp,
                        # supplied_trajectory['trajectories'][j, maxid_traj],
                        markersize_trajectory_rt_choice,
                        color=markercolor_trajectory_rt_choice_dict[
                            supplied_trajectory["trajectory_choices"][j]
                        ],  # markercolor_trajectory_rt_choice,
                        alpha=1,
                        marker=markertype_trajectory_rt_choice,
                        zorder=2000 + j,
                    )


# AF-TODO: Add documentation to this function
def _add_model_cartoon_to_ax(
    sample=None,
    axis=None,
    tmp_model=None,
    keep_slope=True,
    keep_boundary=True,
    keep_ndt=True,
    keep_starting_point=True,
    markersize_starting_point=80,
    markertype_starting_point=1,
    markershift_starting_point=-0.05,
    delta_t_graph=None,
    sample_hist_alpha=None,
    lw_m=None,
    tmp_label=None,
    ylim_low=None,
    ylim_high=None,
    t_s=None,
    zorder_cnt=1,
    color="black",
):
    # Make bounds
    b_low, b_high = _make_bounds(
        tmp_model=tmp_model,
        sample=sample,
        delta_t_graph=delta_t_graph,
        t_s=t_s,
        return_shifted_by_ndt=True,
    )

    # MAKE SLOPES (VIA TRAJECTORIES HERE --> RUN NOISE FREE SIMULATIONS)!
    out = simulator(
        theta=sample[model_config[tmp_model]["params"]].values[0],
        model=tmp_model,
        n_samples=1,
        no_noise=True,
        delta_t=delta_t_graph,
        bin_dim=None,
    )

    tmp_traj = out[2]["trajectory"]
    maxid = np.minimum(np.argmax(np.where(tmp_traj > -999)), t_s.shape[0])

    if "hddm_base" in tmp_model:
        a_tmp = sample.a.values[0] / 2
        tmp_traj = tmp_traj - a_tmp

    if keep_boundary:
        # Upper bound
        axis.plot(
            t_s,  # + sample.t.values[0],
            b_high,
            color=color,
            alpha=sample_hist_alpha,
            zorder=1000 + zorder_cnt,
            linewidth=lw_m,
            label=tmp_label,
        )

        # Lower bound
        axis.plot(
            t_s,  # + sample.t.values[0],
            b_low,
            color=color,
            alpha=sample_hist_alpha,
            zorder=1000 + zorder_cnt,
            linewidth=lw_m,
        )

    # Slope
    if keep_slope:
        axis.plot(
            t_s[:maxid] + sample.t.values[0],
            tmp_traj[:maxid],
            color=color,
            alpha=sample_hist_alpha,
            zorder=1000 + zorder_cnt,
            linewidth=lw_m,
        )  # TOOK AWAY LABEL

    # Non-decision time
    if keep_ndt:
        axis.axvline(
            x=sample.t.values[0],
            ymin=ylim_low,
            ymax=ylim_high,
            color=color,
            linestyle="--",
            linewidth=lw_m,
            zorder=1000 + zorder_cnt,
            alpha=sample_hist_alpha,
        )
    # Starting point
    if keep_starting_point:
        axis.scatter(
            sample.t.values[0] + markershift_starting_point,
            b_low[0] + (sample.z.values[0] * (b_high[0] - b_low[0])),
            markersize_starting_point,
            marker=markertype_starting_point,
            color=color,
            alpha=sample_hist_alpha,
            zorder=1000 + zorder_cnt,
        )


def _make_bounds(
    tmp_model=None,
    sample=None,
    delta_t_graph=None,
    t_s=None,
    return_shifted_by_ndt=True,
):
    # MULTIPLICATIVE BOUND
    if tmp_model == "weibull" or tmp_model == "weibull_cdf":
        b = np.maximum(
            sample.a.values[0]
            * model_config[tmp_model]["boundary"](
                t=t_s, alpha=sample.alpha.values[0], beta=sample.beta.values[0]
            ),
            0,
        )

        # Move boundary forward by the non-decision time
        b_raw_high = deepcopy(b)
        b_raw_low = deepcopy(-b)
        b_init_val = b[0]
        t_shift = np.arange(0, sample.t.values[0], delta_t_graph).shape[0]
        b = np.roll(b, t_shift)
        b[:t_shift] = b_init_val

    # ADDITIVE BOUND
    elif tmp_model == "angle":
        b = np.maximum(
            sample.a.values[0]
            + model_config[tmp_model]["boundary"](t=t_s, theta=sample.theta.values[0]),
            0,
        )

        b_raw_high = deepcopy(b)
        b_raw_low = deepcopy(-b)
        # Move boundary forward by the non-decision time
        b_init_val = b[0]
        t_shift = np.arange(0, sample.t.values[0], delta_t_graph).shape[0]
        b = np.roll(b, t_shift)
        b[:t_shift] = b_init_val

    # CONSTANT BOUND
    elif (
        tmp_model == "ddm"
        or tmp_model == "ornstein"
        or tmp_model == "levy"
        or tmp_model == "full_ddm"
        or tmp_model == "ddm_hddm_base"
        or tmp_model == "full_ddm_hddm_base"
    ):
        b = sample.a.values[0] * np.ones(t_s.shape[0])

        if "hddm_base" in tmp_model:
            b = (sample.a.values[0] / 2) * np.ones(t_s.shape[0])

        b_raw_high = b
        b_raw_low = -b

    # Separate out upper and lower bound:
    b_high = b
    b_low = -b

    if return_shifted_by_ndt:
        return (b_low, b_high)
    else:
        return (b_raw_low, b_raw_high)


def _plot_func_model_n(
    bottom_node,
    axis,
    value_range=None,
    samples=10,
    bin_size=0.05,
    add_posterior_uncertainty_model=False,
    add_posterior_uncertainty_rts=False,
    add_posterior_mean_model=True,
    add_posterior_mean_rts=True,
    linewidth_histogram=0.5,
    linewidth_model=0.5,
    legend_fontsize=7,
    legend_shadow=True,
    legend_location="upper right",
    delta_t_model=0.01,
    add_legend=True,
    alpha=0.01,
    keep_frame=False,
    **kwargs,
):
    """Calculate posterior predictive for a certain bottom node.

    Arguments:
        bottom_node: pymc.stochastic
            Bottom node to compute posterior over.

        axis: matplotlib.axis
            Axis to plot into.

        value_range: numpy.ndarray
            Range over which to evaluate the likelihood.

    Optional:
        samples: int <default=10>
            Number of posterior samples to use.

        bin_size: float <default=0.05>
            Size of bins used for histograms

        alpha: float <default=0.05>
            alpha (transparency) level for the sample-wise elements of the plot

        add_posterior_uncertainty_rts: bool <default=True>
            Add sample by sample histograms?

        add_posterior_mean_rts: bool <default=True>
            Add a mean posterior?

        add_model: bool <default=True>
            Whether to add model cartoons to the plot.

        linewidth_histogram: float <default=0.5>
            linewdith of histrogram plot elements.

        linewidth_model: float <default=0.5>
            linewidth of plot elements concerning the model cartoons.

        legend_loc: str <default='upper right'>
            string defining legend position. Find the rest of the options in the matplotlib documentation.

        legend_shadow: bool <default=True>
            Add shadow to legend box?

        legend_fontsize: float <default=12>
            Fontsize of legend.

        data_color : str <default="blue">
            Color for the data part of the plot.

        posterior_mean_color : str <default="red">
            Color for the posterior mean part of the plot.

        posterior_uncertainty_color : str <default="black">
            Color for the posterior uncertainty part of the plot.


        delta_t_model:
            specifies plotting intervals for model cartoon elements of the graphs.
    """

    color_dict = {
        -1: "black",
        0: "black",
        1: "green",
        2: "blue",
        3: "red",
        4: "orange",
        5: "purple",
        6: "brown",
    }

    # AF-TODO: Add a mean version of this !
    if value_range is None:
        # Infer from data by finding the min and max from the nodes
        raise NotImplementedError("value_range keyword argument must be supplied.")

    if len(value_range) > 2:
        value_range = (value_range[0], value_range[-1])

    # Extract some parameters from kwargs
    bins = np.arange(value_range[0], value_range[-1], bin_size)

    # Relevant for recovery mode
    node_data_full = kwargs.pop("node_data", None)
    tmp_model = kwargs.pop("model_", "angle")

    bottom = 0
    # ------------
    ylim = kwargs.pop("ylim", 3)

    choices = model_config[tmp_model]["choices"]

    # If bottom_node is a DataFrame we know that we are just plotting real data
    if type(bottom_node) == pd.DataFrame:
        samples_tmp = [bottom_node]
        data_tmp = None
    else:
        samples_tmp = _post_pred_generate(
            bottom_node,
            samples=samples,
            data=None,
            append_data=False,
            add_model_parameters=True,
        )
        data_tmp = bottom_node.value.copy()

    axis.set_xlim(value_range[0], value_range[-1])
    axis.set_ylim(0, ylim)

    # ADD MODEL:
    j = 0
    t_s = np.arange(0, value_range[-1], delta_t_model)

    # # MAKE BOUNDS (FROM MODEL CONFIG) !
    if add_posterior_uncertainty_model:  # add_uc_model:
        for sample in samples_tmp:
            tmp_label = None

            if add_legend and (j == 0):
                tmp_label = "PostPred"

            _add_model_n_cartoon_to_ax(
                sample=sample,
                axis=axis,
                tmp_model=tmp_model,
                delta_t_graph=delta_t_model,
                sample_hist_alpha=alpha,
                lw_m=linewidth_model,
                tmp_label=tmp_label,
                linestyle="-",
                ylim=ylim,
                t_s=t_s,
                color_dict=color_dict,
                zorder_cnt=j,
            )

            j += 1

    if add_posterior_mean_model:  # add_mean_model:
        tmp_label = None
        if add_legend:
            tmp_label = "PostPred Mean"

        bottom = _add_model_n_cartoon_to_ax(
            sample=pd.DataFrame(pd.concat(samples_tmp).mean().astype(np.float32)).T,
            axis=axis,
            tmp_model=tmp_model,
            delta_t_graph=delta_t_model,
            sample_hist_alpha=1.0,
            lw_m=linewidth_model + 0.5,
            linestyle="-",
            tmp_label=None,
            ylim=ylim,
            t_s=t_s,
            color_dict=color_dict,
            zorder_cnt=j + 2,
        )

    if node_data_full is not None:
        _add_model_n_cartoon_to_ax(
            sample=node_data_full,
            axis=axis,
            tmp_model=tmp_model,
            delta_t_graph=delta_t_model,
            sample_hist_alpha=1.0,
            lw_m=linewidth_model + 0.5,
            linestyle="dashed",
            tmp_label=None,
            ylim=ylim,
            t_s=t_s,
            color_dict=color_dict,
            zorder_cnt=j + 1,
        )

    # ADD HISTOGRAMS
    # -------------------------------

    # POSTERIOR BASED HISTOGRAM
    if add_posterior_uncertainty_rts:  # add_uc_rts:
        j = 0
        for sample in samples_tmp:
            for choice in choices:
                tmp_label = None

                if add_legend and j == 0:
                    tmp_label = "PostPred"

                weights = np.tile(
                    (1 / bin_size) / sample.shape[0],
                    reps=sample.loc[sample.response == choice, :].shape[0],
                )

                axis.hist(
                    np.abs(sample.rt[sample.response == choice]),
                    bins=bins,
                    bottom=bottom,
                    weights=weights,
                    histtype="step",
                    alpha=alpha,
                    color=color_dict[choice],
                    zorder=-1,
                    label=tmp_label,
                    linewidth=linewidth_histogram,
                )
                j += 1

    if add_posterior_mean_rts:
        concat_data = pd.concat(samples_tmp)
        for choice in choices:
            tmp_label = None
            if add_legend and (choice == choices[0]):
                tmp_label = "PostPred Mean"

            weights = np.tile(
                (1 / bin_size) / concat_data.shape[0],
                reps=concat_data.loc[concat_data.response == choice, :].shape[0],
            )

            axis.hist(
                np.abs(concat_data.rt[concat_data.response == choice]),
                bins=bins,
                bottom=bottom,
                weights=weights,
                histtype="step",
                alpha=1.0,
                color=color_dict[choice],
                zorder=-1,
                label=tmp_label,
                linewidth=linewidth_histogram,
            )

    # DATA HISTOGRAM
    if data_tmp is not None:
        for choice in choices:
            tmp_label = None
            if add_legend and (choice == choices[0]):
                tmp_label = "Data"

            weights = np.tile(
                (1 / bin_size) / data_tmp.shape[0],
                reps=data_tmp.loc[data_tmp.response == choice, :].shape[0],
            )

            axis.hist(
                np.abs(data_tmp.rt[data_tmp.response == choice]),
                bins=bins,
                bottom=bottom,
                weights=weights,
                histtype="step",
                linestyle="dashed",
                alpha=1.0,
                color=color_dict[choice],
                edgecolor=color_dict[choice],
                zorder=-1,
                label=tmp_label,
                linewidth=linewidth_histogram,
            )
    # -------------------------------

    if add_legend:
        if data_tmp is not None:
            custom_elems = [
                Line2D([0], [0], color=color_dict[choice], lw=1) for choice in choices
            ]
            custom_titles = ["response: " + str(choice) for choice in choices]

            custom_elems.append(
                Line2D([0], [0], color="black", lw=1.0, linestyle="dashed")
            )
            custom_elems.append(Line2D([0], [0], color="black", lw=1.0, linestyle="-"))
            custom_titles.append("Data")
            custom_titles.append("Posterior")

            axis.legend(
                custom_elems,
                custom_titles,
                fontsize=legend_fontsize,
                shadow=legend_shadow,
                loc=legend_location,
            )

    # FRAME
    if not keep_frame:
        axis.set_frame_on(False)


def _add_model_n_cartoon_to_ax(
    sample=None,
    axis=None,
    tmp_model=None,
    delta_t_graph=None,
    sample_hist_alpha=None,
    lw_m=None,
    linestyle="-",
    tmp_label=None,
    ylim=None,
    t_s=None,
    zorder_cnt=1,
    color_dict=None,
):
    if "weibull" in tmp_model:
        b = np.maximum(
            sample.a.values[0]
            * model_config[tmp_model]["boundary"](
                t=t_s, alpha=sample.alpha.values[0], beta=sample.beta.values[0]
            ),
            0,
        )

    elif "angle" in tmp_model:
        b = np.maximum(
            sample.a.values[0]
            + model_config[tmp_model]["boundary"](t=t_s, theta=sample.theta.values[0]),
            0,
        )

    else:
        b = sample.a.values[0] * np.ones(t_s.shape[0])

    # Upper bound
    axis.plot(
        t_s + sample.t.values[0],
        b,
        color="black",
        alpha=sample_hist_alpha,
        zorder=1000 + zorder_cnt,
        linewidth=lw_m,
        linestyle=linestyle,
        label=tmp_label,
    )

    # Starting point
    axis.axvline(
        x=sample.t.values[0],
        ymin=-ylim,
        ymax=ylim,
        color="black",
        linestyle=linestyle,
        linewidth=lw_m,
        alpha=sample_hist_alpha,
    )

    # # MAKE SLOPES (VIA TRAJECTORIES HERE --> RUN NOISE FREE SIMULATIONS)!
    out = simulator(
        theta=sample[model_config[tmp_model]["params"]].values[0],
        model=tmp_model,
        n_samples=1,
        no_noise=True,
        delta_t=delta_t_graph,
        bin_dim=None,
    )

    # # AF-TODO: Add trajectories
    tmp_traj = out[2]["trajectory"]

    for i in range(len(model_config[tmp_model]["choices"])):
        tmp_maxid = np.minimum(np.argmax(np.where(tmp_traj[:, i] > -999)), t_s.shape[0])

        # Slope
        axis.plot(
            t_s[:tmp_maxid] + sample.t.values[0],
            tmp_traj[:tmp_maxid, i],
            color=color_dict[i],
            linestyle=linestyle,
            alpha=sample_hist_alpha,
            zorder=1000 + zorder_cnt,
            linewidth=lw_m,
        )  # TOOK AWAY LABEL

    return b[0]


def _plot_func_pair(
    bottom_node,
    model_="ddm_hddm_base",
    n_samples=200,
    figsize=(8, 6),
    title="",
    **kwargs,
):
    """Generates a posterior pair plot for a given kabuki node.

    Arguments:

        bottom_node: kabuki_node
            Observed node of a kabuki.Hierarchical model.

        n_samples: int <default=200>
            Number of posterior samples to consider for the plot.

        figsize: (int, int) <default=(8,6)>
            Size of the figure in inches.

        title: str <default=''>
            Plot title.

        model: str <default='ddm_hddm_base'>

    """

    params = model_config[model_]["params"]
    parent_keys = bottom_node.parents.value.keys()
    param_intersection = set(params).intersection(set(parent_keys))
    df = pd.DataFrame(
        np.empty((n_samples, len(param_intersection))), columns=param_intersection
    )
    node_data_full = kwargs.pop("node_data", None)

    for i in range(n_samples):
        _parents_to_random_posterior_sample(bottom_node)
        for key_tmp in param_intersection:
            df.loc[i, key_tmp] = bottom_node.parents.value[key_tmp]

    sns.set_theme(
        style="ticks", color_codes=True
    )  # , rc = {'figure.figsize': figsize})

    g = sns.PairGrid(data=df, corner=True)
    g.fig.set_size_inches(figsize[0], figsize[1])
    g = g.map_diag(plt.hist, histtype="step", color="black", alpha=0.8)
    g = g.map_lower(sns.kdeplot, cmap="Reds")

    # Adding ground truth if calling function was in parameter recovery mode
    if node_data_full is not None:
        for i in range(1, g.axes.shape[0], 1):
            for j in range(0, i, 1):
                tmp_y_label = g.axes[g.axes.shape[0] - 1, i].get_xlabel()
                tmp_x_label = g.axes[g.axes.shape[0] - 1, j].get_xlabel()
                g.axes[i, j].scatter(
                    node_data_full[tmp_x_label].values[0],
                    node_data_full[tmp_y_label].values[0],
                    color="blue",
                    marker="+",
                    s=100,
                    zorder=1000,
                )

        # Adding ground truth to axes !
        for i in range(g.axes.shape[0]):
            if i == 0:
                y_lims_tmp = g.axes[i, i].get_ylim()
                g.axes[i, i].set_ylim(0, y_lims_tmp[1])

            tmp_x_label = g.axes[g.axes.shape[0] - 1, i].get_xlabel()
            g.axes[i, i].scatter(
                node_data_full[tmp_x_label].values[0],
                g.axes[i, i].get_ylim()[0],
                color="blue",
                marker="|",
                s=100,
            )

    g.fig.suptitle(title)
    g.fig.subplots_adjust(top=0.95, hspace=0.3, wspace=0.2)
    return g


# ONE OFF PLOTS
def _group_node_names_by_param(model):
    tmp_params_allowed = model_config[model.model]["params"].copy()
    if hasattr(model, "rlssm_model"):
        if (
            model.rlssm_model
        ):  # TODO: Turns out basic hddm classes have rlssm_model attribute but set to false ....
            tmp_params_allowed.extend(model_config_rl[model.rl_rule]["params"])
    tmp_params_allowed.append("dc")  # to accomodate HDDMStimcoding class
    keys_by_param = {}

    # Cycle through all nodes
    for key_ in model.nodes_db.index:
        if "offset" in key_:
            continue

        # Cycle through model relevant parameters
        for param_tmp in tmp_params_allowed:  # model_config[model.model]["params"]:
            # Initialize param_tmp key if not yet done
            if param_tmp not in keys_by_param.keys():
                keys_by_param[param_tmp] = []

            # Get ids of identifiers
            param_id = key_.find(param_tmp)
            underscore_id = key_.find(param_tmp + "_")
            bracket_id = key_.find(param_tmp + "(")

            # Take out 'trans' and 'tau' and observed nodes
            if (
                not ("trans" in key_)
                and not ("tau" in key_)
                and not ((param_tmp + "_reg") in key_)
                and not ("_rate" in key_)
                and not ("_shape" in key_)
                and not (model.nodes_db.loc[key_].observed)
            ):
                if param_id == 0:
                    if (bracket_id == 0) or (underscore_id == 0):
                        keys_by_param[param_tmp].append(key_)
                    elif key_ == param_tmp:
                        keys_by_param[param_tmp].append(key_)

    # Drop keys that didn't receive and stochastics
    drop_list = []
    for key_ in keys_by_param.keys():
        if len(keys_by_param[key_]) == 0:
            drop_list.append(key_)

    for drop_key in drop_list:
        del keys_by_param[drop_key]

    return keys_by_param


def _group_traces_via_grouped_nodes(model, group_dict):
    grouped_traces = {}
    for key_ in group_dict.keys():
        tmp_traces = {}
        tmp_nodes_db = model.nodes_db.loc[group_dict[key_], :]

        for i in range(tmp_nodes_db.shape[0]):
            tmp_traces[tmp_nodes_db.iloc[i].node.__str__()] = tmp_nodes_db.iloc[
                i
            ].node.trace()

        grouped_traces[key_] = pd.DataFrame.from_dict(tmp_traces, orient="columns")
    return grouped_traces


def plot_caterpillar(
    hddm_model=None,
    ground_truth_parameter_dict=None,
    drop_sd=True,
    keep_key=None,
    figsize=(10, 10),
    columns=3,
    save=False,
    path=None,
    format="png",
    y_tick_size=10,
    x_tick_size=10,
):
    """An alternative posterior predictive plot. Works for all models listed in hddm (e.g. 'ddm', 'angle', 'weibull_cdf', 'levy', 'ornstein')

    Arguments:
        hddm_model: hddm model object <default=None>
            If you supply a ground truth model, the data you supplied to the hddm model should include trial by trial parameters.
        ground_truth_parameter_dict: dict <default=None>
            Parameter dictionary (for example coming out of the function simulator_h_c()) that provides ground truth values
            for the parameters fit in the hddm_model.
        drop_sd: bool <default=True>
            Whether or not to drop group level standard deviations from the caterpillar plot.
            This is sometimes useful because scales can be off if included.
        figsize: tuple <default=(10, 15)>
            Size of initial figure.
        keep_key: list <default=None>
            If you want to keep only a specific list of parameters in the caterpillar plot, supply those here as
            a list. All other parameters for which you supply traces in the posterior samples are going to be ignored.
        save: bool <default=False>
            Whether to save the plot
        format: str <default='png'>
            File format in which to save the figure.
        path: str <default=None>
            Path in which to save the figure.

    Return: plot object
    """

    if hddm_model is None:
        return "No HDDM object supplied"

    out = _group_node_names_by_param(model=hddm_model)
    traces_by_param = _group_traces_via_grouped_nodes(model=hddm_model, group_dict=out)

    ncolumns = columns
    nrows = int(np.ceil(len(out.keys()) / ncolumns))

    fig = plt.figure(figsize=figsize)
    fig.suptitle("")
    fig.subplots_adjust(top=1.0, hspace=0.2, wspace=0.4)

    i = 1
    for key_ in traces_by_param.keys():
        ax = fig.add_subplot(nrows, ncolumns, i)
        sns.despine(right=True, ax=ax)
        traces_tmp = traces_by_param[key_]

        ecdfs = {}
        plot_vals = {}  # [0.01, 0.9], [0.01, 0.99], [mean]

        for k in traces_tmp.keys():
            # If we want to keep only a specific parameter we skip all traces which don't include it in
            # their names !
            if keep_key is not None and k not in keep_key:
                continue

            # Deal with
            if "std" in k and drop_sd:
                pass

            else:
                ok_ = 1

                if drop_sd == True:
                    if "_sd" in k:
                        ok_ = 0
                if ok_:
                    # Make empirical CDFs and extract the 10th, 1th / 99th, 90th percentiles
                    print("tracename: ")
                    print(k)

                    if hasattr(hddm_model, "rlssm_model"):
                        if "rl_alpha" in k and not "std" in k:
                            vals = traces_tmp[k].values
                            transformed_trace = np.exp(vals) / (1 + np.exp(vals))
                            ecdfs[k] = ECDF(transformed_trace)
                            tmp_sorted = sorted(transformed_trace)
                        else:
                            ecdfs[k] = ECDF(traces_tmp[k].values)
                            tmp_sorted = sorted(traces_tmp[k].values)
                    else:
                        ecdfs[k] = ECDF(traces_tmp[k].values)
                        tmp_sorted = sorted(traces_tmp[k].values)

                    _p01 = tmp_sorted[np.sum(ecdfs[k](tmp_sorted) <= 0.01) - 1]
                    _p99 = tmp_sorted[np.sum(ecdfs[k](tmp_sorted) <= 0.99) - 1]
                    _p1 = tmp_sorted[np.sum(ecdfs[k](tmp_sorted) <= 0.1) - 1]
                    _p9 = tmp_sorted[np.sum(ecdfs[k](tmp_sorted) <= 0.9) - 1]
                    _pmean = traces_tmp[k].mean()
                    plot_vals[k] = [[_p01, _p99], [_p1, _p9], _pmean]

        x = [plot_vals[k][2] for k in plot_vals.keys()]

        # Create y-axis labels first
        ax.scatter(x, plot_vals.keys(), c="black", marker="s", alpha=0)
        i += 1

        # Plot the actual cdf-based data
        for k in plot_vals.keys():
            ax.plot(plot_vals[k][1], [k, k], c="grey", zorder=-1, linewidth=5)
            ax.plot(plot_vals[k][0], [k, k], c="black", zorder=-1)

            # Add in ground truth if supplied
            if ground_truth_parameter_dict is not None:
                ax.scatter(ground_truth_parameter_dict[k], k, c="blue", marker="|")

        ax.tick_params(axis="y", rotation=45)
        ax.tick_params(axis="y", labelsize=y_tick_size)
        ax.tick_params(axis="x", labelsize=x_tick_size)

    if save:
        fname = "caterpillar_" + hddm_model.model

        if path is None:
            path = "."
        if isinstance(format, str):
            format = [format]

        print(["%s.%s" % (os.path.join(path, fname), x) for x in format])
        [
            fig.savefig("%s.%s" % (os.path.join(path, fname), x), format=x)
            for x in format
        ]

    plt.show()


"""
=== RLSSM functions ===
"""


def get_mean_correct_responses_rlssm(trials, nbins, data):
    """Gets the mean proportion of correct responses condition-wise.

    Arguments:
        trials: int
            Number of initial trials to consider for computing proportion of correct responses.

        nbins: int
            Number of bins to put the trials into (Num. of trials per bin = trials/nbin).

        data: pandas.DataFrame
            Pandas DataFrame for the observed or simulated data.

    Return:
        mean_correct_responses: dict
            Dictionary of conditions containing proportion of mean correct responses (for each bin).
        up_err: dict
            Dictionary of conditions containing upper intervals of HDI of mean correct responses (for each bin).
        low_err: dict
            Dictionary of conditions containing lower intervals of HDI of mean correct responses (for each bin).
    """

    data_ppc = data[data.trial <= trials].copy()
    data_ppc.loc[data_ppc["response"] < 1, "response"] = 0

    data_ppc["bin_trial"] = np.array(
        pd.cut(data_ppc.trial, nbins, labels=np.linspace(0, nbins, nbins))
    )

    sums = data_ppc.groupby(["bin_trial", "split_by", "trial"]).mean().reset_index()

    ppc_sim = sums.groupby(["bin_trial", "split_by"]).mean().reset_index()

    # initiate columns that will have the upper and lower bound of the hpd
    ppc_sim["upper_hpd"] = 0
    ppc_sim["lower_hpd"] = 0
    for i in range(0, ppc_sim.shape[0]):
        # calculate the hpd/hdi of the predicted mean responses across bin_trials
        hdi = pymc.utils.hpd(
            sums.response[
                (sums["bin_trial"] == ppc_sim.bin_trial[i])
                & (sums["split_by"] == ppc_sim.split_by[i])
            ],
            alpha=0.05,
        )
        ppc_sim.loc[i, "upper_hpd"] = hdi[1]
        ppc_sim.loc[i, "lower_hpd"] = hdi[0]

    # calculate error term as the distance from upper bound to mean
    ppc_sim["up_err"] = ppc_sim["upper_hpd"] - ppc_sim["response"]
    ppc_sim["low_err"] = ppc_sim["response"] - ppc_sim["lower_hpd"]

    mean_correct_responses = {}
    up_err = {}
    low_err = {}
    for cond in np.unique(ppc_sim.split_by):
        mean_correct_responses[cond] = ppc_sim[ppc_sim.split_by == cond]["response"]
        up_err[cond] = ppc_sim[ppc_sim.split_by == cond]["up_err"]
        low_err[cond] = ppc_sim[ppc_sim.split_by == cond]["low_err"]

    return mean_correct_responses, up_err, low_err


def gen_ppc_rlssm(
    model_ssm,
    config_ssm,
    model_rl,
    config_rl,
    data,
    traces,
    nsamples,
    p_lower,
    p_upper,
    save_data=False,
    save_name=None,
    save_path=None,
):
    """Generates data (for posterior predictives) using samples from the given trace as parameters.

    Arguments:
        model_ssm: str
            Name of the sequential sampling model used.

        config_ssm: dict
            Config dictionary for the specified sequential sampling model.

        model_rl: str
            Name of the reinforcement learning model used.

        config_rl: dict
            Config dictionary for the specified reinforcement learning model.

        data: pandas.DataFrame
            Pandas DataFrame for the observed data.

        traces: pandas.DataFrame
            Pandas DataFrame containing the traces.

        nsamples: int
            Number of posterior samples to draw for each subject.

        p_lower: dict
            Dictionary of conditions containing the probability of reward for the lower choice/action in the 2-armed bandit task.

        p_upper: dict
            Dictionary of conditions containing the probability of reward for the upper choice/action in the 2-armed bandit task.

        save_data: bool <default=False>
            Boolean denoting whether to save the data as csv.

        save_name: str <default=None>
            Specifies filename to save the data.

        save_path: str <default=None>
            Specifies path to save the data.


    Return:
        ppc_sdata: pandas.DataFrame
            Pandas DataFrame containing the simulated data (for posterior predictives).
    """

    def transform_param(param, param_val):
        if param == "rl_alpha":
            transformed_param_val = np.exp(param_val) / (1 + np.exp(param_val))
        else:
            transformed_param_val = param_val

        return transformed_param_val

    sim_data = pd.DataFrame()

    nsamples += 1
    for i in tqdm(range(1, nsamples)):
        sample = np.random.randint(0, traces.shape[0] - 1)

        for subj in data.subj_idx.unique():
            sub_data = pd.DataFrame()

            for cond in np.unique(data.split_by):
                sampled_param_ssm = list()
                for p in config_ssm["params"]:
                    p_val = traces.loc[sample, p + "_subj." + str(subj)]
                    p_val = transform_param(p, p_val)
                    sampled_param_ssm.append(p_val)

                sampled_param_rl = list()
                for p in config_rl["params"]:
                    p_val = traces.loc[sample, p + "_subj." + str(subj)]
                    p_val = transform_param(p, p_val)
                    sampled_param_rl.append(p_val)

                cond_size = len(
                    data[
                        (data["subj_idx"] == subj) & (data["split_by"] == cond)
                    ].trial.unique()
                )
                ind_cond_data = gen_rand_rlssm_data_MAB_RWupdate(
                    model_ssm,
                    sampled_param_ssm,
                    sampled_param_rl,
                    size=cond_size,
                    p_lower=p_lower[cond],
                    p_upper=p_upper[cond],
                    subjs=1,
                    split_by=cond,
                )

                # append the conditions
                # sub_data = sub_data.append([ind_cond_data], ignore_index=False)
                sub_data = pd.concat([sub_data, ind_cond_data], ignore_index=False)

            # assign subj_idx
            sub_data["subj_idx"] = subj

            # identify the simulated data
            sub_data["samp"] = i

            # append data from each subject
            # sim_data = sim_data.append(sub_data, ignore_index=True)
            sim_data = pd.concat([sim_data, sub_data], ignore_index=True)

    ppc_sdata = sim_data[
        ["subj_idx", "response", "split_by", "rt", "trial", "feedback", "samp"]
    ].copy()

    if save_data:
        if save_name is None:
            save_name = "ppc_data"
        if save_path is None:
            save_path = "."
        ppc_sdata.to_csv("%s.%s" % (os.path.join(save_path, save_name), "csv"))
        print("ppc data saved at %s.%s" % (os.path.join(save_path, save_name), "csv"))

    return ppc_sdata


def plot_ppc_choice_rlssm(
    obs_data, sim_data, trials, nbins, save_fig=False, save_name=None, save_path=None
):
    """Plot posterior preditive plot for choice data.

    Arguments:
        obs_data: pandas.DataFrame
            Pandas DataFrame for the observed data.

        sim_data: pandas.DataFrame
            Pandas DataFrame for the simulated data.

        trials: int
            Number of initial trials to consider for computing proportion of correct responses.

        nbins: int
            Number of bins to put the trials into (Num. of trials per bin = trials/nbin).

        save_fig: bool <default=False>
            Boolean denoting whether to save the plot.

        save_name: str <default=None>
            Specifies filename to save the figure.

        save_path: str <default=None>
            Specifies path to save the figure.


    Return:
        fig: matplotlib.Figure
            plot object
    """

    res_obs, up_err_obs, low_err_obs = get_mean_correct_responses_rlssm(
        trials, nbins, obs_data
    )
    res_sim, up_err_sim, low_err_sim = get_mean_correct_responses_rlssm(
        trials, nbins, sim_data
    )

    cond_list = np.unique(obs_data.split_by)
    rows = 1
    cols = len(cond_list)
    fig, ax = plt.subplots(rows, cols, constrained_layout=False, tight_layout=True)

    cond_index = 0
    for ay in range(cols):
        cond = cond_list[cond_index]

        ax[ay].errorbar(
            1 + np.arange(len(res_obs[cond])),
            res_obs[cond],
            yerr=[low_err_obs[cond], up_err_obs[cond]],
            label="observed",
            color="royalblue",
        )
        ax[ay].errorbar(
            1 + np.arange(len(res_sim[cond])),
            res_sim[cond],
            yerr=[low_err_sim[cond], up_err_sim[cond]],
            label="simulated",
            color="tomato",
        )

        ax[ay].set_ylim((0, 1))

        # ax[ay].legend()
        ax[ay].set_title("split_by=" + str(cond), fontsize=12)
        ax[ay].grid()

        cond_index += 1

    fig = plt.gcf()
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc="lower right")

    fig.supxlabel("Trial bins", fontsize=12)
    fig.supylabel("Proportion of Correct Responses", fontsize=12)
    fig.set_size_inches(4 * len(cond_list), 4)

    if save_fig:
        if save_name is None:
            save_name = "ppc_choice"
        if save_path is None:
            save_path = "."
        fig.savefig("%s.%s" % (os.path.join(save_path, save_name), "png"))
        print("fig saved at %s.%s" % (os.path.join(save_path, save_name), "png"))

    return fig


def plot_ppc_rt_rlssm(
    obs_data, sim_data, trials, bw=0.1, save_fig=False, save_name=None, save_path=None
):
    """Plot posterior preditive plot for reaction time data.

    Arguments:
        obs_data: pandas.DataFrame
            Pandas DataFrame for the observed data.

        sim_data: pandas.DataFrame
            Pandas DataFrame for the simulated data.

        trials: int
            Number of initial trials to consider for computing proportion of correct responses.

        bw: float <default=0.1>
            Bandwidth parameter for kernel-density estimates.

        save_fig: bool <default=False>
            Boolean denoting whether to save the plot.

        save_name: str <default=None>
            Specifies filename to save the figure.

        save_path: str <default=None>
            Specifies path to save the figure.


    Return:
        fig: matplotlib.Figure
            plot object
    """

    obs_data_ppc = obs_data[obs_data.trial <= trials].copy()
    sim_data_ppc = sim_data[sim_data.trial <= trials].copy()

    cond_list = np.unique(obs_data.split_by)
    rows = 1
    cols = len(cond_list)
    fig, ax = plt.subplots(rows, cols, constrained_layout=False, tight_layout=True)

    cond_index = 0
    for ay in range(cols):
        cond = cond_list[cond_index]

        rt_ppc_sim = np.where(
            sim_data_ppc[sim_data_ppc.split_by == cond].response == 1,
            sim_data_ppc[sim_data_ppc.split_by == cond].rt,
            0 - sim_data_ppc[sim_data_ppc.split_by == cond].rt,
        )
        rt_ppc_obs = np.where(
            obs_data_ppc[obs_data_ppc.split_by == cond].response == 1,
            obs_data_ppc[obs_data_ppc.split_by == cond].rt,
            0 - obs_data_ppc[obs_data_ppc.split_by == cond].rt,
        )

        sns.kdeplot(
            rt_ppc_sim, label="simulated", color="tomato", ax=ax[ay], bw_method=bw
        ).set(ylabel=None)
        sns.kdeplot(
            rt_ppc_obs, label="observed", color="royalblue", ax=ax[ay], bw_method=bw
        ).set(ylabel=None)

        # ax[ay].legend()
        ax[ay].set_title("split_by=" + str(cond), fontsize=12)
        ax[ay].grid()

        cond_index += 1

    fig = plt.gcf()
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc="lower right")

    fig.supxlabel("Reaction Time", fontsize=12)
    fig.supylabel("Density", fontsize=12)
    fig.set_size_inches(4 * len(cond_list), 4)

    if save_fig:
        if save_name is None:
            save_name = "ppc_rt"
        if save_path is None:
            save_path = "."
        fig.savefig("%s.%s" % (os.path.join(save_path, save_name), "png"))
        print("fig saved at %s.%s" % (os.path.join(save_path, save_name), "png"))

    return fig


def plot_posterior_pairs_rlssm(
    tracefile, param_list, save_fig=False, save_name=None, save_path=None, **kwargs
):
    """Plot posterior pairs.

    Arguments:
        tracefile: dict
            Dictionary containing the traces.

        param_list: list
            List of model parameters to be included in the posterior pair plots.

        save_fig: bool <default=False>
            Boolean denoting whether to save the plot.

        save_name: str <default=None>
            Specifies filename to save the figure.

        save_path: str <default=None>
            Specifies path to save the figure.


    Return:
        fig: matplotlib.Figure
            plot object
    """

    traces = hddm.utils.get_traces_rlssm(tracefile)
    tr = traces.copy()
    tr_trunc = tr[param_list]
    tr_dataset = az.dict_to_dataset(tr_trunc)
    tr_inf_data = az.convert_to_inference_data(tr_dataset)

    axes = az.plot_pair(
        tr_inf_data,
        kind="kde",
        marginals=True,
        point_estimate="mean",
        textsize=20,
        **kwargs,
    )

    fig = axes.ravel()[0].figure

    if save_fig:
        if save_name is None:
            save_name = "posterior_pair"
        if save_path is None:
            save_path = "."
        fig.savefig("%s.%s" % (os.path.join(save_path, save_name), "png"))
        print("fig saved at %s.%s" % (os.path.join(save_path, save_name), "png"))

    return fig
