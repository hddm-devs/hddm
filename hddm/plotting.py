from hddm.simulators import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

import os
import warnings

# import pymc as pm
# import hddm

import pandas as pd

from kabuki.analyze import _post_pred_generate, _parents_to_random_posterior_sample
from statsmodels.distributions.empirical_distribution import ECDF

from hddm.model_config import model_config

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
    **kwargs
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
    if hasattr(model, "model"):
        kwargs["model_"] = model.model
    else:
        kwargs["model_"] = "ddm_vanilla"

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
                print("passing_print")
                if len(tag) == 0:
                    fname = "ppq_subject_" + str(subj_i)
                else:
                    fname = "ppq_" + ".".join(tag) + "_subject_" + str(subj_i)

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
    generative_model="ddm_vanilla",
    plot_func=None,
    columns=None,
    save=False,
    path=None,
    groupby="subj_idx",
    figsize=(8, 6),
    format="png",
    **kwargs
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
    fig.suptitle(title_, fontsize=12)
    fig.subplots_adjust(top=0.9, hspace=0.4, wspace=0.3)

    i = 1
    for group_id, df_tmp in df.groupby(groupby):
        nrows = np.ceil(n_plots / columns)

        # Plot individual subjects (if present)
        ax = fig.add_subplot(np.ceil(nrows), columns, i)

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
        print(tag)

        ax.set_title(tag, fontsize=ax_title_size)

        # Call plot function on ax
        # This function should manipulate the ax object, and is expected to not return anything.
        plot_func(df_tmp, ax, **kwargs)
        i += 1

        # Save figure if necessary
        if save:
            fname = "ppq_" + tag
            if path is None:
                path = "."
            if isinstance(format, str):
                format = [format]
            [
                fig.savefig("%s.%s" % (os.path.join(path, fname), x), format=x)
                for x in format
            ]


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
    **kwargs
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
        kwargs["model_"] = "ddm_vanilla"

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

        nrows = num_subjs or np.ceil(len(nodes) / columns)

        if len(nodes) - (int(nrows) * columns) > 0:
            nrows += 1

        # Plot individual subjects (if present)
        i = 0
        for subj_i, (node_name, bottom_node) in enumerate(nodes.iterrows()):
            i += 1
            if not hasattr(bottom_node["node"], required_method):
                continue  # skip nodes that do not define the required_method

            ax = fig.add_subplot(np.ceil(nrows), columns, subj_i + 1)
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

            if i > (np.ceil(nrows) * columns):
                warnings.warn("Too many nodes. Consider increasing number of columns.")
                break

            if num_subjs is not None and i >= num_subjs:
                break

        # Save figure if necessary
        if save:
            fname = "ppq_" + ".".join(tag)
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
    **kwargs
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
        model : str <default='ddm_vanilla'>
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

    model_ = kwargs.pop("model_", "ddm_vanilla")
    choices = model_config[model_]["choices"]
    n_choices = len(model_config[model_]["choices"])

    # data_color = kwargs.pop("data_color", "blue")
    # posterior_color = kwargs.pop("posterior_color", "red")

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
    **kwargs
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
                rt_dat.loc[rt_dat.response != 1, "rt"] = (-1) * rt_dat.rt[
                    rt_dat.response != 1
                ].values

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
    add_posterior_uncertainty_model=False,
    add_posterior_uncertainty_rts=False,
    add_posterior_mean_model=True,
    add_posterior_mean_rts=True,
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
    add_legend=True,
    **kwargs
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

    # AF-TODO: Add a mean version of this !
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

    axis.set_xlim(value_range[0], value_range[-1])
    axis.set_ylim(-ylim, ylim)
    axis_twin_up = axis.twinx()
    axis_twin_down = axis.twinx()
    axis_twin_up.set_ylim(-ylim, ylim)
    axis_twin_up.set_yticks([])
    axis_twin_down.set_ylim(ylim, -ylim)
    axis_twin_down.set_yticks([])

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
                histtype="step",
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
                histtype="step",
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
            histtype="step",
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
            histtype="step",
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
            tmp_label = "Data"

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
            histtype="step",
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
            histtype="step",
            alpha=1,
            color=data_color,
            edgecolor=data_color,
            linewidth=linewidth_histogram,
            zorder=-1,
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
                delta_t_graph=delta_t_model,
                sample_hist_alpha=alpha,
                lw_m=linewidth_model,
                tmp_label=tmp_label,
                ylim=ylim,
                t_s=t_s,
                color=posterior_uncertainty_color,
                zorder_cnt=j,
            )

    if (node_data_full is not None) and add_data_model:
        _add_model_cartoon_to_ax(
            sample=node_data_full,
            axis=axis,
            tmp_model=tmp_model,
            delta_t_graph=delta_t_model,
            sample_hist_alpha=1.0,
            lw_m=linewidth_model + 0.5,
            tmp_label=None,
            ylim=ylim,
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
            delta_t_graph=delta_t_model,
            sample_hist_alpha=1.0,
            lw_m=linewidth_model + 0.5,
            tmp_label=None,
            ylim=ylim,
            t_s=t_s,
            color=posterior_mean_color,
            zorder_cnt=j + 2,
        )


def _add_model_cartoon_to_ax(
    sample=None,
    axis=None,
    tmp_model=None,
    delta_t_graph=None,
    sample_hist_alpha=None,
    lw_m=None,
    tmp_label=None,
    ylim=None,
    t_s=None,
    zorder_cnt=1,
    color="black",
):
    if (
        tmp_model == "weibull_cdf"
        or tmp_model == "weibull_cdf2"
        or tmp_model == "weibull_cdf_concave"
        or tmp_model == "weibull"
    ):

        b = np.maximum(
            sample.a.values[0]
            * model_config[tmp_model]["boundary"](
                t=t_s, alpha=sample.alpha.values[0], beta=sample.beta.values[0]
            ),
            0,
        )

    if tmp_model == "angle" or tmp_model == "angle2":
        b = np.maximum(
            sample.a.values[0]
            + model_config[tmp_model]["boundary"](t=t_s, theta=sample.theta.values[0]),
            0,
        )

    if (
        tmp_model == "ddm"
        or tmp_model == "ornstein"
        or tmp_model == "levy"
        or tmp_model == "full_ddm"
    ):

        b = sample.a.values[0] * np.ones(t_s.shape[0])

    # MAKE SLOPES (VIA TRAJECTORIES HERE --> RUN NOISE FREE SIMULATIONS)!
    out = simulator(
        theta=sample[model_config[tmp_model]["params"]].values[0],
        model=tmp_model,
        n_samples=1,
        no_noise=True,
        delta_t=delta_t_graph,
        bin_dim=None,
    )

    # AF-TODO: Add trajectories
    tmp_traj = out[2]["trajectory"]
    maxid = np.minimum(np.argmax(np.where(tmp_traj > -999)), t_s.shape[0])

    # Upper bound
    axis.plot(
        t_s + sample.t.values[0],
        b,
        color=color,
        alpha=sample_hist_alpha,
        zorder=1000 + zorder_cnt,
        linewidth=lw_m,
        label=tmp_label,
    )

    # Lower bound
    axis.plot(
        t_s + sample.t.values[0],
        -b,
        color=color,
        alpha=sample_hist_alpha,
        zorder=1000 + zorder_cnt,
        linewidth=lw_m,
    )

    # Slope
    axis.plot(
        t_s[:maxid] + sample.t.values[0],
        tmp_traj[:maxid],
        color=color,
        alpha=sample_hist_alpha,
        zorder=1000 + zorder_cnt,
        linewidth=lw_m,
    )  # TOOK AWAY LABEL

    # Starting point
    axis.axvline(
        x=sample.t.values[0],
        ymin=-ylim,
        ymax=ylim,
        color=color,
        linestyle="--",
        linewidth=lw_m,
        alpha=sample_hist_alpha,
    )


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
    **kwargs
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
    bottom_node, model_="ddm_vanilla", n_samples=200, figsize=(8, 6), title="", **kwargs
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

        model: str <default='ddm_vanilla'>

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

    g = sns.PairGrid(
        data=df, corner=True
    )  # height = figsize[0], aspect = figsize[1] / figsize[0])
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
    tmp_params_allowed.append("dc")  # to accomodate HDDMStimcoding class
    keys_by_param = {}

    # Cycle through all nodes
    for key_ in model.nodes_db.index:

        # Cycle through model relevant parameters
        for param_tmp in model_config[model.model]["params"]:

            # Initialize param_tmp key if not yet done
            if param_tmp not in keys_by_param.keys():
                keys_by_param[param_tmp] = []

            # Get ids of identifiers
            param_id = key_.find(param_tmp)
            underscore_id = key_.find(param_tmp + "_")
            bracket_id = key_.find(param_tmp + "(")

            # Take out 'trans' and 'tau' and observed nodes
            if (
                ("trans" not in key_)
                and ("tau" not in key_)
                and not ((param_tmp + "_reg") in key_)
                and (model.nodes_db.loc[key_].observed == False)
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

    """An alternative posterior predictive plot. Works for all models listed in hddm (e.g. 'ddm', 'angle', 'weibull', 'levy', 'ornstein')

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
        x_limits: float <default=2>
            Sets the limit on the x-axis
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
    nrows = np.ceil(len(out.keys()) / ncolumns)

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
            if keep_key is not None and keep_key not in k:
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
        print("passing_print")

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
