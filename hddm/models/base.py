"""
.. module:: HDDM
   :platform: Agnostic
   :synopsis: Definition of HDDM models.

.. moduleauthor:: Thomas Wiecki <thomas.wiecki@gmail.com>
                  Imri Sofer <imrisofer@gmail.com>


"""

from __future__ import division
from collections import OrderedDict

import numpy as np
import pymc as pm

import hddm
import kabuki
import inspect

from kabuki.hierarchical import Knode
from scipy.optimize import fmin_powell, fmin

class AccumulatorModel(kabuki.Hierarchical):
    def __init__(self, data, **kwargs):
        # Flip sign for lower boundary RTs
        data = hddm.utils.flip_errors(data)

        self.group_only_nodes = kwargs.pop('group_only_nodes', ())

        super(AccumulatorModel, self).__init__(data, **kwargs)


    def _create_an_average_model(self):
        raise NotImplementedError, "This method has to be overloaded. See HDDMBase."


    def _quantiles_optimization(self, method, quantiles=(.1, .3, .5, .7, .9 ), n_runs=3):
        """
        quantile optimization using chi^2.
        Input:
            quantiles <sequance> - a sequance of quantiles.
                the default values are the one used by Ratcliff (.1, .3, .5, .7, .9).
        Output:
            results <dict> - a results dictionary of the parameters values.
            The values of the nodes in single subject model is update according to the results.
            The nodes of group models are not updated
        """

        #run optimization for group model
        if self.is_group_model:

            #get all obs nodes
            obs_db = self.get_observeds()

            #create an average model (avergae of all subjects)
            try:
                average_model = self._create_an_average_model()
            except AttributeError:
                raise AttributeError("User must define _create_an_average_model in order to use the quantiles optimization method")

            #group obs nodes according to their tag and (condittion)
            #and for each group average the quantiles
            for (tag, tag_obs_db) in obs_db.groupby(obs_db.tag):

                #set quantiles for each observed_node
                obs_nodes = tag_obs_db.node;
                [obs.compute_quantiles_stats(quantiles) for obs in obs_nodes]

                #get n_samples, freq_obs, and emp_rt
                stats = [obs.get_quantiles_stats() for obs in obs_nodes]
                n_samples = sum([x['n_samples'] for x in stats])
                freq_obs = sum(np.array([x['freq_obs'] for x in stats]),0)
                emp_rt = np.mean(np.array([x['emp_rt'] for x in stats]),0)

                #set average quantiles  to have the same statitics
                obs_knode = [x for x in self.knodes if x.name == 'wfpt'][0]
                node_name = obs_knode.create_node_name(tag) #get node name
                average_node = average_model.nodes_db.ix[node_name]['node'] #get the average node
                average_node.set_quantiles_stats(n_samples, emp_rt, freq_obs) #set the quantiles

            #optimize
            results, bic_info = average_model._optimization_single(method=method, quantiles=quantiles,
                                                                   n_runs=n_runs, compute_stats=False)

        #run optimization for single subject model
        else:
            results, bic_info = self._optimization_single(method=method, quantiles=quantiles,
                                                          n_runs=n_runs, compute_stats=True)

        if bic_info is not None:
            self.bic_info = bic_info

        return results


    def optimize(self, method, quantiles=(.1, .3, .5, .7, .9 ), n_runs=3):
        """
        optimization using ML, chi^2 or G^2

        Input:
            method <string> - name of method ('ML', 'chisquare' or 'gsquare')
            quantiles <sequance> - a sequance of quantiles used for chi^2 and G^2
                the default values are the one used by Ratcliff (.1, .3, .5, .7, .9).
        Output:
            results <dict> - a results dictionary of the parameters values.
            The values of the nodes in single subject model is update according to the results.
            The nodes of group models are not updated
        """


        if method == 'ML':
            if self.is_group_model:
                raise TypeError, "optimization method is not defined for group models"
            else:
                results, _ = self._optimization_single(method, quantiles, n_runs=n_runs)
                return results

        else:
            return self._quantiles_optimization(method, quantiles, n_runs=n_runs)

    def _optimization_single(self, method, quantiles, n_runs, compute_stats=True):
        """
        function used by chisquare_optimization to fit the a single subject model
        Input:
         quantiles <sequance> - same as in chisquare_optimization
         cmopute_stats <boolean> - whether to copmute the quantile stats using the node's
             compute_quantiles_stats method

        Output:
            results <dict> - same as in chisquare_optimization
        """

        #get obs_nodes
        obs_nodes = self.get_observeds()['node']

        #set quantiles for each observed_node (if needed)
        if (method != 'ML') and compute_stats:
            [obs.compute_quantiles_stats(quantiles) for obs in obs_nodes]

        #get all stochastic parents of observed nodes
        db = self.nodes_db
        parents = db[(db.stochastic == True) & (db.observed == False)]['node']
        original_values = np.array([x.value for x in parents])
        names = [x.__name__ for x in parents]

        #define objective
        #ML method
        if method == 'ML':
            def objective(values):
                for (i, value) in enumerate(values):
                    parents[i].value = value
                try:
                    return -sum([obs.logp for obs in obs_nodes])
                except pm.ZeroProbability:
                    return np.inf

        #chi^2 method
        elif method == 'chisquare':
            def objective(values):
                for (i, value) in enumerate(values):
                    parents[i].value = value
                score = sum([obs.chisquare() for obs in obs_nodes])
                if score < 0:
                    kabuki.debug_here()
                return score

        #G^2 method
        elif method == 'gsquare':
            def objective(values):
                for (i, value) in enumerate(values):
                    parents[i].value = value
                return -sum([obs.gsquare() for obs in obs_nodes])

        else:
            raise ValueError('unknown optimzation method')

        #optimze
        best_score = np.inf
        all_results = []
        values = original_values.copy()
        inf_objective = False
        for i_run in xrange(n_runs):
            #initalize values to a random point
            values_iter = 0
            while inf_objective:
                values_iter += 1
                values = original_values + np.random.randn(len(values))*(2**-values_iter)
                self.set_values(dict(zip(names, values)))
                inf_objective = np.isinf(objective(values))

            #optimze
            try:
                res_tuple = fmin_powell(objective, values, full_output=True)
            except Exception:
                res_tuple = fmin(objective, values, full_output=True)
            all_results.append(res_tuple)

            #reset inf_objective so values be resampled
            inf_objective = True

        #get best results
        best_idx = np.nanargmin([x[1] for x in all_results])
        best_values = all_results[best_idx][0]
        self.set_values(dict(zip(names, best_values)))
        results = self.values

        #calc BIC for G^2
        if method == 'gsquare':
            values = [x.value for x in parents]
            score = objective(values)
            penalty = len(results) * np.log(len(self.data))
            bic_info = {'likelihood': -score, 'penalty': penalty, 'bic': score + penalty}
            return results, bic_info

        else:
            return results, None



class HDDMBase(AccumulatorModel):
    """Implements the hierarchical Ratcliff drift-diffusion model
    using the Navarro & Fuss likelihood and numerical integration over
    the variability parameters.

    :Arguments:
        data : numpy.recarray
            Input data with a row for each trial.
             Must contain the following columns:
              * 'rt': Reaction time of trial in seconds.
              * 'response': Binary response (e.g. 0->error, 1->correct)
             May contain:
              * 'subj_idx': A unique ID (int) of the subject.
              * Other user-defined columns that can be used in depends on keyword.
    :Example:
        >>> data, params = hddm.generate.gen_rand_data() # gen data
        >>> model = hddm.HDDM(data) # create object
        >>> mcmc = model.mcmc() # Create pymc.MCMC object
        >>> mcmc.sample() # Sample from posterior
    :Optional:
        include : iterable
            Optional inter-trial variability parameters to include.
             Can be any combination of 'sv', 'sz' and 'st'. Passing the string
            'all' will include all three.

            Note: Including 'sz' and/or 'st' will increase run time significantly!

            is_group_model : bool
                If True, this results in a hierarchical
                model with separate parameter distributions for each
                subject. The subject parameter distributions are
                themselves distributed according to a group parameter
                distribution.

                If not provided, this parameter is set to True if data
                provides a column 'subj_idx' and False otherwise.

            depends_on : dict
                Specifies which parameter depends on data
                of a column in data. For each unique element in that
                column, a separate set of parameter distributions will be
                created and applied. Multiple columns can be specified in
                a sequential container (e.g. list)

                :Example:

                    >>> hddm.HDDM(data, depends_on={'v':'difficulty'})

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

            plot_var : bool
                 Plot group variability parameters when calling pymc.Matplot.plot()
                 (i.e. variance of Normal distribution.)

            wiener_params : dict
                 Parameters for wfpt evaluation and
                 numerical integration.

                 :Parameters:
                     * err: Error bound for wfpt (default 1e-4)
                     * n_st: Maximum depth for numerical integration for st (default 2)
                     * n_sz: Maximum depth for numerical integration for Z (default 2)
                     * use_adaptive: Whether to use adaptive numerical integration (default True)
                     * simps_err: Error bound for Simpson integration (default 1e-3)

    """

    def __init__(self, data, bias=False, include=(),
                 wiener_params=None, p_outlier=0., **kwargs):

        self._kwargs = kwargs

        self.include = set(['v', 'a', 't'])
        if include is not None:
            if include == 'all':
                [self.include.add(param) for param in ('z', 'st','sv','sz', 'p_outlier')]
            elif isinstance(include, str):
                self.include.add(include)
            else:
                [self.include.add(param) for param in include]

        if bias:
            self.include.add('z')

        possible_parameters = ('v', 'a', 't', 'z', 'st', 'sz', 'sv', 'p_outlier')
        assert self.include.issubset(possible_parameters), """Received and invalid parameter using the 'include' keyword.
        parameters received: %s
        parameters allowed: %s """ % (tuple(self.include), possible_parameters)

        #set wiener params
        if wiener_params is None:
            self.wiener_params = {'err': 1e-4, 'n_st':2, 'n_sz':2,
                                  'use_adaptive':1,
                                  'simps_err':1e-3,
                                  'w_outlier': 0.1}
        else:
            self.wiener_params = wiener_params
        wp = self.wiener_params
        self.p_outlier = p_outlier

        #set cdf_range
        cdf_bound = max(np.abs(data['rt'])) + 1;
        cdf_range = (-cdf_bound, cdf_bound)

        #set wfpt class
        self.wfpt_class = hddm.likelihoods.generate_wfpt_stochastic_class(wp, cdf_range=cdf_range)

        super(HDDMBase, self).__init__(data, **kwargs)

    def _create_wfpt_parents_dict(self, knodes):
        wfpt_parents = OrderedDict()
        wfpt_parents['a'] = knodes['a_bottom']
        wfpt_parents['v'] = knodes['v_bottom']
        wfpt_parents['t'] = knodes['t_bottom']

        wfpt_parents['sv'] = knodes['sv_bottom'] if 'sv' in self.include else 0
        wfpt_parents['sz'] = knodes['sz_bottom'] if 'sz' in self.include else 0
        wfpt_parents['st'] = knodes['st_bottom'] if 'st' in self.include else 0
        wfpt_parents['z'] = knodes['z_bottom'] if 'z' in self.include else 0.5
        wfpt_parents['p_outlier'] = knodes['p_outlier_bottom'] if 'p_outlier' in self.include else self.p_outlier
        return wfpt_parents

    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)

        return Knode(self.wfpt_class, 'wfpt', observed=True, col_name='rt', **wfpt_parents)

    def create_knodes(self):
        knodes = self._create_stochastic_knodes(self.include)
        knodes['wfpt'] = self._create_wfpt_knode(knodes)

        return knodes.values()

    def plot_posterior_predictive(self, *args, **kwargs):
        if 'value_range' not in kwargs:
            kwargs['value_range'] = np.linspace(-5, 5, 100)
        kabuki.analyze.plot_posterior_predictive(self, *args, **kwargs)

    def plot_posterior_quantiles(self, *args, **kwargs):
        """Plot posterior predictive quantiles.

        :Arguments:

            model : HDDM model

        :Optional:

            value_range : numpy.ndarray
                Range over which to evaluate the CDF.

            samples : int (default=10)
                Number of posterior samples to use.

            alpha : float (default=.75)
               Alpha (transparency) of posterior quantiles.

            hexbin : bool (default=False)
               Whether to plot posterior quantile density
               using hexbin.

            data_plot_kwargs : dict (default=None)
               Forwarded to data plotting function call.

            predictive_plot_kwargs : dict (default=None)
               Forwareded to predictive plotting function call.

            columns : int (default=3)
                How many columns to use for plotting the subjects.

            savefig : bool (default=False)
                Whether to save the figure to a file.

            path : str (default=None)
                Save figure into directory prefix

        """
        hddm.utils.plot_posterior_quantiles(self, *args, **kwargs)

    def _create_an_average_model(self):
        """
        create an average model for group model quantiles optimization.

        The function return an instance of the model that was initialize with the same parameters as the
        original model but with the is_group_model argument set to False.
        since it depends on the specifics of the class it should be implemented by the user for each new class.
        """

        #this code only check that the arguments are as expected, i.e. the constructor was not change
        #since we wrote this function
        init_args = set(inspect.getargspec(self.__init__).args)
        known_args = set(['wiener_params', 'include', 'self', 'bias', 'data'])
        assert known_args == init_args, "Arguments of the constructor are not as expected"

        #create the avg model
        avg_model  = self.__class__(self.data, include=self.include, is_group_model=False, **self._kwargs)
        return avg_model

if __name__ == "__main__":
    import doctest
    doctest.testmod()
