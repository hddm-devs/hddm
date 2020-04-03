"""
.. module:: HDDM
   :platform: Agnostic
   :synopsis: Definition of HDDM models.

.. moduleauthor:: Thomas Wiecki <thomas.wiecki@gmail.com>
                  Imri Sofer <imrisofer@gmail.com>


"""


from collections import OrderedDict

import numpy as np
import pymc as pm
import pandas as pd

import hddm
import kabuki
import inspect

from kabuki.hierarchical import Knode
from scipy.optimize import fmin_powell, fmin

try:
    from IPython import parallel
    from IPython.parallel.client.asyncresult import AsyncResult
except ImportError:
    pass



class AccumulatorModel(kabuki.Hierarchical):
    def __init__(self, data, **kwargs):
        # Flip sign for lower boundary RTs
        data = hddm.utils.flip_errors(data)
        self.std_depends = kwargs.pop('std_depends', False)

        super(AccumulatorModel, self).__init__(data, **kwargs)


    def _create_an_average_model(self):
        raise NotImplementedError("This method has to be overloaded. See HDDMBase.")


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

            #create an average model
            average_model = self.get_average_model(quantiles)

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

    def get_average_model(self, quantiles=(.1, .3, .5, .7, .9)):

        #create an average model (avergae of all subjects)
        try:
            average_model = self._create_an_average_model()
            average_model._is_average_model = True
        except AttributeError:
            raise AttributeError("User must define _create_an_average_model in order to use the quantiles optimization method")

        #get all obs nodes
        obs_db = self.get_observeds()

        #group obs nodes according to their tag and (condittion)
        #and for each group average the quantiles
        for (tag, tag_obs_db) in obs_db.groupby(obs_db.tag):

            #set quantiles for each observed_node
            obs_nodes = tag_obs_db.node;

            #get n_samples, freq_obs, and emp_rt
            stats = [obs.get_quantiles_stats(quantiles) for obs in obs_nodes]
            n_samples = sum([x['n_samples'] for x in stats])
            freq_obs = sum(np.array([x['freq_obs'] for x in stats]),0)
            emp_rt = np.mean(np.array([x['emp_rt'] for x in stats]),0)

            #get p_upper
            p_upper = np.mean(np.array([obs.empirical_quantiles(quantiles)[2] for obs in obs_nodes]),0)

            #set average quantiles  to have the same statitics
            obs_knode = [x for x in self.knodes if x.name == 'wfpt'][0]
            node_name = obs_knode.create_node_name(tag) #get node name
            average_node = average_model.nodes_db.loc[node_name]['node'] #get the average node
            average_node.set_quantiles_stats(quantiles, n_samples, emp_rt, freq_obs, p_upper) #set the quantiles

        return average_model

    def optimize(self, method, quantiles=(.1, .3, .5, .7, .9 ), n_runs=3, n_bootstraps=0, parallel_profile=None):
        """
        Optimize model using ML, chi^2 or G^2.

        :Input:
            method : str
                Optimization method ('ML', 'chisquare' or 'gsquare').

            quantiles : tuple
                A sequence of quantiles to be used for chi^2 and G^2.
                Default values are the ones used by Ratcliff (.1, .3, .5, .7, .9).

            n_runs : int <default=3>
                Number of attempts to optimize.

            n_bootstraps : int <default=0>
                Number of bootstrap iterations.

            parrall_profile : str <default=None>
                IPython profile for parallelization.

        :Output:
            results <dict> - a results dictionary of the parameters values.

        :Note:
            The values of the nodes in single subject model is updated according to the results.
            The nodes of group models are not updated
        """

        results = self._run_optimization(method=method, quantiles=quantiles, n_runs=n_runs)

        #bootstrap if requested
        if n_bootstraps == 0:
            return results

        #init DataFrame to save results
        res =  pd.DataFrame(np.zeros((n_bootstraps, len(self.values))), columns=list(self.values.keys()))

        #prepare view for parallelization
        if parallel_profile is not None: #create view
            client = parallel.Client(profile=parallel_profile)
            view = client.load_balanced_view()
            runs_list = [None] * n_bootstraps
        else:
            view = None

        #define single iteration bootstrap function
        def single_bootstrap(data,
                             accumulator_class=self.__class__, class_kwargs=self._kwargs,
                             method=method, quantiles=quantiles, n_runs=n_runs):

            #resample data
            new_data = data.iloc[np.random.randint(0, len(data), len(data))]
            new_data = new_data.set_index(pd.Index(list(range(len(data)))))
            h = accumulator_class(new_data, **class_kwargs)

            #run optimization
            h._run_optimization(method=method, quantiles=quantiles, n_runs=n_runs)

            return pd.Series(h.values, dtype=np.float)

        #bootstrap iterations
        for i_strap in range(n_bootstraps):
            if view is None:
                res.iloc[i_strap] = single_bootstrap(self.data)
            else:
                # append to job queue
                runs_list[i_strap] = view.apply_async(single_bootstrap, self.data)

        #get parallel results
        if view is not None:
            view.wait(runs_list)
            for i_strap in range(n_bootstraps):
                res.iloc[i_strap] = runs_list[i_strap].get()

        #get statistics
        stats = res.describe()
        for q in [2.5, 97.5]:
            stats = stats.append(pd.DataFrame(res.quantile(q/100.), columns=[repr(q) + '%']).T)

        self.bootstrap_stats = stats.sort_index()
        return results

    def _run_optimization(self, method, quantiles, n_runs):
        """function used by optimize.
        """

        if method == 'ML':
            if self.is_group_model:
                raise TypeError("optimization method is not defined for group models")
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

         compute_stats <boolean> - whether to copmute the quantile stats using the node's
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
        for i_run in range(n_runs):
            #initalize values to a random point
            values_iter = 0
            while inf_objective:
                values_iter += 1
                values = original_values + np.random.randn(len(values))*(2**-values_iter)
                self.set_values(dict(list(zip(names, values))))
                inf_objective = np.isinf(objective(values))

            #optimze
            try:
                res_tuple = fmin_powell(objective, values, full_output=True, maxiter=100, maxfun=50000)
            except Exception:
                res_tuple = fmin(objective, values, full_output=True, maxiter=100, maxfun=50000)
            all_results.append(res_tuple)

            #reset inf_objective so values be resampled
            inf_objective = True

        #get best results
        best_idx = np.nanargmin([x[1] for x in all_results])
        best_values = all_results[best_idx][0]
        self.set_values(dict(list(zip(names, best_values))))
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

    def _create_family_normal(self, name, value=0, g_mu=None,
                             g_tau=15**-2, std_lower=1e-10,
                             std_upper=100, std_value=.1):
        """Create a family of knodes. A family is a group of knodes
        that belong together.

        For example, a family could consist of the following distributions:
        * group mean g_mean (Normal(g_mu, g_tau))
        * group standard deviation g_std (Uniform(std_lower, std_upper))
        * transform node g_std_trans for g_std (x -> x**-2)
        * subject (Normal(g_mean, g_std_trans))

        In fact, if is_group_model is True and the name does not appear in
        group_only nodes, this is the family that will be created.

        Otherwise, only a Normal knode will be returned.

        :Arguments:
            name : str
                Name of the family. Each family member will have this name prefixed.

        :Optional:
            value : float
                Starting value.
            g_mu, g_tau, std_lower, std_upper, std_value : float
                The hyper parameters for the different family members (see above).

        :Returns:
            OrderedDict: member name -> member Knode
        """
        if g_mu is None:
            g_mu = value

        knodes = OrderedDict()

        if self.is_group_model and name not in self.group_only_nodes:
            g = Knode(pm.Normal, '%s' % name, mu=g_mu, tau=g_tau,
                      value=value, depends=self.depends[name])
            depends_std = self.depends[name] if self.std_depends else ()
            std = Knode(pm.Uniform, '%s_std' % name, lower=std_lower,
                        upper=std_upper, value=std_value, depends=depends_std)
            tau = Knode(pm.Deterministic, '%s_tau' % name,
                        doc='%s_tau' % name, eval=lambda x: x**-2, x=std,
                        plot=False, trace=False, hidden=True)
            subj = Knode(pm.Normal, '%s_subj' % name, mu=g, tau=tau,
                         value=value, depends=('subj_idx',),
                         subj=True, plot=self.plot_subjs)
            knodes['%s'%name] = g
            knodes['%s_std'%name] = std
            knodes['%s_tau'%name] = tau
            knodes['%s_bottom'%name] = subj

        else:
            subj = Knode(pm.Normal, name, mu=g_mu, tau=g_tau,
                         value=value, depends=self.depends[name])

            knodes['%s_bottom'%name] = subj

        return knodes


    def _create_family_trunc_normal(self, name, value=0, lower=None,
                                   upper=None, std_lower=1e-10,
                                   std_upper=100, std_value=.1):
        """Similar to _create_family_normal() but creates a Uniform
        group distribution and a truncated subject distribution.

        See _create_family_normal() help for more information.

        """
        knodes = OrderedDict()

        if self.is_group_model and name not in self.group_only_nodes:
            g = Knode(pm.Uniform, '%s' % name, lower=lower,
                      upper=upper, value=value, depends=self.depends[name])

            depends_std = self.depends[name] if self.std_depends else ()
            std = Knode(pm.Uniform, '%s_std' % name, lower=std_lower,
                        upper=std_upper, value=std_value, depends=depends_std)
            tau = Knode(pm.Deterministic, '%s_tau' % name,
                        doc='%s_tau' % name, eval=lambda x: x**-2, x=std,
                        plot=False, trace=False, hidden=True)
            subj = Knode(pm.TruncatedNormal, '%s_subj' % name, mu=g,
                         tau=tau, a=lower, b=upper, value=value,
                         depends=('subj_idx',), subj=True, plot=self.plot_subjs)

            knodes['%s'%name] = g
            knodes['%s_std'%name] = std
            knodes['%s_tau'%name] = tau
            knodes['%s_bottom'%name] = subj

        else:
            subj = Knode(pm.Uniform, name, lower=lower,
                         upper=upper, value=value,
                         depends=self.depends[name])
            knodes['%s_bottom'%name] = subj

        return knodes

    def _create_family_normal_non_centered(self, name, value=0, g_mu=None,
                             g_tau=15**-2, std_lower=1e-10,
                             std_upper=100, std_value=.1):
        """Similar to _create_family_normal() but using a non-centered 
        approach to estimating individual differences.

        See _create_family_normal() help for more information.

        """
        if g_mu is None:
            g_mu = value

        knodes = OrderedDict()

        if self.is_group_model and name not in self.group_only_nodes:
            g = Knode(pm.Normal, '%s' % name, mu=g_mu, tau=g_tau,
                      value=value, depends=self.depends[name])
            depends_std = self.depends[name] if self.std_depends else ()
            std = Knode(pm.Uniform, '%s_std' % name, lower=std_lower,
                        upper=std_upper, value=std_value, depends=depends_std)
            offset_subj = Knode(pm.Normal, '%s_offset_subj' % name, mu=0, tau=5**-2,
                                value=0, depends=('subj_idx',),
                                subj=True, hidden=True, plot=False)
            subj = Knode(pm.Deterministic, '%s_subj'%name, eval=lambda x,y,z: x+y*z,
                         x=g,y=offset_subj,z=std,
                         depends=('subj_idx',), plot=self.plot_subjs,
                         trace=True, hidden=False, subj=True)
            knodes['%s'%name] = g
            knodes['%s_std'%name] = std
            knodes['%s_offset_subj'%name] = offset_subj
            knodes['%s_bottom'%name] = subj

        else:
            subj = Knode(pm.Normal, name, mu=g_mu, tau=g_tau,
                         value=value, depends=self.depends[name])

            knodes['%s_bottom'%name] = subj

        return knodes
        
    def _create_family_invlogit(self, name, value, g_mu=None, g_tau=15**-2,
                               std_std=0.2, std_value=.1):
        """Similar to _create_family_normal_normal_hnormal() but adds a invlogit
        transform knode to the subject and group mean nodes. This is useful
        when the parameter space is restricted from [0, 1].

        See _create_family_normal_normal_hnormal() help for more information.

        """

        if g_mu is None:
            g_mu = value

        # logit transform values
        value_trans = np.log(value) - np.log(1-value)
        g_mu_trans = np.log(g_mu) - np.log(1-g_mu)

        knodes = OrderedDict()

        if self.is_group_model and name not in self.group_only_nodes:
            g_trans = Knode(pm.Normal,
                            '%s_trans'%name,
                            mu=g_mu_trans,
                            tau=g_tau,
                            value=value_trans,
                            depends=self.depends[name],
                            plot=False,
                            hidden=True
            )

            g = Knode(pm.InvLogit, name, ltheta=g_trans, plot=True,
                      trace=True)

            depends_std = self.depends[name] if self.std_depends else ()
            std = Knode(pm.HalfNormal, '%s_std' % name, tau=std_std**-2,
                        value=std_value, depends=depends_std)

            tau = Knode(pm.Deterministic, '%s_tau'%name, doc='%s_tau'
                        % name, eval=lambda x: x**-2, x=std,
                        plot=False, trace=False, hidden=True)

            subj_trans = Knode(pm.Normal, '%s_subj_trans'%name,
                               mu=g_trans, tau=tau, value=value_trans,
                               depends=('subj_idx',), subj=True,
                               plot=False, hidden=True)

            subj = Knode(pm.InvLogit, '%s_subj'%name,
                         ltheta=subj_trans, depends=('subj_idx',),
                         plot=self.plot_subjs, trace=True, subj=True)

            knodes['%s_trans'%name]      = g_trans
            knodes['%s'%name]            = g
            knodes['%s_std'%name]        = std
            knodes['%s_tau'%name]        = tau

            knodes['%s_subj_trans'%name] = subj_trans
            knodes['%s_bottom'%name]     = subj

        else:
            g_trans = Knode(pm.Normal, '%s_trans'%name, mu=g_mu_trans,
                            tau=g_tau, value=value_trans,
                            depends=self.depends[name], plot=False, hidden=True)

            g = Knode(pm.InvLogit, '%s'%name, ltheta=g_trans, plot=True,
                      trace=True )

            knodes['%s_trans'%name] = g_trans
            knodes['%s_bottom'%name] = g

        return knodes

    def _create_family_exp(self, name, value=0, g_mu=None,
                           g_tau=15**-2, std_lower=1e-10, std_upper=100, std_value=.1):
        """Similar to create_family_normal() but adds an exponential
        transform knode to the subject and group mean nodes. This is useful
        when the parameter space is restricted from [0, +oo).

        See create_family_normal() help for more information.

        """
        if g_mu is None:
            g_mu = value

        value_trans = np.log(value)
        g_mu_trans = np.log(g_mu)

        knodes = OrderedDict()
        if self.is_group_model and name not in self.group_only_nodes:
            g_trans = Knode(pm.Normal, '%s_trans' % name, mu=g_mu_trans,
                            tau=g_tau, value=value_trans,
                            depends=self.depends[name], plot=False, hidden=True)

            g = Knode(pm.Deterministic, '%s'%name, eval=lambda x: np.exp(x),
                      x=g_trans, plot=True)

            depends_std = self.depends[name] if self.std_depends else ()
            std = Knode(pm.Uniform, '%s_std' % name, lower=std_lower, upper=std_upper,
                        value=std_value, depends=depends_std)

            tau = Knode(pm.Deterministic, '%s_tau' % name, eval=lambda x: x**-2,
                        x=std, plot=False, trace=False, hidden=True)

            subj_trans = Knode(pm.Normal, '%s_subj_trans'%name, mu=g_trans,
                         tau=tau, value=value_trans, depends=('subj_idx',),
                         subj=True, plot=False, hidden=True)

            subj = Knode(pm.Deterministic, '%s_subj'%name, eval=lambda x: np.exp(x),
                         x=subj_trans,
                         depends=('subj_idx',), plot=self.plot_subjs,
                         trace=True, subj=True)

            knodes['%s_trans'%name]      = g_trans
            knodes['%s'%name]            = g
            knodes['%s_std'%name]        = std
            knodes['%s_tau'%name]        = tau
            knodes['%s_subj_trans'%name] = subj_trans
            knodes['%s_bottom'%name]     = subj

        else:
            g_trans = Knode(pm.Normal, '%s_trans' % name, mu=g_mu_trans,
                            tau=g_tau, value=value_trans,
                            depends=self.depends[name], plot=False, hidden=True)

            g = Knode(pm.Deterministic, '%s'%name, doc='%s'%name, eval=lambda x: np.exp(x), x=g_trans, plot=True)
            knodes['%s_trans'%name] = g_trans
            knodes['%s_bottom'%name] = g

        return knodes

    def _create_family_normal_normal_hnormal(self, name, value=0, g_mu=None,
                             g_tau=15**-2, std_std=2,
                             std_value=.1):
        """Create a family of knodes. A family is a group of knodes
        that belong together.

        For example, a family could consist of the following distributions:
        * group mean g_mean (Normal(g_mu, g_tau))
        * group standard deviation g_std (Uniform(std_lower, std_upper))
        * transform node g_std_trans for g_std (x -> x**-2)
        * subject (Normal(g_mean, g_std_trans))

        In fact, if is_group_model is True and the name does not appear in
        group_only nodes, this is the family that will be created.

        Otherwise, only a Normal knode will be returned.

        :Arguments:
            name : str
                Name of the family. Each family member will have this name prefixed.

        :Optional:
            value : float
                Starting value.
            g_mu, g_tau, std_lower, std_upper, std_value : float
                The hyper parameters for the different family members (see above).

        :Returns:
            OrderedDict: member name -> member Knode
        """
        if g_mu is None:
            g_mu = value

        knodes = OrderedDict()

        if self.is_group_model and name not in self.group_only_nodes:
            g = Knode(pm.Normal, '%s' % name, mu=g_mu, tau=g_tau,
                      value=value, depends=self.depends[name])
            depends_std = self.depends[name] if self.std_depends else ()
            std = Knode(pm.HalfNormal, '%s_std' % name, tau=std_std**-2,
                        value=std_value, depends=depends_std)
            tau = Knode(pm.Deterministic, '%s_tau' % name,
                        doc='%s_tau' % name, eval=lambda x: x**-2, x=std,
                        plot=False, trace=False, hidden=True)
            subj = Knode(pm.Normal, '%s_subj' % name, mu=g, tau=tau,
                         value=value, depends=('subj_idx',),
                         subj=True, plot=self.plot_subjs)
            knodes['%s'%name] = g
            knodes['%s_std'%name] = std
            knodes['%s_tau'%name] = tau
            knodes['%s_bottom'%name] = subj

        else:
            subj = Knode(pm.Normal, name, mu=g_mu, tau=g_tau,
                         value=value, depends=self.depends[name])

            knodes['%s_bottom'%name] = subj

        return knodes


    def _create_family_gamma_gamma_hnormal(self, name, value=1, g_mean=1, g_std=1, std_std=2, std_value=.1):
        """Similar to _create_family_normal_normal_hnormal() but adds an exponential
        transform knode to the subject and group mean nodes. This is useful
        when the parameter space is restricted from [0, +oo).

        See _create_family_normal_normal_hnormal() help for more information.

        """

        knodes = OrderedDict()
        g_shape = (g_mean**2) / (g_std**2)
        g_rate = g_mean / (g_std**2)
        if self.is_group_model and name not in self.group_only_nodes:
            g = Knode(pm.Gamma, name, alpha=g_shape, beta=g_rate,
                            value=g_mean, depends=self.depends[name])
            depends_std = self.depends[name] if self.std_depends else ()
            std = Knode(pm.HalfNormal, '%s_std' % name, tau=std_std**-2,
                        value=std_value, depends=depends_std)

            shape = Knode(pm.Deterministic, '%s_shape' % name, eval=lambda x,y: (x**2)/(y**2),
                        x=g, y=std, plot=False, trace=False, hidden=True)

            rate = Knode(pm.Deterministic, '%s_rate' % name, eval=lambda x,y: x/(y**2),
                        x=g, y=std, plot=False, trace=False, hidden=True)


            subj = Knode(pm.Gamma, '%s_subj'%name, alpha=shape, beta=rate,
                         value=value, depends=('subj_idx',),
                         subj=True, plot=False)

            knodes['%s'%name]            = g
            knodes['%s_std'%name]        = std
            knodes['%s_rate'%name]       = rate
            knodes['%s_shape'%name]      = shape
            knodes['%s_bottom'%name]     = subj

        else:
            g = Knode(pm.Gamma, name, alpha=g_shape, beta=g_rate, value=value,
                            depends=self.depends[name])

            knodes['%s_bottom'%name] = g

        return knodes

class HDDMBase(AccumulatorModel):
    """HDDM base class. Not intended to be used directly. Instead, use hddm.HDDM.
    """

    def __init__(self, data, bias=False, include=(),
                 wiener_params=None, p_outlier=0.05, **kwargs):

        self.default_intervars = kwargs.pop('default_intervars', {'sz': 0, 'st': 0, 'sv': 0})

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

        possible_parameters = ('v', 'a', 't', 'z', 'st', 'sz', 'sv', 'p_outlier', 'alpha')
        assert self.include.issubset(possible_parameters), """Received an invalid parameter using the 'include' keyword.
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
        self.cdf_range = (-cdf_bound, cdf_bound)

        #set wfpt class
        self.wfpt_class = hddm.likelihoods.generate_wfpt_stochastic_class(wp, cdf_range=self.cdf_range)

        super(HDDMBase, self).__init__(data, **kwargs)

    def __getstate__(self):
        d = super(HDDMBase, self).__getstate__()
        del d['wfpt_class']

        return d

    def __setstate__(self, d):
        self.wfpt_class = hddm.likelihoods.generate_wfpt_stochastic_class(d['wiener_params'], cdf_range=d['cdf_range'])
        super(HDDMBase, self).__setstate__(d)

    def _create_wfpt_parents_dict(self, knodes):
        wfpt_parents = OrderedDict()
        wfpt_parents['a'] = knodes['a_bottom']
        wfpt_parents['v'] = knodes['v_bottom']
        wfpt_parents['t'] = knodes['t_bottom']

        wfpt_parents['sv'] = knodes['sv_bottom'] if 'sv' in self.include else self.default_intervars['sv']
        wfpt_parents['sz'] = knodes['sz_bottom'] if 'sz' in self.include else self.default_intervars['sz']
        wfpt_parents['st'] = knodes['st_bottom'] if 'st' in self.include else self.default_intervars['st']
        wfpt_parents['z'] = knodes['z_bottom'] if 'z' in self.include else 0.5
        wfpt_parents['p_outlier'] = knodes['p_outlier_bottom'] if 'p_outlier' in self.include else self.p_outlier
        return wfpt_parents

    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)

        return Knode(self.wfpt_class, 'wfpt', observed=True, col_name='rt', **wfpt_parents)

    def create_knodes(self):
        knodes = self._create_stochastic_knodes(self.include)
        knodes['wfpt'] = self._create_wfpt_knode(knodes)

        return list(knodes.values())

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

            save : bool (default=False)
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
        known_args = set(['wiener_params', 'include', 'self', 'bias', 'data','p_outlier'])
        assert known_args.issuperset(init_args), "Arguments of the constructor are not as expected"

        #create the avg model
        avg_model  = self.__class__(self.data, include=self.include, is_group_model=False, p_outlier=self.p_outlier, **self._kwargs)
        return avg_model

if __name__ == "__main__":
    import doctest
    doctest.testmod()
