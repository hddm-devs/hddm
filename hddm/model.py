"""
.. module:: HDDM
   :platform: Agnostic
   :synopsis: Defvalueion of HDDM models.

.. moduleauthor:: Thomas Wiecki <thomas.wiecki@gmail.com>
                  Imri Sofer <imrisofer@gmail.com>


"""

from __future__ import division
from collections import OrderedDict

import numpy as np
import pymc as pm

import hddm
import kabuki
import kabuki.step_methods as steps
import scipy as sp
import inspect

from kabuki.hierarchical import Knode
from copy import deepcopy
from scipy.optimize import fmin_powell
from scipy import stats

class AccumulatorModel(kabuki.Hierarchical):
    def __init__(self, data, **kwargs):
        # Flip sign for lower boundary RTs
        data = hddm.utils.flip_errors(data)

        self.group_only_nodes = kwargs.pop('group_only_nodes', ())

        super(AccumulatorModel, self).__init__(data, **kwargs)


    def _create_an_average_model(self):
        raise NotImplementedError, "This method has to be overloaded. See HDDMBase."


    def quantiles_chisquare_optimization(self, quantiles=(.1, .3, .5, .7, .9 )):
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
                n_samples = sum([x.n_samples for x in stats])
                freq_obs = sum(np.array([x.freq_obs for x in stats]),0)
                emp_rt = np.mean(np.array([x.emp_rt for x in stats]),0)

                #set average quantiles  to have the same statitics
                obs_knode = [x for x in self.knodes if x.name == 'wfpt'][0]
                node_name = obs_knode.create_node_name(tag) #get node name
                average_node = average_model.nodes_db.ix[node_name]['node'] #get the average node
                average_node.set_quantiles_stats(n_samples, emp_rt, freq_obs) #set the quantiles

            #optimize
            results = average_model._quantiles_chisquare_optimization_single(quantiles=quantiles, compute_stats=False)


        #run optimization for single subject model
        else:
            results = self._quantiles_chisquare_optimization_single(quantiles=quantiles, compute_stats=True)

        return results


    def _quantiles_chisquare_optimization_single(self, quantiles, compute_stats):
        """
        function used by quantiles_chisquare_optimization to fit the a single subject model
        Input:
         quantiles <sequance> - same as in quantiles_chisquare_optimization
         cmopute_stats <boolean> - whether to copmute the quantile stats using the node's
             compute_quantiles_stats method

        Output:
            results <dict> - same as in quantiles_chisquare_optimization
        """

        #get obs_nodes
        obs_nodes = self.get_observeds()['node']

        #set quantiles for each observed_node (if needed)
        if compute_stats:
            [obs.compute_quantiles_stats(quantiles) for obs in obs_nodes]

        #get all stochastic parents of observed nodes
        db = self.nodes_db
        parents = db[(db.stochastic == True) & (db.observed == False)]['node']
        values = [x.value for x in parents]

        #define objective
        def objective(values):
            for (i, value) in enumerate(values):
                parents[i].value = value
            return sum([obs.chisquare() for obs in obs_nodes])

        #optimze
        fmin_powell(objective, values)
        results = self.values

        return results

    def ML_optimization(self):
        """
        optimization of model using Maximum likelihood

        Output:
            results <dict> - same as in quantiles_chisquare_optimization
        """

        if self.is_group_model:
            raise TypeError, "ML optimization is not defined for group models"

        #get obs_nodes
        obs_nodes = self.get_observeds()['node']


        #get all stochastic parents of observed nodes
        db = self.nodes_db
        parents = db[(db.stochastic == True) & (db.observed == False)]['node']
        values = [x.value for x in parents]

        #define objective
        def objective(values):
            for (i, value) in enumerate(values):
                parents[i].value = value
            try:
                return -sum([obs.logp for obs in obs_nodes])
            except pm.ZeroProbability:
                return np.inf

        #optimze
        fmin_powell(objective, values)
        results = self.values

        return results




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
                 wiener_params=None, **kwargs):

        self._kwargs = kwargs

        self.include = set()
        if include is not None:
            if include == 'all':
                [self.include.add(param) for param in ('st','sv','sz')]
            elif isinstance(include, str):
                self.include.add(include)
            else:
                [self.include.add(param) for param in include]

        if bias:
            self.include.add('z')

        if wiener_params is None:
            self.wiener_params = {'err': 1e-4, 'n_st':2, 'n_sz':2,
                                  'use_adaptive':1,
                                  'simps_err':1e-3}
        else:
            self.wiener_params = wiener_params
        wp = self.wiener_params

        #set cdf_range
        cdf_bound = max(np.abs(data['rt'])) + 1;
        cdf_range = (-cdf_bound, cdf_bound)

        #set wfpt class
        self.wfpt_class = hddm.likelihoods.generate_wfpt_stochastic_class(wp, cdf_range=cdf_range)

        super(HDDMBase, self).__init__(data, **kwargs)

    def create_wfpt_knode(self, knodes):
        wfpt_parents = OrderedDict()
        wfpt_parents['a'] = knodes['a_bottom']
        wfpt_parents['v'] = knodes['v_bottom']
        wfpt_parents['t'] = knodes['t_bottom']

        wfpt_parents['sv'] = knodes['sv_bottom'] if 'sv' in self.include else 0
        wfpt_parents['sz'] = knodes['sz_bottom'] if 'sz' in self.include else 0
        wfpt_parents['st'] = knodes['st_bottom'] if 'st' in self.include else 0
        wfpt_parents['z'] = knodes['z_bottom'] if 'z' in self.include else 0.5

        return Knode(self.wfpt_class, 'wfpt', observed=True, col_name='rt', **wfpt_parents)

    def plot_posterior_predictive(self, *args, **kwargs):
        if 'value_range' not in kwargs:
            kwargs['value_range'] = np.linspace(-5, 5, 100)
        kabuki.analyze.plot_posterior_predictive(self, *args, **kwargs)


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


class HDDMTruncated(HDDMBase):
   def create_knodes(self):
        """Returns list of model parameters.
        """

        knodes = OrderedDict()
        knodes.update(self.create_family_trunc_normal('a', lower=1e-3, upper=1e3, value=1))
        knodes.update(self.create_family_normal('v', value=0))
        knodes.update(self.create_family_trunc_normal('t', lower=1e-3, upper=1e3, value=.01))
        if 'sv' in self.include:
            knodes.update(self.create_family_trunc_normal('sv', lower=0, upper=1e3, value=1))
        if 'sz' in self.include:
            knodes.update(self.create_family_trunc_normal('sz', lower=0, upper=1, value=.1))
        if 'st' in self.include:
            knodes.update(self.create_family_trunc_normal('st', lower=0, upper=1e3, value=.01))
        if 'z' in self.include:
            knodes.update(self.create_family_trunc_normal('z', lower=0, upper=1, value=.5))

        knodes['wfpt'] = self.create_wfpt_knode(knodes)

        return knodes.values()


class HDDM(HDDMBase):
    def __init__(self, *args, **kwargs):
        self.use_gibbs = kwargs.pop('use_gibbs_for_mean', True)
        self.use_slice = kwargs.pop('use_slice_for_std', True)

        super(self.__class__, self).__init__(*args, **kwargs)

    def pre_sample(self):
        if not self.is_group_model:
            return

        # apply gibbs sampler to normal group nodes
        for name, node_descr in self.iter_group_nodes():
            node = node_descr['node']
            knode_name = node_descr['knode_name'].replace('_trans', '')
            if self.use_gibbs and isinstance(node, pm.Normal) and knode_name not in self.group_only_nodes:
                self.mc.use_step_method(steps.kNormalNormal, node)
            if self.use_slice and isinstance(node, pm.Uniform) and knode_name not in self.group_only_nodes:
                self.mc.use_step_method(steps.UniformPriorNormalstd, node)

    def create_knodes(self):
        knodes = OrderedDict()
        knodes.update(self.create_family_exp('a', value=1))
        knodes.update(self.create_family_normal('v', value=0))
        knodes.update(self.create_family_exp('t', value=.01))
        if 'sv' in self.include:
            # TW: Use kabuki.utils.HalfCauchy, S=10, value=1 instead?
            knodes.update(self.create_family_trunc_normal('sv', lower=0, upper=1e3, value=1))
            #knodes.update(self.create_family_exp('sv', value=1))
        if 'sz' in self.include:
            knodes.update(self.create_family_invlogit('sz', value=.1))
        if 'st' in self.include:
            knodes.update(self.create_family_exp('st', value=.01))
        if 'z' in self.include:
            knodes.update(self.create_family_invlogit('z', value=.5))

        knodes['wfpt'] = self.create_wfpt_knode(knodes)

        return knodes.values()

if __name__ == "__main__":
    import doctest
    doctest.testmod()
