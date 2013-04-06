from collections import OrderedDict
import inspect
import numpy as np
import pymc as pm
import kabuki.step_methods as steps
from hddm.models import HDDMBase
from kabuki.hierarchical import Knode

class HDDM(HDDMBase):
    """Create hierarchical drift-diffusion model in which each subject
    has a set of parameters that are constrained by a group distribution.

    :Arguments:
        data : pandas.DataFrame
            Input data with a row for each trial.
            Must contain the following columns:
              * 'rt': Reaction time of trial in seconds.
              * 'response': Binary response (e.g. 0->error, 1->correct)
              * 'subj_idx': A unique ID (int) of each subject.
              * Other user-defined columns that can be used in depends_on
                keyword.

    :Optional:
        informative : bool <default=True>
            Whether to use informative priors (True) or vague priors
            (False).  Information about the priors can be found in the
            methods section.  If you run a classical DDM experiment
            you should use this. However, if you apply the DDM to a
            novel domain like saccade data where RTs are much lower,
            or RTs of rats, you should probably set this to False.

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

        p_outlier : double (default=0)
            The probability of outliers in the data. if p_outlier is passed in the
            'include' argument, then it is estimated from the data and the value passed
            using the p_outlier argument is ignored.

        default_intervars : dict (default = {'sz': 0, 'st': 0, 'sv': 0})
            Fix intertrial variabilities to a certain value. Note that this will only
            have effect for variables not estimated from the data.

        plot_var : bool
             Plot group variability parameters when calling pymc.Matplot.plot()
             (i.e. variance of Normal distribution.)

        trace_subjs : bool
             Save trace for subjs (needed for many
             statistics so probably a good idea.)

        wiener_params : dict
             Parameters for wfpt evaluation and
             numerical integration.

         :Parameters:
             * err: Error bound for wfpt (default 1e-4)
             * n_st: Maximum depth for numerical integration for st (default 2)
             * n_sz: Maximum depth for numerical integration for Z (default 2)
             * use_adaptive: Whether to use adaptive numerical integration (default True)
             * simps_err: Error bound for Simpson integration (default 1e-3)

    :Example:
        >>> data, params = hddm.generate.gen_rand_data() # gen data
        >>> model = hddm.HDDM(data) # create object
        >>> mcmc.sample(5000, burn=20) # Sample from posterior

    """

    def __init__(self, *args, **kwargs):
        self.slice_widths = {'a':1, 't':0.01, 'a_var': 1, 't_var': 0.15, 'sz': 1.1, 'v': 1.5,
                             'st': 0.1, 'sv': 3, 'z_trans': 0.2, 'z': 0.1,
                             'p_outlier':1., 'v_var': 1}

        self.is_informative = kwargs.pop('informative', True)

        super(HDDM, self).__init__(*args, **kwargs)

    def _create_stochastic_knodes(self, include):
        if self.is_informative:
            return self._create_stochastic_knodes_info(include)
        else:
            return self._create_stochastic_knodes_noninfo(include)

    def _create_stochastic_knodes_info(self, include):
        knodes = OrderedDict()
        if 'a' in include:
            knodes.update(self._create_family_gamma_gamma_hnormal('a', g_mean=1.5, g_std=0.75, std_std=2, var_value=0.1, value=1))
        if 'v' in include:
            knodes.update(self._create_family_normal_normal_hnormal('v', value=2, g_mu=2, g_tau=3**-2, std_std=2))
        if 't' in include:
            knodes.update(self._create_family_gamma_gamma_hnormal('t', g_mean=.4, g_std=0.2, value=0.001, std_std=1, var_value=0.2))
        if 'sv' in include:
            knodes['sv_bottom'] = Knode(pm.HalfNormal, 'sv', tau=2**-2, value=1, depends=self.depends['sv'])
        if 'sz' in include:
            knodes['sz_bottom'] = Knode(pm.Beta, 'sz', alpha=1, beta=3, value=0.01, depends=self.depends['sz'])
        if 'st' in include:
            knodes['st_bottom'] = Knode(pm.HalfNormal, 'st', tau=0.3**-2, value=0.001, depends=self.depends['st'])
        if 'z' in include:
            knodes.update(self._create_family_invlogit('z', value=.5, g_tau=0.5**-2, std_std=0.05))
        if 'p_outlier' in include:
            knodes['p_outlier_bottom'] = Knode(pm.Beta, 'p_outlier', alpha=1, beta=15, value=0.01, depends=self.depends['p_outlier'])

        return knodes

    def _create_stochastic_knodes_noninfo(self, include):
        knodes = OrderedDict()
        if 'a' in include:
            knodes.update(self._create_family_trunc_normal('a', lower=1e-3, upper=1e3, value=1))
        if 'v' in include:
            knodes.update(self._create_family_normal_normal_hnormal('v', value=0, g_tau=50**-2, std_std=10))
        if 't' in include:
            knodes.update(self._create_family_trunc_normal('t', lower=1e-3, upper=1e3, value=.01))
        if 'sv' in include:
            knodes['sv_bottom'] = Knode(pm.Uniform, 'sv', lower=1e-6, upper=1e3, value=1, depends=self.depends['sv'])
        if 'sz' in include:
            knodes['sz_bottom'] = Knode(pm.Beta, 'sz', alpha=1, beta=1, value=0.01, depends=self.depends['sz'])
        if 'st' in include:
            knodes['st_bottom'] = Knode(pm.Uniform, 'st', lower=1e-6, upper=1e3, value=0.01, depends=self.depends['st'])
        if 'z' in include:
            knodes.update(self._create_family_invlogit('z', value=.5, g_tau=10**-2, std_std=0.5))
        if 'p_outlier' in include:
            knodes['p_outlier_bottom'] = Knode(pm.Beta, 'p_outlier', alpha=1, beta=1, value=0.01, depends=self.depends['p_outlier'])

        return knodes

    def pre_sample(self, use_slice=True):
        for name, node_descr in self.iter_stochastics():
            node = node_descr['node']
            if isinstance(node, pm.Normal) and np.all([isinstance(x, pm.Normal) for x in node.extended_children]):
                self.mc.use_step_method(steps.kNormalNormal, node)
            else:
                knode_name = node_descr['knode_name'].replace('_subj', '')
                if knode_name in ['st', 'sv', 'sz']:
                    left = 0
                else:
                    left = None
                self.mc.use_step_method(steps.SliceStep, node, width=self.slice_widths.get(knode_name, 1),
                                        left=left, maxiter=5000)


    def _create_family_normal_normal_hnormal(self, name, value=0, g_mu=None,
                             g_tau=15**-2, std_std=2,
                             var_value=.1):
        """Create a family of knodes. A family is a group of knodes
        that belong together.

        For example, a family could consist of the following distributions:
        * group mean g_mean (Normal(g_mu, g_tau))
        * group variability g_var (Uniform(var_lower, var_upper))
        * transform node g_var_trans for g_var (x -> x**-2)
        * subject (Normal(g_mean, g_var_trans))

        In fact, if is_group_model is True and the name does not appear in
        group_only nodes, this is the family that will be created.

        Otherwise, only a Normal knode will be returned.

        :Arguments:
            name : str
                Name of the family. Each family member will have this name prefixed.

        :Optional:
            value : float
                Starting value.
            g_mu, g_tau, var_lower, var_upper, var_value : float
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
            var = Knode(pm.HalfNormal, '%s_var' % name, tau=std_std**-2, value=var_value)
            tau = Knode(pm.Deterministic, '%s_tau' % name,
                        doc='%s_tau' % name, eval=lambda x: x**-2, x=var,
                        plot=False, trace=False, hidden=True)
            subj = Knode(pm.Normal, '%s_subj' % name, mu=g, tau=tau,
                         value=value, depends=('subj_idx',),
                         subj=True, plot=self.plot_subjs)
            knodes['%s'%name] = g
            knodes['%s_var'%name] = var
            knodes['%s_tau'%name] = tau
            knodes['%s_bottom'%name] = subj

        else:
            subj = Knode(pm.Normal, name, mu=g_mu, tau=g_tau,
                         value=value, depends=self.depends[name])

            knodes['%s_bottom'%name] = subj

        return knodes


    def _create_family_gamma_gamma_hnormal(self, name, value=1, g_mean=1, g_std=1, std_std=2, var_value=.1):
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

            var = Knode(pm.HalfNormal, '%s_var' % name, tau=std_std**-2, value=var_value)

            shape = Knode(pm.Deterministic, '%s_shape' % name, eval=lambda x,y: (x**2)/(y**2),
                        x=g, y=var, plot=False, trace=False, hidden=True)

            rate = Knode(pm.Deterministic, '%s_rate' % name, eval=lambda x,y: x/(y**2),
                        x=g, y=var, plot=False, trace=False, hidden=True)


            subj = Knode(pm.Gamma, '%s_subj'%name, alpha=shape, beta=rate,
                         value=value, depends=('subj_idx',),
                         subj=True, plot=False)

            knodes['%s'%name]            = g
            knodes['%s_var'%name]        = var
            knodes['%s_rate'%name]       = rate
            knodes['%s_shape'%name]      = shape
            knodes['%s_bottom'%name]     = subj

        else:
            g = Knode(pm.Gamma, name, alpha=g_shape, beta=g_rate, value=value,
                            depends=self.depends[name])

            knodes['%s_bottom'%name] = g

        return knodes

    def _create_family_invlogit(self, name, value, g_mu=None, g_tau=15**-2,
                               std_std=0.2, var_value=.1):
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

            var = Knode(pm.HalfNormal, '%s_var' % name, tau=std_std**-2, value=var_value)

            tau = Knode(pm.Deterministic, '%s_tau'%name, doc='%s_tau'
                        % name, eval=lambda x: x**-2, x=var,
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
            knodes['%s_var'%name]        = var
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

    def _create_an_average_model(self):
        """
        create an average model for group model quantiles optimization.
        """

        #this code only check that the arguments are as expected, i.e. the constructor was not change
        #since we wrote this function
        super_init_function = super(self.__class__, self).__init__
        init_args = set(inspect.getargspec(super_init_function).args)
        known_args = set(['wiener_params', 'include', 'self', 'bias', 'data', 'p_outlier'])
        assert known_args.issuperset(init_args), "Arguments of the constructor are not as expected"

        #create the avg model
        avg_model  = self.__class__(self.data, include=self.include, is_group_model=False, **self._kwargs)
        return avg_model
