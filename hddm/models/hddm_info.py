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

        std_depends : bool (default=False)
             Should the depends_on keyword affect the group std node.
             If True it means that both, group mean and std will be split
             by condition.

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
        self.slice_widths = {'a':1, 't':0.01, 'a_std': 1, 't_std': 0.15, 'sz': 1.1, 'v': 1.5,
                             'st': 0.1, 'sv': 3, 'z_trans': 0.2, 'z': 0.1,
                             'p_outlier':1., 'v_std': 1, 'alpha': 1.5, 'pos_alpha': 1.5}
        self.emcee_dispersions = {'a':1, 't': 0.1, 'a_std': 1, 't_std': 0.15, 'sz': 1.1, 'v': 1.5,
                                  'st': 0.1, 'sv': 3, 'z_trans': 0.2, 'z': 0.1,
                                  'p_outlier':1., 'v_std': 1, 'alpha': 1.5, 'pos_alpha': 1.5}


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
            knodes.update(self._create_family_gamma_gamma_hnormal('a', g_mean=1.5, g_std=0.75, std_std=2, std_value=0.1, value=1))
        if 'v' in include:
            knodes.update(self._create_family_normal_normal_hnormal('v', value=2, g_mu=2, g_tau=3**-2, std_std=2))
        if 't' in include:
            knodes.update(self._create_family_gamma_gamma_hnormal('t', g_mean=.4, g_std=0.2, value=0.001, std_std=1, std_value=0.2))
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
