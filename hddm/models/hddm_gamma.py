from collections import OrderedDict
import inspect

import pymc as pm
import kabuki.step_methods as steps
from hddm.models import HDDMBase

class HDDMGamma(HDDMBase):
    def __init__(self, *args, **kwargs):
        super(HDDMGamma, self).__init__(*args, **kwargs)

    def pre_sample(self):
        if not self.is_group_model:
            return

        slice_widths = {'a':2, 't':0.5, 'a_var': 0.2, 't_var': 0.15}

        # apply gibbs sampler to normal group nodes
        for name, node_descr in self.iter_group_nodes():
            node = node_descr['node']
            knode_name = node_descr['knode_name'].replace('_trans', '')
            if knode_name in self.group_only_nodes:
                continue
            if knode_name == 'v':
                self.mc.use_step_method(steps.kNormalNormal, node)
            elif knode_name == 'v_var':
                self.mc.use_step_method(steps.UniformPriorNormalstd, node)
            else:
                try:
                    self.mc.use_step_method(steps.SliceStep, node, width=slice_widths[knode_name],
                                        left=0)
                except KeyError:
                    pass



    def _create_stochastic_knodes(self, include):
        knodes = OrderedDict()
        if 'a' in include:
            knodes.update(self.create_family_gamma('a', value=1.5, var_value=0.75))
        if 'v' in include:
            knodes.update(self.create_family_normal('v', value=0))
        if 't' in include:
            knodes.update(self.create_family_gamma('t', value=.03, var_value=0.2))
        if 'sv' in include:
            # TW: Use kabuki.utils.HalfCauchy, S=10, value=1 instead?
            knodes.update(self.create_family_trunc_normal('sv', lower=0, upper=1e3, value=1))
            #knodes.update(self.create_family_exp('sv', value=1))
        if 'sz' in include:
            knodes.update(self.create_family_invlogit('sz', value=.1))
        if 'st' in include:
            knodes.update(self.create_family_exp('st', value=.01))
        if 'z' in self.include:
            knodes.update(self.create_family_invlogit('z', value=.5))
        if 'p_outlier' in include:
            knodes.update(self.create_family_invlogit('p_outlier', value=0.05))

        return knodes

    def _create_an_average_model(self):
        """
        create an average model for group model quantiles optimization.
        """

        #this code only check that the arguments are as expected, i.e. the constructor was not change
        #since we wrote this function
        super_init_function = super(self.__class__, self).__init__
        init_args = set(inspect.getargspec(super_init_function).args)
        known_args = set(['wiener_params', 'include', 'self', 'bias', 'data'])
        assert known_args == init_args, "Arguments of the constructor are not as expected"

        #create the avg model
        avg_model  = self.__class__(self.data, include=self.include, is_group_model=False, **self._kwargs)
        return avg_model
