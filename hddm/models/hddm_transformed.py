from collections import OrderedDict
import inspect

import pymc as pm
import kabuki.step_methods as steps
from hddm.models import HDDMBase

class HDDM(HDDMBase):
    def __init__(self, *args, **kwargs):
        self.use_gibbs_for_mean = kwargs.pop('use_gibbs_for_mean', True)
        self.use_reject_for_std = kwargs.pop('use_reject_for_std', True)

        super(HDDM, self).__init__(*args, **kwargs)

    def pre_sample(self):
        if not self.is_group_model:
            return

        # apply gibbs sampler to normal group nodes
        for name, node_descr in self.iter_group_nodes():
            node = node_descr['node']
            knode_name = node_descr['knode_name'].replace('_trans', '')
            if self.use_gibbs_for_mean and isinstance(node, pm.Normal) and knode_name not in self.group_only_nodes:
                self.mc.use_step_method(steps.kNormalNormal, node)
            if self.use_reject_for_std and isinstance(node, pm.Uniform) and knode_name not in self.group_only_nodes:
                self.mc.use_step_method(steps.UniformPriorNormalstd, node)

    def _create_stochastic_knodes(self, include):
        knodes = OrderedDict()
        if 'a' in include:
            knodes.update(self.create_family_exp('a', value=1))
        if 'v' in include:
            knodes.update(self.create_family_normal('v', value=0))
        if 't' in include:
            knodes.update(self.create_family_exp('t', value=.01))
        if 'sv' in include:
            # TW: Use kabuki.utils.HalfCauchy, S=10, value=1 instead?
            knodes.update(self.create_family_trunc_normal('sv', lower=0, upper=1e3, value=1))
            #knodes.update(self.create_family_exp('sv', value=1))
        if 'sz' in include:
            knodes.update(self.create_family_invlogit('sz', value=.1))
        if 'st' in include:
            knodes.update(self.create_family_exp('st', value=.01))
        if 'z' in include:
            knodes.update(self.create_family_invlogit('z', value=.5))
        if 'p_outlier' in include:
            knodes.update(self.create_family_invlogit('p_outlier', value=0.05))

        return knodes

