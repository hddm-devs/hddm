import hddm
from hddm.model import Base
import pymc as pm
from kabuki import Parameter

class HLBA(Base):
    param_names = (('a',True), ('z',True), ('t',True), ('V',True), ('v0',True), ('v1',True), ('lba',False))

    def __init__(self, data, model_type=None, trace_subjs=True, normalize_v=True, no_bias=True, fix_sv=None, init=False, exclude=None, **kwargs):
        super(self.__class__, self).__init__(data, **kwargs)

        # LBA model
        self.normalize_v = normalize_v
        self.init_params = {}
            
        self.param_ranges = {'a_lower': .2,
                             'a_upper': 4.,
                             'v_lower': 0.1,
                             'v_upper': 3.,
                             'z_lower': .0,
                             'z_upper': 2.,
                             't_lower': .05,
                             't_upper': 2.,
                             'V_lower': .2,
                             'V_upper': 2.}
            
        if self.normalize_v:
            self.param_ranges['v_lower'] = 0.
            self.param_ranges['v_upper'] = 1.

    def get_rootless_child(self, param, params):
        return hddm.likelihoods.LBA(param.full_name,
                                    value=param.data['rt'],
                                    a=params['a'],
                                    z=params['z'],
                                    t=params['t'],
                                    v0=params['v0'],
                                    v1=params['v1'],
                                    V=params['V'],
                                    normalize_v=self.normalize_v,
                                    observed=True)

    def get_root_node(self, param):
        """Create and return a prior distribution for [param]. [tag] is
        used in case of dependent parameters.
        """
        if param.name == 'V' and self.fix_sv is not None: # drift rate variability
            return pm.Lambda(param.full_name, lambda x=self.fix_sv: x)
        else:
            return super(self.__class__, self).get_root_param(self, param)

    def get_child_node(self, param, plot=False):
        if param.name.startswith('V') and self.fix_sv is not None:
            return pm.Lambda(param.full_name, lambda x=param.root: x,
                             plot=plot, trace=self.trace_subjs)
        else:
            return super(self.__class__, self).get_child_node(param, plot=plot)
