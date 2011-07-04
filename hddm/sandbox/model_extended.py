import hddm
from hddm.model import Base
import pymc as pm
from kabuki import Parameter

class HDDMFullExtended(Base):
    def get_params(self):
        [self.include.add(param) for param in ['V','Z','T']]
        
        params = [Parameter('z_trls', False),
                  Parameter('v_trls', False),
                  Parameter('t_trls', False)]
        params += list(super(self.__class__, self).get_params())

        return params

    def get_rootless_child(self, param, params):
        trials = len(param.data)

        if param.name == 'z_trls':
            return [hddm.likelihoods.CenterUniform(param.full_name+str(i),
                                                   center=params['z'],
                                                   width=params['Z']) for i in range(trials)]

        elif param.name == 'v_trls':
            return [pm.Normal(param.full_name+str(i),
                              mu=params['v'],
                              tau=params['V']**-2) for i in range(trials)]

        elif param.name == 't_trls':
            return [hddm.likelihoods.CenterUniform(param.full_name+str(i),
                                                   center=params['t'],
                                                   width=params['T']) for i in range(trials)]

        elif param.name == 'wfpt':
            return hddm.likelihoods.WienerSingleTrial(param.full_name,
                                                      value=param.data['rt'],
                                                      v=params['v_trls'],
                                                      t=params['t_trls'], 
                                                      a=[params['a'] for i in range(trials)],
                                                      z=params['z_trls'],
                                                      observed=True)
        else:
            raise KeyError, "Rootless child node named %s not found." % param.name
