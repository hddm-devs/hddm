import hddm
from hddm.model import Base
import pymc as pm
from kabuki import Parameter

class HDDMGPU(Base):
    """Experimental."""
    def get_rootless_child(self, param, params):
        import pycuda.driver as cuda
        import pycuda.autoinit
        import pycuda.gpuarray as gpuarray

        if param.name == 'wfpt':
            data = param.data['rt'].flatten().astype(np.float32)
            data_gpu = gpuarray.to_gpu(data)
            out_gpu = gpuarray.empty_like(data_gpu)
            return hddm.likelihoods.WienerGPU(param.full_name,
                                              value=data_gpu,
                                              v = params['v'],
                                              V = self.get_node('V', params),
                                              a = params['a'],
                                              z = self.get_node('z',params),
                                              t = params['t'],
                                              out = out_gpu,
                                              observed=True)

        else:
            raise KeyError, "Rootless parameter named %s not found." % param.name
