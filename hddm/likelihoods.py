from __future__ import division
import pymc as pm
import numpy as np
import scipy as sp
from scipy import stats

from kabuki.distributions import scipy_stochastic

np.seterr(divide='ignore')

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    import pycuda.cumath as cumath
    import wfpt_gpu
    gpu_imported = True
except:
    gpu_imported = False

import hddm

def wiener_like_contaminant(value, cont_x, v, V, a, z, Z, t, T, t_min, t_max,
                            err, nT, nZ, use_adaptive, simps_err):
    """Log-likelihood for the simple DDM including contaminants"""
    return hddm.wfpt.wiener_like_contaminant(value, cont_x.astype(np.int32), v, V, a, z, Z, t, T,
                                                  t_min, t_max, err, nT, nZ, use_adaptive, simps_err)

WienerContaminant = pm.stochastic_from_dist(name="Wiener Simple Diffusion Process",
                                       logp=wiener_like_contaminant,
                                       dtype=np.float,
                                       mv=True)

def general_WienerCont(err=1e-4, nT=2, nZ=2, use_adaptive=1, simps_err=1e-3):
    _like = lambda  value, cont_x, v, V, a, z, Z, t, T, t_min, t_max, err=err, nT=nT, nZ=nZ, \
    use_adaptive=use_adaptive, simps_err=simps_err: \
    wiener_like_contaminant(value, cont_x, v, V, a, z, Z, t, T, t_min, t_max,\
                            err=err, nT=nT, nZ=nZ, use_adaptive=use_adaptive, simps_err=simps_err)
    _like.__doc__ = wiener_like_contaminant.__doc__
    return pm.stochastic_from_dist(name="Wiener Diffusion Contaminant Process",
                                       logp=_like,
                                       dtype=np.float,
                                       mv=False)

class wfpt_gen(stats.distributions.rv_continuous):
    wiener_params = {'err': 1e-4, 'nT':2, 'nZ':2,
                                  'use_adaptive':1,
                                  'simps_err':1e-3}
    sampling_method = 'cdf'
    dt=1e-4

    def _argcheck(self, *args):
        return True

    def _logp(self, x, v, V, a, z, Z, t, T):
        """Log-likelihood for the full DDM using the interpolation method"""
        return hddm.wfpt.wiener_like(x, v, V, a, z, Z, t, T, self.wiener_params['err'], self.wiener_params['nT'], self.wiener_params['nZ'], self.wiener_params['use_adaptive'], self.wiener_params['simps_err'])

    def _pdf(self, x, v, V, a, z, Z, t, T):
        if np.isscalar(x):
            out = hddm.wfpt.full_pdf(x, v, V, a, z, Z, t, T, self.dt)
        else:
            out = hddm.wfpt.pdf_array(x, v[0], V[0], a[0], z[0], Z[0], t[0], T[0], self.dt, logp=False)
            #out = np.empty_like(x)
            #for i in xrange(len(x)):
            #    out[i] = hddm.wfpt.full_pdf(x[i], v[i], V[i], a[i], z[i], Z[i], t[i], T[i], self.dt)

        return out

    def _rvs(self, v, V, a, z, Z, t, T):
        param_dict = {'v':v, 'z':z, 't':t, 'a':a, 'Z':Z, 'V':V, 'T':T}
        sampled_rts = hddm.generate.gen_rts(param_dict, method=self.sampling_method, samples=self._size, dt=self.dt)
        return sampled_rts

    def random(self, v=1., V=0., a=2, z=.5, Z=.1, t=.3, T=.1, size=100):
        self._size = size
        return self._rvs(v, V, a, z, Z, t, T)

wfpt_like = scipy_stochastic(wfpt_gen, name='wfpt', longname="""Wiener first passage time likelihood function""", extradoc="""Wiener first passage time (WFPT) likelihood function of the Ratcliff Drift Diffusion Model (DDM). Models two choice decision making tasks as a drift process that accumulates evidence across time until it hits one of two boundaries and executes the corresponding response. Implemented using the Navarro & Fuss (2009) method.

Parameters:
***********
v: drift-rate
a: threshold
z: bias [0,1]
t: non-decision time

References:
***********
Fast and accurate calculations for first-passage times in Wiener diffusion models
Navarro & Fuss - Journal of Mathematical Psychology, 2009 - Elsevier
""")

def wiener_like_gpu(value, v, V, a, z, t, out, err=1e-4):
    """Log-likelihood for the simple DDM including contaminants"""
    # Check if parameters are in allowed range
    if z<0 or z>1 or t<0 or a <= 0 or V<=0:
        return -np.inf

    wfpt_gpu.pdf_gpu(value, float(v), float(V), float(a), float(z), float(t), err, out)
    logp = gpuarray.sum(out).get() #cumath.log(out)).get()

    return np.asscalar(logp)

WienerGPU = pm.stochastic_from_dist(name="Wiener Simple Diffusion Process",
                                    logp=wiener_like_gpu,
                                    dtype=np.float32,
                                    mv=False)
