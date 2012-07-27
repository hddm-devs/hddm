from __future__ import division
import pymc as pm
import numpy as np
import scipy as sp
from scipy import stats
from scipy.stats.mstats import mquantiles

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

def wiener_like_contaminant(value, cont_x, v, sv, a, z, sz, t, st, t_min, t_max,
                            err, n_st, n_sz, use_adaptive, simps_err):
    """Log-likelihood for the simple DDM including contaminants"""
    return hddm.wfpt.wiener_like_contaminant(value, cont_x.astype(np.int32), v, sv, a, z, sz, t, st,
                                                  t_min, t_max, err, n_st, n_sz, use_adaptive, simps_err)

WienerContaminant = pm.stochastic_from_dist(name="Wiener Simple Diffusion Process",
                                       logp=wiener_like_contaminant,
                                       dtype=np.float,
                                       mv=True)

def general_WienerCont(err=1e-4, n_st=2, n_sz=2, use_adaptive=1, simps_err=1e-3):
    _like = lambda  value, cont_x, v, sv, a, z, sz, t, st, t_min, t_max, err=err, n_st=n_st, n_sz=n_sz, \
    use_adaptive=use_adaptive, simps_err=simps_err: \
    wiener_like_contaminant(value, cont_x, v, sv, a, z, sz, t, st, t_min, t_max,\
                            err=err, n_st=n_st, n_sz=n_sz, use_adaptive=use_adaptive, simps_err=simps_err)
    _like.__doc__ = wiener_like_contaminant.__doc__
    return pm.stochastic_from_dist(name="Wiener Diffusion Contaminant Process",
                                       logp=_like,
                                       dtype=np.float,
                                       mv=False)

class wfpt_gen(stats.distributions.rv_continuous):
    wiener_params = {'err': 1e-4, 'n_st':2, 'n_sz':2,
                                  'use_adaptive':1,
                                  'simps_err':1e-3}
    sampling_method = 'cdf'
    dt=1e-4
    cdf_range = (-2,2)

    def _argcheck(self, *args):
        return True

    def _logp(self, x, v, sv, a, z, sz, t, st):
        """Log-likelihood for the full DDM using the interpolation method"""
        return hddm.wfpt.wiener_like(x, v, sv, a, z, sz, t, st, self.wiener_params['err'], self.wiener_params['n_st'], self.wiener_params['n_sz'], self.wiener_params['use_adaptive'], self.wiener_params['simps_err'])

    def _pdf(self, x, v, sv, a, z, sz, t, st):
        if np.isscalar(x):
            out = hddm.wfpt.full_pdf(x, v, sv, a, z, sz, t, st, self.dt)
        else:
            out = hddm.wfpt.pdf_array(x, v[0], sv[0], a[0], z[0], sz[0], t[0], st[0], self.dt, logp=False)
            #out = np.empty_like(x)
            #for i in xrange(len(x)):
            #    out[i] = hddm.wfpt.full_pdf(x[i], v[i], sv[i], a[i], z[i], Z[i], t[i], st[i], self.dt)

        return out

    def _rvs(self, v, sv, a, z, sz, t, st):
        param_dict = {'v': v, 'z': z, 't': t, 'a': a, 'sz': sz, 'sv': sv, 'st': st}
        sampled_rts = hddm.generate.gen_rts(param_dict, method=self.sampling_method,
                                            samples=self._size, dt=self.dt, range_=self.cdf_range)
        return sampled_rts

    def random(self, v=1., sv=0., a=2, z=.5, sz=.1, t=.3, st=.1, size=100):
        self._size = size
        return self._rvs(v, sv, a, z, sz, t, st)


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



def generate_wfpt_stochastic_class(wiener_params=None, sampling_method='cdf', cdf_range=(-5,5), sampling_dt=1e-4):

    #set wiener_params
    if wiener_params is None:
        wiener_params = {'err': 1e-4, 'n_st':2, 'n_sz':2,
                      'use_adaptive':1,
                      'simps_err':1e-3}
    wp = wiener_params

    #create likelihood function
    def wfpt_like(x, v, sv, a, z, sz, t, st):
        return hddm.wfpt.wiener_like(x, v, sv, a, z, sz, t, st, wp['err'], wp['n_st'],
                                      wp['n_sz'], wp['use_adaptive'], wp['simps_err'])

    #create random function
    def random(v, sv, a, z, sz, t, st, size=1):
        param_dict = {'v': v, 'z': z, 't': t, 'a': a, 'sz': sz, 'sv': sv, 'st': st}
        sampled_rts = hddm.generate.gen_rts(param_dict, method=sampling_method,
                                            samples=size, dt=sampling_dt, range_=cdf_range)

    #set pymc node
    wfpt_pymc = pm.stochastic_from_dist('wfpt', wfpt_like, random=random)


    #turn pymc node into the final wfpt_node
    class wfpt_node(wfpt_pymc):

        def cdf_vec(self):
            return hddm.wfpt.gen_cdf(time=cdf_range[1], **self.parents)

        def set_qunatiles_stats(self, quantiles):
            """
            """
            data = self.value

            #get proportion of data fall between the quantiles
            quantiles = np.array(quantiles)
            pos_proportion = np.diff(np.concatenate((np.array([0.]), quantiles, np.array([1.]))))
            neg_proportion = pos_proportion[::-1]
            proportion = np.concatenate((neg_proportion[::-1],  pos_proportion))
            self._n_samples = len(data)

            # extract empirical RT at the quantiles
            ub_emp_rt = mquantiles(data[data>0], prob=quantiles)
            lb_emp_rt = -mquantiles(-data[data<0], prob=quantiles)
            self._emp_rt = np.concatenate((lb_emp_rt[::-1], np.array([0.]), ub_emp_rt))

            #get frequancy of observed values
            freq_obs = np.zeros(len(proportion))
            freq_obs[:len(quantiles)+1] = sum(data<0) * neg_proportion
            freq_obs[len(quantiles)+1:] = sum(data>0) * pos_proportion
            self._freq_obs = freq_obs

        def chisquare(self):
            """
            """
            # generate CDF
            try:
                x_cdf, cdf = self.cdf_vec()
            except ValueError:
                return np.inf

            # extract theoretical RT indices
            theo_idx = np.searchsorted(x_cdf, self._emp_rt)

            #get probablities associated with theoretical RT indices
            theo_cdf = np.concatenate((np.array([0.]), cdf[theo_idx], np.array([1.])))

            #chisquare
            theo_proportion = np.diff(theo_cdf)
            freq_exp = theo_proportion * self._n_samples
            score,_ = stats.chisquare(self._freq_obs, freq_exp)

            return score

    return wfpt_node

def wiener_like_gpu(value, v, sv, a, z, t, out, err=1e-4):
    """Log-likelihood for the simple DDM including contaminants"""
    # Check if parameters are in allowed range
    if z<0 or z>1 or t<0 or a <= 0 or sv<=0:
        return -np.inf

    wfpt_gpu.pdf_gpu(value, float(v), float(sv), float(a), float(z), float(t), err, out)
    logp = gpuarray.sum(out).get() #cumath.log(out)).get()

    return np.asscalar(logp)

WienerGPU = pm.stochastic_from_dist(name="Wiener Simple Diffusion Process",
                                    logp=wiener_like_gpu,
                                    dtype=np.float32,
                                    mv=False)
