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

def generate_wfpt_stochastic_class(wiener_params=None, sampling_method='cdf', cdf_range=(-5,5), sampling_dt=1e-4):
    """
    create a wfpt stochastic class by creating a pymc nodes and then adding quantile functions.
    Input:
        wiener_params <dict> - dictonary of wiener_params for wfpt likelihoods
        sampling_method <string> - an argument used by hddm.generate.gen_rts
        cdf_range <sequance> -  an argument used by hddm.generate.gen_rts
        sampling_dt <float> - an argument used by hddm.generate.gen_rts
    Ouput:
        wfpt <class> - the wfpt stochastic
    """

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
    def random(v, sv, a, z, sz, t, st, size=None):
        param_dict = {'v': v, 'z': z, 't': t, 'a': a, 'sz': sz, 'sv': sv, 'st': st}
        return hddm.generate.gen_rts(param_dict, method=sampling_method,
                                    samples=size, dt=sampling_dt, range_=cdf_range)


    #create pdf function
    def pdf(self, x):
        kwargs = wp.copy()
        kwargs.update(self.parents)

        if np.isscalar(x):
            out = hddm.wfpt.full_pdf(x, **kwargs)
        else:
            out = hddm.wfpt.pdf_array(x, logp=False, **kwargs)
        return out

    #create wfpt class
    wfpt = pm.stochastic_from_dist('wfpt', wfpt_like, random=random)

    #add pdf and cdf_vec to the class
    wfpt.pdf = pdf
    wfpt.cdf_vec = lambda self: hddm.wfpt.gen_cdf(time=cdf_range[1], precision=3., **self.parents)

    #add quantiles functions
    add_quantiles_functions_to_pymc_class(wfpt)

    return wfpt

def add_quantiles_functions_to_pymc_class(pymc_class):
    """
    add quantiles methods to a pymc class
    Input:
        pymc_class <class>
    """

    #turn pymc node into the final wfpt_node
    def compute_quantiles_stats(self, quantiles):
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

    def set_quantiles_stats(self, n_samples, emp_rt, freq_obs):
        self._n_samples = n_samples
        self._emp_rt = emp_rt
        self._freq_obs = freq_obs

    def get_quantiles_stats(self):
        stats = {'n_samples': self._n_samples, 'emp_rt': self._emp_rt, 'freq_obs': self._freq_obs}
        return stats

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

    pymc_class.compute_quantiles_stats = compute_quantiles_stats
    pymc_class.set_quantiles_stats = set_quantiles_stats
    pymc_class.get_quantiles_stats = get_quantiles_stats
    pymc_class.chisquare = chisquare


#create default Wfpt class
Wfpt = generate_wfpt_stochastic_class()

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

