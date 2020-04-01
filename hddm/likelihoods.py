
import pymc as pm
import numpy as np
from scipy import stats

from kabuki.utils import stochastic_from_dist

np.seterr(divide='ignore')

import hddm

def wiener_like_contaminant(value, cont_x, v, sv, a, z, sz, t, st, t_min, t_max,
                            err, n_st, n_sz, use_adaptive, simps_err):
    """Log-likelihood for the simple DDM including contaminants"""
    return hddm.wfpt.wiener_like_contaminant(value, cont_x.astype(np.int32), v, sv, a, z, sz, t, st,
                                             t_min, t_max, err, n_st, n_sz, use_adaptive, simps_err)

WienerContaminant = stochastic_from_dist(name="Wiener Simple Diffusion Process",
                                         logp=wiener_like_contaminant)

def general_WienerCont(err=1e-4, n_st=2, n_sz=2, use_adaptive=1, simps_err=1e-3):
    _like = lambda  value, cont_x, v, sv, a, z, sz, t, st, t_min, t_max, err=err, n_st=n_st, n_sz=n_sz, \
    use_adaptive=use_adaptive, simps_err=simps_err: \
    wiener_like_contaminant(value, cont_x, v, sv, a, z, sz, t, st, t_min, t_max,\
                            err=err, n_st=n_st, n_sz=n_sz, use_adaptive=use_adaptive, simps_err=simps_err)
    _like.__doc__ = wiener_like_contaminant.__doc__
    return stochastic_from_dist(name="Wiener Diffusion Contaminant Process",
                                logp=_like)

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
                         'simps_err':1e-3,
                         'w_outlier': 0.1,
                         }
    wp = wiener_params

    #create likelihood function
    def wfpt_like(x, v, sv, a, z, sz, t, st, p_outlier=0):
        if x['rt'].abs().max() < 998:
            return hddm.wfpt.wiener_like(x['rt'].values, v, sv, a, z, sz, t, st,
                                         p_outlier=p_outlier, **wp)
        else:  # for missing RTs. Currently undocumented.
            noresponse = x['rt'].abs() >= 999
            ## get sum of log p for trials with RTs as usual ##
            logp_resp = hddm.wfpt.wiener_like(x.loc[~noresponse, 'rt'].values,
                                             v, sv, a, z, sz, t, st, p_outlier=p_outlier, **wp)

            # get number of no-response trials
            n_noresponse = sum(noresponse)
            k_upper = sum(x.loc[noresponse, 'rt'] > 0)

            # percentage correct according to probability to get to upper boundary
            if v == 0:
                p_upper = z
            else:
                p_upper = (np.exp(-2 * a * z * v) - 1) / (np.exp(-2 * a * v) - 1)

            logp_noresp = stats.binom.logpmf(k_upper, n_noresponse, p_upper)
            return logp_resp + logp_noresp

    #create random function
    def random(self):
        return hddm.utils.flip_errors(hddm.generate.gen_rts(method=sampling_method,
                                                            size=self.shape, dt=sampling_dt,
                                                            range_=cdf_range,
                                                            structured=True,
                                                            **self.parents.value))


    #create pdf function
    def pdf(self, x):
        out = hddm.wfpt.pdf_array(x, **self.parents)
        return out

    #create cdf function
    def cdf(self, x):
        return hddm.cdfdif.dmat_cdf_array(x, w_outlier=wp['w_outlier'], **self.parents)

    #create wfpt class
    wfpt = stochastic_from_dist('wfpt', wfpt_like)

    #add pdf and cdf_vec to the class
    wfpt.pdf = pdf
    wfpt.cdf_vec = lambda self: hddm.wfpt.gen_cdf_using_pdf(time=cdf_range[1], **dict(list(self.parents.items()) + list(wp.items())))
    wfpt.cdf = cdf
    wfpt.random = random

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
    def compute_quantiles_stats(self, quantiles=(0.1, 0.3, 0.5, 0.7, 0.9)):
        """
        compute quantiles statistics
        Input:
            quantiles : sequence
                the sequence of quantiles,  e.g. (0.1, 0.3, 0.5, 0.7, 0.9)
        """
        try:
            if all(self._quantiles_edges == np.asarray(quantiles)):
                return
        except AttributeError:
            pass

        if hasattr(self, '_is_average_node'):
            raise AttributeError('cannot recompute stats of average model')

        self._quantiles_edges = np.asarray(quantiles)

        data = self.value
        
        if np.all(~np.isnan(data['rt'])):
            
            #get proportion of data fall between the quantiles
            quantiles = np.array(quantiles)
            pos_proportion = np.diff(np.concatenate((np.array([0.]), quantiles, np.array([1.]))))
            neg_proportion = pos_proportion[::-1]
            proportion = np.concatenate((neg_proportion[::-1],  pos_proportion))
            self._n_samples = len(data)

            # extract empirical RT at the quantiles
            self._empirical_quantiles = hddm.utils.data_quantiles(data, quantiles)
            ub_emp_rt = self._empirical_quantiles[1]
            lb_emp_rt = -self._empirical_quantiles[0]
            self._emp_rt = np.concatenate((lb_emp_rt[::-1], np.array([0.]), ub_emp_rt))

            #get frequency of observed values
            freq_obs = np.zeros(len(proportion))
            freq_obs[:len(quantiles)+1] = sum(data.rt<0) * neg_proportion
            freq_obs[len(quantiles)+1:] = sum(data.rt>0) * pos_proportion
            self._freq_obs = freq_obs
        
        else:
            
            #get proportion of data fall between the quantiles
            quantiles = np.array(quantiles)
            pos_proportion = np.diff(np.concatenate((np.array([0.]), quantiles, np.array([1.]))))
            neg_proportion = np.array([1])
            proportion = np.concatenate((neg_proportion[::-1],  pos_proportion))
            self._n_samples = len(data)
            
            # extract empirical RT at the quantiles
            self._empirical_quantiles = hddm.utils.data_quantiles(data, quantiles)
            ub_emp_rt = self._empirical_quantiles[1]
            lb_emp_rt = -self._empirical_quantiles[0]
            self._emp_rt = np.concatenate((np.array([0.]), ub_emp_rt))
            
            #get frequency of observed values
            freq_obs = np.zeros(len(proportion))
            freq_obs[0] = sum(np.isnan(data.rt)) * neg_proportion
            freq_obs[1:] = sum(data.rt>0) * pos_proportion
            self._freq_obs = freq_obs
        
    def set_quantiles_stats(self, quantiles, n_samples, emp_rt, freq_obs, p_upper):
        """
        set quantiles statistics (used when one do not to compute the statistics from the stochastic's value)
        """
        self._quantiles_edges = np.asarray(quantiles)
        self._n_samples = n_samples
        self._emp_rt = emp_rt
        self._freq_obs = freq_obs

        nq = len(quantiles)
        q_lower = -emp_rt[:nq][::-1]
        q_upper = emp_rt[nq+1:]
        self._empirical_quantiles = (q_lower, q_upper,p_upper)

    def get_quantiles_stats(self, quantiles=(0.1, 0.3, 0.5, 0.7, 0.9)):
        """
        get quantiles statistics (after they were computed using compute_quantiles_stats)
        """
        self.compute_quantiles_stats(quantiles)

        stats = {'n_samples': self._n_samples, 'emp_rt': self._emp_rt, 'freq_obs': self._freq_obs}
        return stats

    def _get_theoretical_proportion(self):

        #get cdf
        cdf = self.cdf(self._emp_rt)

        #get probabilities associated with theoretical RT indices
        theo_cdf = np.concatenate((np.array([0.]), cdf, np.array([1.])))

        #theoretical porportion
        proportion = np.diff(theo_cdf)

        #make sure there is no zeros since it causes bugs later on
        epsi = 1e-6
        proportion[proportion <= epsi] = epsi
        return proportion

    def chisquare(self):
        """
        compute the chi-square statistic over the stocastic's value
        """
        try:
            theo_proportion = self._get_theoretical_proportion()
        except (ValueError, FloatingPointError):
            return np.inf
        freq_exp = theo_proportion * self._n_samples
        score,_ = stats.chisquare(self._freq_obs, freq_exp)

        return score

    def gsquare(self):
        """
        compute G^2 (likelihood chi-square) statistic over the stocastic's value
        Note:
         this does return the actual G^2, but G^2 up to a constant which depend on the data
        """
        try:
            theo_proportion = self._get_theoretical_proportion()
        except ValueError:
            return -np.inf
        return 2 * sum(self._freq_obs * np.log(theo_proportion))

    def empirical_quantiles(self, quantiles=(.1, .3, .5, .7, .9)):
        """
        return the quantiles of the Stochastic's value
        Output:
            q_lower - lower boundary quantiles
            q_upper - upper_boundary_quantiles
            p_upper - probability of hitting the upper boundary
        """
        self.compute_quantiles_stats(quantiles)

        return self._empirical_quantiles

    def theoretical_quantiles(self, quantiles=(.1, .3, .5, .7, .9)):
        """
        return the theoretical quantiles based on Stochastic's parents
        Output:
            q_lower - lower boundary quantiles
            q_upper - upper_boundary_quantiles
            p_upper - probability of hitting the upper boundary
        """

        quantiles = np.asarray(quantiles)
        # generate CDF
        x_lower, cdf_lower, x_upper, cdf_upper = hddm.wfpt.split_cdf(*self.cdf_vec())

        # extract theoretical RT indices
        lower_idx = np.searchsorted(cdf_lower, quantiles*cdf_lower[-1])
        upper_idx = np.searchsorted(cdf_upper, quantiles*cdf_upper[-1])

        q_lower = x_lower[lower_idx]
        q_upper = x_upper[upper_idx]
        p_upper = cdf_upper[-1]

        return (q_lower, q_upper,p_upper)

    pymc_class.compute_quantiles_stats = compute_quantiles_stats
    pymc_class.set_quantiles_stats = set_quantiles_stats
    pymc_class.get_quantiles_stats = get_quantiles_stats
    pymc_class.chisquare = chisquare
    pymc_class.gsquare = gsquare
    pymc_class._get_theoretical_proportion = _get_theoretical_proportion
    pymc_class.empirical_quantiles = empirical_quantiles
    pymc_class.theoretical_quantiles = theoretical_quantiles

#create default Wfpt class
Wfpt = generate_wfpt_stochastic_class()
