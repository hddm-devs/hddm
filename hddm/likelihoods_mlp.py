import numpy as np
import hddm
from functools import partial
from kabuki.utils import stochastic_from_dist
from hddm.simulators import *
#import data_simulators
from copy import deepcopy

def make_mlp_likelihood_complete(model,
                                 **kwargs):
    """Defines the likelihoods for the MLP networks.

    :Arguments:
        model: str <default='ddm>
            String that determines which model you would like to fit your data to.
            Currently available models are: 'ddm', 'full_ddm', 'angle', 'weibull', 'ornstein', 'levy'
        **kwargs: dict
            Dictionary of additional keyword arguments. 
            Importantly here, this carries the preloaded CNN.
    :Returns:
        pymc.object: Returns a stochastic object as defined by PyMC2
    """

    def random(self):
        """
            Generate random samples from a given model (the dataset matches the size of the respective observated dataset supplied as an attribute of 'self').
        """

        # this can be simplified so that we pass parameters directly to the simulator ...
        theta = np.array(model_config[model]['default_params'], dtype = np.float32)
        keys_tmp = self.parents.value.keys()
        cnt = 0
        
        for param in model_config[model]['params']:
            if param in keys_tmp:
                theta[cnt] = np.array(self.parents.value[param]).astype(np.float32)
            cnt += 1
        
        sim_out = simulator(theta = theta, 
                            model = model, 
                            n_samples = self.shape[0], 
                            max_t = 20)
        return hddm_preprocess(sim_out, keep_negative_responses = True)

    if model == 'ddm':
        def wienernn_like_ddm(x, 
                              v,  
                              a, 
                              z,  
                              t, 
                              p_outlier = 0,
                              w_outlier = 0.1,
                              **kwargs):
            """
                LAN Log-likelihood for the DDM
            """  

            return hddm.wfpt.wiener_like_nn_ddm(x['rt'].values,
                                                x['response'].values,  
                                                v, # sv,
                                                a, 
                                                z, # sz,
                                                t, # st,
                                                p_outlier = p_outlier,
                                                w_outlier = w_outlier,
                                                **kwargs)

        def pdf_ddm(self, x):
            rt = np.array(x, dtype = np.float32)
            response = rt / np.abs(rt)
            rt = np.abs(rt)
 
            # model_config[] # TODO FILL THIS IN SO THAT WE CREATE THE APPROPRIATE ARRAY AS INPUT TO THE SIMULATOR
            out = hddm.wfpt.wiener_like_nn_ddm_pdf(x = rt, response = response, network = kwargs['network'], **self.parents)# **kwargs) # This may still be buggy !
            return out

        def cdf_ddm(self, x):
            # TODO: Implement the CDF method for neural networks
            return 'Not yet implemented'

        # Create wfpt class
        wfpt_nn = stochastic_from_dist('Wienernn_' + model, partial(wienernn_like_ddm, **kwargs))

        wfpt_nn.pdf = pdf_ddm
        wfpt_nn.cdf_vec = None # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
        wfpt_nn.cdf = cdf_ddm
        wfpt_nn.random = random
        return wfpt_nn

    if model == 'weibull_cdf' or model == 'weibull':
        def wienernn_like_weibull(x, 
                                  v,
                                  a, 
                                  alpha,
                                  beta,
                                  z,
                                  t,
                                  p_outlier = 0,
                                  w_outlier = 0,
                                  **kwargs): #theta

            """
                LAN Log-likelihood for the WEIBULL MODEL
            """  

            return hddm.wfpt.wiener_like_nn_weibull(x['rt'].values,
                                                    x['response'].values, 
                                                    v, 
                                                    a, 
                                                    alpha, 
                                                    beta,
                                                    z, 
                                                    t, 
                                                    p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
                                                    w_outlier = w_outlier,
                                                    **kwargs)

        def pdf_weibull(self, x):
            rt = np.array(x, dtype = np.float32)
            response = rt / np.abs(rt)
            rt = np.abs(rt)
            out = hddm.wfpt.wiener_like_nn_weibull_pdf(x = rt, response = response, network = kwargs['network'], **self.parents) # **kwargs) # This may still be buggy !
            return out

        def cdf_weibull(self, x):
            # TODO: Implement the CDF method for neural networks
            return 'Not yet implemented'

        # Create wfpt class
        wfpt_nn = stochastic_from_dist('Wienernn_' + model, partial(wienernn_like_weibull, **kwargs))

        wfpt_nn.pdf = pdf_weibull
        wfpt_nn.cdf_vec = None # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
        wfpt_nn.cdf = cdf_weibull
        wfpt_nn.random = random
        return wfpt_nn
    
    if model == 'ddm_sdv':
        def wienernn_like_ddm_sdv(x, 
                          v,
                          sv,
                          a, 
                          z, 
                          t,
                          p_outlier = 0,
                          w_outlier = 0,
                          **kwargs):
            """
                LAN Log-likelihood for the DDM-SDV MODEL
            """  

            return hddm.wfpt.wiener_like_nn_ddm_sdv(x['rt'].values,
                                                    x['response'].values,  
                                                    v,
                                                    sv,
                                                    a,
                                                    z, 
                                                    t, 
                                                    p_outlier = p_outlier,
                                                    w_outlier = w_outlier,
                                                    **kwargs)

        def pdf_ddm_sdv(self, x):
            rt = np.array(x, dtype = np.float32)
            response = rt / np.abs(rt)
            rt = np.abs(rt)
            out = hddm.wfpt.wiener_like_nn_ddm_sdv_pdf(x = rt, response = response, network = kwargs['network'], **self.parents) # **kwargs) # This may still be buggy !
            return out

        def cdf_ddm_sdv(self, x):
            # TODO: Implement the CDF method for neural networks
            return 'Not yet implemented'

        # Create wfpt class
        wfpt_nn = stochastic_from_dist('Wienernn_' + model, partial(wienernn_like_ddm_sdv, **kwargs))

        wfpt_nn.pdf = pdf_ddm_sdv
        wfpt_nn.cdf_vec = None # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
        wfpt_nn.cdf = cdf_ddm_sdv
        wfpt_nn.random = random
        return wfpt_nn
    
    if model == 'ddm_sdv_analytic':
        def wienernn_like_ddm_sdv_analytic(x, 
                                           v, 
                                           sv,
                                           a, 
                                           z, 
                                           t, 
                                           p_outlier = 0,
                                           w_outlier = 0,
                                           **kwargs):
            """
                LAN Log-likelihood for the DDM-SDV MODEL (Trained on analytic likelihoods).
            """  

            return hddm.wfpt.wiener_like_nn_ddm_sdv_analytic(x['rt'].values,
                                                             x['response'].values,  
                                                             v, 
                                                             sv, 
                                                             a, 
                                                             z, 
                                                             t,
                                                             p_outlier = p_outlier,
                                                             w_outlier = w_outlier,
                                                             **kwargs)

        def pdf_ddm_sdv_analytic(self, x):
            rt = np.array(x, dtype = np.float32)
            response = rt / np.abs(rt)
            rt = np.abs(rt)
            out = hddm.wfpt.wiener_like_nn_ddm_sdv_analytic_pdf(x = rt, response = response, network = kwargs['network'], **self.parents) # **kwargs) # This may still be buggy !
            return out

        def cdf_ddm_sdv_analytic(self, x):
            # TODO: Implement the CDF method for neural networks
            return 'Not yet implemented'

        # Create wfpt class
        wfpt_nn = stochastic_from_dist('Wienernn_' + model, partial(wienernn_like_ddm_sdv_analytic, **kwargs))

        wfpt_nn.pdf = pdf_ddm_sdv_analytic
        wfpt_nn.cdf_vec = None # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
        wfpt_nn.cdf = cdf_ddm_sdv_analytic
        wfpt_nn.random = random
        return wfpt_nn

    if model == 'levy':
        def wienernn_like_levy(x, 
                               v, 
                               a, 
                               alpha,
                               z,
                               t,
                               p_outlier = 0.1,
                               w_outlier = 0.1,
                               **kwargs): #theta
            """
                LAN Log-likelihood for the LEVY MODEL
            """  

            return hddm.wfpt.wiener_like_nn_levy(x['rt'].values,
                                    x['response'].values, 
                                    v,
                                    a, 
                                    alpha, 
                                    z,
                                    t, 
                                    p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
                                    w_outlier = w_outlier,
                                    **kwargs)

        def pdf_levy(self, x):
            rt = np.array(x, dtype = np.float32)
            response = rt / np.abs(rt)
            rt = np.abs(rt)
            out = hddm.wfpt.wiener_like_nn_levy_pdf(x = rt, response = response, network = kwargs['network'], **self.parents) # **kwargs) # This may still be buggy !
            return out

        def cdf_levy(self, x):
            # TODO: Implement the CDF method for neural networks
            return 'Not yet implemented'

        # Create wfpt class
        wfpt_nn = stochastic_from_dist('Wienernn_' + model, partial(wienernn_like_levy, **kwargs))

        wfpt_nn.pdf = pdf_levy
        wfpt_nn.cdf_vec = None # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
        wfpt_nn.cdf = cdf_levy
        wfpt_nn.random = random
        return wfpt_nn

    if model == 'ornstein':
        def wienernn_like_ornstein(x,
                                   v, 
                                   a, 
                                   g,
                                   z, 
                                   t,
                                   p_outlier = 0,
                                   w_outlier = 0,
                                   **kwargs): #theta
            """
                LAN Log-likelihood for the ORNSTEIN MODEL
            """  
    
            return hddm.wfpt.wiener_like_nn_ornstein(x['rt'].values,
                                                     x['response'].values, 
                                                     v, 
                                                     a, 
                                                     g, 
                                                     z, 
                                                     t, 
                                                     p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
                                                     w_outlier = w_outlier,
                                                     **kwargs)

        def pdf_ornstein(self, x):
            rt = np.array(x, dtype = np.float32)
            response = rt / np.abs(rt)
            rt = np.abs(rt)
            out = hddm.wfpt.wiener_like_nn_ornstein_pdf(x = rt, response = response, network = kwargs['network'], **self.parents) # **kwargs) # This may still be buggy !
            return out

        def cdf_ornstein(self, x):
            # TODO: Implement the CDF method for neural networks
            return 'Not yet implemented'

        # Create wfpt class
        wfpt_nn = stochastic_from_dist('Wienernn_' + model, partial(wienernn_like_ornstein, **kwargs))

        wfpt_nn.pdf = pdf_ornstein
        wfpt_nn.cdf_vec = None # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
        wfpt_nn.cdf = cdf_ornstein
        wfpt_nn.random = random
        return wfpt_nn

    if model == 'full_ddm' or model == 'full_ddm2':
        def wienernn_like_full_ddm(x, 
                                   v, 
                                   sv, 
                                   a, 
                                   z, 
                                   sz, 
                                   t, 
                                   st, 
                                   p_outlier = 0,
                                   w_outlier = 0,
                                   **kwargs):
            """
                LAN Log-likelihood for the FULL DDM MODEL
            """  

            return hddm.wfpt.wiener_like_nn_full_ddm(x['rt'].values,
                                                     x['response'].values,
                                                     v,
                                                     sv,
                                                     a,
                                                     z, 
                                                     sz, 
                                                     t,
                                                     st,
                                                     p_outlier = p_outlier,
                                                     w_outlier = w_outlier,
                                                     **kwargs)

        #return wienernn_like_full_ddm
        def pdf_full_ddm(self, x):
            rt = np.array(x, dtype = np.float32)
            response = rt / np.abs(rt)
            rt = np.abs(rt)
            out = hddm.wfpt.wiener_like_nn_full_ddm_pdf(x = rt, response = response, network = kwargs['network'], **self.parents) # **kwargs) # This may still be buggy !
            return out

        def cdf_full_ddm(self, x):
            # TODO: Implement the CDF method for neural networks
            return 'Not yet implemented'

        # Create wfpt class
        wfpt_nn = stochastic_from_dist('Wienernn_' + model, partial(wienernn_like_full_ddm, **kwargs))

        wfpt_nn.pdf = pdf_full_ddm
        wfpt_nn.cdf_vec = None # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
        wfpt_nn.cdf = cdf_full_ddm
        wfpt_nn.random = random
        #return wienernn_like_ddm_sdv_analytic
        #return wienernn_like_ornstein
        return wfpt_nn

    if model == 'angle':
        def wienernn_like_angle(x, 
                                v, 
                                a,
                                theta,
                                z,
                                t,
                                p_outlier = 0,
                                w_outlier = 0,
                                **kwargs):
            """
                LAN Log-likelihood for the ANGLE MODEL
            """  

            return hddm.wfpt.wiener_like_nn_angle(x['rt'].values,
                                                  x['response'].values,  
                                                  v,
                                                  a, 
                                                  theta,
                                                  z,
                                                  t,
                                                  p_outlier = p_outlier,
                                                  w_outlier = w_outlier,
                                                  **kwargs)
            
        def pdf_angle(self, x):
            rt = np.array(x, dtype = np.float32)
            response = rt / np.abs(rt)
            rt = np.abs(rt)
            out = hddm.wfpt.wiener_like_nn_angle_pdf(x = rt, response = response, network = kwargs['network'], **self.parents) # **kwargs) # This may still be buggy !
            return out

        def cdf_angle(self, x):
            # TODO: Implement the CDF method for neural networks
            return 'Not yet implemented'

        # Create wfpt class
        wfpt_nn = stochastic_from_dist('Wienernn_' + model, partial(wienernn_like_angle, **kwargs))

        wfpt_nn.pdf = pdf_angle
        wfpt_nn.cdf_vec = None # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
        wfpt_nn.cdf = cdf_angle
        wfpt_nn.random = random
        return wfpt_nn
    else:
        return 'Not implemented errror: Failed to load likelihood because the model specified is not implemented'

# REGRESSOR LIKELIHOODS
def generate_wfpt_nn_ddm_reg_stochastic_class(model = None,
                                              **kwargs):
    """Defines the regressor likelihoods for the MLP networks.

    :Arguments:
        model: str <default='ddm>
            String that determines which model you would like to fit your data to.
            Currently available models are: 'ddm', 'full_ddm', 'angle', 'weibull', 'ornstein', 'levy'
        **kwargs: dict
            Dictionary of additional keyword arguments. 
            Importantly here, this carries the preloaded CNN.
    :Returns:
        pymc.object: Returns a stochastic object as defined by PyMC2
    """

    # Need to rewrite these random parts !
    def random(self, keep_negative_responses = True, add_model_parameters = False, keep_subj_idx = False):
        """
            Function to sample from a regressor based likelihood. Conditions on the covariates.
        """
        param_dict = deepcopy(self.parents.value)
        del param_dict['reg_outcomes']

        # size = sampled_rts.shape[0]
        n_params = model_config[model]['n_params']
        param_data = np.zeros((self.value.shape[0], n_params), dtype = np.float32)

        cnt = 0
        for tmp_str in model_config[model]['params']: #['v', 'a', 'z', 't']:
            if tmp_str in self.parents['reg_outcomes']:
                param_data[:, cnt] = param_dict[tmp_str].iloc[self.value.index, 0]
            else:
                param_data[:, cnt] = param_dict[tmp_str]
            cnt += 1

        sim_out = simulator(theta = param_data, # n_trials = size,
                            model = model,
                            n_samples = 1,
                            max_t = 20)

        return hddm_preprocess(sim_out, 
                               keep_negative_responses = keep_negative_responses, 
                               add_model_parameters = add_model_parameters, 
                               keep_subj_idx = keep_subj_idx)

    if model == 'ddm':
        def wiener_multi_like_nn_ddm(value, v, a, z, t, 
                                     reg_outcomes, 
                                     p_outlier = 0, 
                                     w_outlier = 0.1,
                                     **kwargs):

            """LAN Log-likelihood for the DDM"""

            params = {'v': v, 'a': a, 'z': z, 't': t}
            n_params = 4 #model_config[model]['n_params']
            size = int(value.shape[0])
            data = np.zeros((size, 6), dtype = np.float32)
            data[:, n_params:] = np.stack([ np.absolute(value['rt']).astype(np.float32), value['response'].astype(np.float32) ], axis = 1)

            cnt = 0
            for tmp_str in ['v', 'a', 'z', 't']: # model_config[model]['params']:
                if tmp_str in reg_outcomes:
                    data[:, cnt] = params[tmp_str].loc[value['rt'].index].values[:, 0]
                    if (data[:, cnt].min() < model_config[model]['param_bounds'][0][cnt]) or (data[:, cnt].max() > model_config[model]['param_bounds'][1][cnt]):
                        print('boundary violation of regressor part')
                        return - np.inf
                else:
                    data[:, cnt] = params[tmp_str]
                cnt += 1

            # Has optimization potential --> AF-TODO: For next version!
            return hddm.wfpt.wiener_like_multi_nn_ddm(data,
                                                      p_outlier = p_outlier,
                                                      w_outlier = w_outlier,
                                                      **kwargs)

        # ADD IN THE PDF PART HERE !
        # AF-COMMENT: This is left out for now because it is unclear what the pdf should accept as input 
        # as well as produce as output. 
        # 

        # def pdf_ddm(self, x):
        #     print(self.parents)
        #     print(x.shape)
        #     #return 'Tried to print pdf'
        #     params = self.parents
        #     n_params = 4
        #     size = int(x.shape[0])
        #     if size != self.shape[0]:
        #         return 'Error: The supplied data does not match '
        #     data = np.zeros((size, 6), dtype = np.float32)
        #     data[: n_params:] = np.stack([ np.absolute(x['rt']).astype(np.float32), x['response'].astype(np.float32) ], axis = 1)

        #     cnt = 0
        #     for tmp_str in ['v', 'a', 'z', 't']:
        #         data[:, cnt] = params[tmp_str].loc[value['rt']]

            # rt = np.array(x, dtype = np.float32)
            # response = rt / np.abs(rt)
            # rt = np.abs(rt)

            # params = self.params
            # n_params = 4
            # size = 
        
        def pdf_ddm(self, x):
            return 'Not yet implemented'

        def cdf_ddm(self, x):
            # TODO: Implement the CDF method for neural networks
            return 'Not yet implemented'

        stoch = stochastic_from_dist('wfpt_reg', partial(wiener_multi_like_nn_ddm, **kwargs))
        stoch.pdf = pdf_ddm
        stoch.cdf = cdf_ddm
        stoch.random = random

    if model == 'full_ddm' or model == 'full_ddm2':
        def wiener_multi_like_nn_full_ddm(value, v, sv, a, z, sz, t, st, 
                                         reg_outcomes, 
                                         p_outlier = 0, 
                                         w_outlier = 0.1,
                                         **kwargs):
            """
                LAN Log-likelihood for the FULL DDM
            """                             

            params = {'v': v, 'a': a, 'z': z, 't': t, 'sz': sz, 'sv': sv, 'st': st}

            n_params = int(7)
            size = int(value.shape[0])
            data = np.zeros((size, 9), dtype = np.float32)
            data[:, n_params:] = np.stack([ np.absolute(value['rt']).astype(np.float32), value['response'].astype(np.float32) ], axis = 1)

            cnt = 0
            for tmp_str in ['v', 'a', 'z', 't', 'sz', 'sv', 'st']:

                if tmp_str in reg_outcomes:
                    data[:, cnt] = params[tmp_str].loc[value['rt'].index].values[:, 0]
                    if (data[:, cnt].min() < model_config[model]['param_bounds'][0][cnt]) or (data[:, cnt].max() > model_config[model]['param_bounds'][1][cnt]):
                        print('boundary violation of regressor part')
                        return - np.inf
                else:
                    data[:, cnt] = params[tmp_str]

                cnt += 1

            # Has optimization potential --> AF-TODO: For next version!
            return hddm.wfpt.wiener_like_multi_nn_full_ddm(data,
                                                        p_outlier = p_outlier,
                                                        w_outlier = w_outlier,
                                                        **kwargs)

        stoch = stochastic_from_dist('wfpt_reg', partial(wiener_multi_like_nn_full_ddm, **kwargs))
        stoch.random = random

    if model == 'angle':
        def wiener_multi_like_nn_angle(value, v, a, theta, z, t, 
                                       reg_outcomes, 
                                       p_outlier = 0, 
                                       w_outlier = 0.1,
                                       **kwargs):

            """
                LAN Log-likelihood for the ANGLE MODEL
            """  

            params = {'v': v, 'a': a, 'z': z, 't': t, 'theta': theta}

            n_params = int(5)
            size = int(value.shape[0])
            data = np.zeros((size, 7), dtype = np.float32)
            data[:, n_params:] = np.stack([ np.absolute(value['rt']).astype(np.float32), value['response'].astype(np.float32) ], axis = 1)

            cnt = 0
            for tmp_str in ['v', 'a', 'z', 't', 'theta']:

                if tmp_str in reg_outcomes:
                    #print('printing values')
                    #print(value)
                    #data[:, cnt] = params[tmp_str].loc[value['rt'].index].values[:, 0]
                    data[:, cnt] = params[tmp_str].loc[value['rt'].index].values
                    if (data[:, cnt].min() < model_config[model]['param_bounds'][0][cnt]) or (data[:, cnt].max() > model_config[model]['param_bounds'][1][cnt]):
                        print('boundary violation of regressor part')
                        return - np.inf
                else:
                    data[:, cnt] = params[tmp_str]

                cnt += 1

            # Has optimization potential --> AF-TODO: For next version!
            return hddm.wfpt.wiener_like_multi_nn_angle(data,
                                                        p_outlier = p_outlier,
                                                        w_outlier = w_outlier,
                                                        **kwargs)

        stoch = stochastic_from_dist('wfpt_reg', partial(wiener_multi_like_nn_angle, **kwargs))
        stoch.random = random

    if model == 'levy':
        def wiener_multi_like_nn_levy(value, v, a, alpha, z, t, 
                                        reg_outcomes, 
                                        p_outlier = 0, 
                                        w_outlier = 0.1,
                                        **kwargs):

            """
                LAN Log-likelihood for the LEVY MODEL
            """  

            params = {'v': v, 'a': a, 'z': z, 'alpha': alpha, 't': t}
            n_params = int(5)
            size = int(value.shape[0])
            data = np.zeros((size, 7), dtype = np.float32)
            data[:, n_params:] = np.stack([ np.absolute(value['rt']).astype(np.float32), value['response'].astype(np.float32) ], axis = 1)

            cnt = 0
            for tmp_str in ['v', 'a', 'z', 'alpha', 't']:

                if tmp_str in reg_outcomes:
                    data[:, cnt] = params[tmp_str].loc[value['rt'].index].values[:, 0]
                    if (data[:, cnt].min() < model_config[model]['param_bounds'][0][cnt]) or (data[:, cnt].max() > model_config[model]['param_bounds'][1][cnt]):
                        print('boundary violation of regressor part')
                        return - np.inf
                else:
                    data[:, cnt] = params[tmp_str]

                cnt += 1

            # Has optimization potential --> AF-TODO: For next version!
            return hddm.wfpt.wiener_like_multi_nn_levy(data,
                                                    p_outlier = p_outlier,
                                                    w_outlier = w_outlier,
                                                    **kwargs)

        stoch = stochastic_from_dist('wfpt_reg', partial(wiener_multi_like_nn_levy, **kwargs))
        stoch.random = random
    
    if model == 'ornstein':
        def wiener_multi_like_nn_ornstein(value, v, a, g, z, t, 
                                      reg_outcomes, 
                                      p_outlier = 0, 
                                      w_outlier = 0.1,
                                      **kwargs):

            """
                LAN Log-likelihood for the ORNSTEIN MODEL
            """  

            params = {'v': v, 'a': a, 'z': z, 'g': g, 't': t}
            
            n_params = int(5)
            size = int(value.shape[0])
            data = np.zeros((size, 7), dtype = np.float32)
            data[:, n_params:] = np.stack([ np.absolute(value['rt']).astype(np.float32), value['response'].astype(np.float32) ], axis = 1)

            cnt = 0
            for tmp_str in ['v', 'a', 'z', 'g', 't']:

                if tmp_str in reg_outcomes:
                    data[:, cnt] = params[tmp_str].loc[value['rt'].index].values[:, 0]
                    if (data[:, cnt].min() < model_config[model]['param_bounds'][0][cnt]) or (data[:, cnt].max() > model_config[model]['param_bounds'][1][cnt]):
                        print('boundary violation of regressor part')
                        return - np.inf
                else:
                    data[:, cnt] = params[tmp_str]

                cnt += 1

            # Has optimization potential --> AF-TODO: For next version!
            return hddm.wfpt.wiener_like_multi_nn_ornstein(data,
                                                        p_outlier = p_outlier,
                                                        w_outlier = w_outlier,
                                                        **kwargs)

        stoch = stochastic_from_dist('wfpt_reg', partial(wiener_multi_like_nn_ornstein, **kwargs))
        stoch.random = random

    if model == 'weibull_cdf' or model == 'weibull':
        def wiener_multi_like_nn_weibull(value, v, a, alpha, beta, z, t, 
                                         reg_outcomes, 
                                         p_outlier = 0, 
                                         w_outlier = 0.1,
                                         **kwargs):

            """
                LAN Log-likelihood for the WEIBULL MODEL
            """  

            params = {'v': v, 'a': a, 'z': z, 't': t, 'alpha': alpha, 'beta': beta}
            n_params = int(6)
            size = int(value.shape[0])
            data = np.zeros((size, 8), dtype = np.float32)
            data[:, n_params:] = np.stack([ np.absolute(value['rt']).astype(np.float32), value['response'].astype(np.float32) ], axis = 1)

            cnt = 0
            for tmp_str in ['v', 'a', 'z', 't', 'alpha', 'beta']:

                if tmp_str in reg_outcomes:
                    data[:, cnt] = params[tmp_str].loc[value['rt'].index].values[:, 0]
                    if (data[:, cnt].min() < model_config[model]['param_bounds'][0][cnt]) or (data[:, cnt].max() > model_config[model]['param_bounds'][1][cnt]):
                        print('boundary violation of regressor part')
                        return - np.inf
                else:
                    data[:, cnt] = params[tmp_str]

                cnt += 1

            # Has optimization potential --> AF-TODO: For next version!
            return hddm.wfpt.wiener_like_multi_nn_weibull(data,
                                                        p_outlier = p_outlier,
                                                        w_outlier = w_outlier,
                                                        **kwargs)

        stoch = stochastic_from_dist('wfpt_reg', partial(wiener_multi_like_nn_weibull, **kwargs))
        stoch.random = random
    return stoch

