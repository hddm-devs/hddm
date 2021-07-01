import numpy as np
import hddm
from functools import partial
from kabuki.utils import stochastic_from_dist
from hddm.simulators import *
#import data_simulators
from copy import deepcopy

# Defining the likelihood functions
def make_cnn_likelihood(model, pdf_multiplier = 1,  **kwargs):
    """Defines the likelihoods for the CNN networks.

    :Arguments:
        model: str <default='ddm>
            String that determines which model you would like to fit your data to.
            Currently available models are: 'ddm', 'full_ddm', 'angle', 'weibull', 'ornstein', 'levy'
        pdf_multiplier: int <default=1>
            Currently not used. Is meant to adjust for the bin size on which CNN RT histograms were based, to get 
            the right proportionality constant. 
        **kwargs: dict
            Dictionary of additional keyword arguments. 
            Importantly here, this carries the preloaded CNN.
    :Returns:
        pymc.object: Returns a stochastic object as defined by PyMC2
    """

    def random(self):
        # print(self.parents)
        # print('printing the dir of self.parents directly')
        # print(dir(self.parents))
        # print('printing dir of the v variable')
        # print(dir(self.parents['v']))
        # print(self.parents['v'].value)
        # print(self.parents.value)
        # print('trying to print the value part of parents')
        # print(dict(self.parents.value))
        # print('tying to pring the values part of parents')
        # print(self.parents.values)

        # this can be simplified so that we pass parameters directly to the simulator ...
        theta = np.array(model_config[model]['default_params'], dtype = np.float32)
        keys_tmp = self.parents.value.keys()
        cnt = 0
        
        for param in model_config[model]['params']:
            if param in keys_tmp:
                theta[cnt] = np.array(self.parents.value[param]).astype(np.float32)
            cnt += 1
        
        #print('print theta from random function in wfpt_nn')
        #print(theta)

        #new_func = partial(simulator, model = model, n_samples = self.shape, max_t = 20) # This may still be buggy !
        #print('self shape: ')
        #print(self.shape)
        sim_out = simulator(theta = theta, model = model, n_samples = self.shape[0], max_t = 20.0)
        return hddm_preprocess(sim_out)

    def pdf(self, x):
        #print('type of x')
        #print(type(x))
        #print(x)
        #print(self.parents)
        #print(**self.parents)
        #print(self.parents['a'])
        #print(dir(self.parents['a']))
        #print(self.parents['a'].value)
        #print(kwargs)
        #print(self.parents['a'].value)
        # Note as per kabuki it seems that x tends to come in as a 'value_range', which is essetially a 1d ndarray
        # We could change this ...

        #rt = np.array()
        #print('rt')
        rt = np.array(x, dtype = np.int_)
        #print(rt)
        response = rt.copy()
        response[rt < 0] = 0
        response[rt > 0] = 1
        response = response.astype(np.int_)
        rt = np.abs(rt)
        #print(rt)
        #print(response)
        #response = rt / np.abs(rt)
        #rt = np.abs(rt)
        
        # this can be simplified so that we pass parameters directly to the simulator ...
        theta = np.array(model_config[model]['default_params'], dtype = np.float32)
        keys_tmp = self.parents.value.keys()
        cnt = 0
        
        for param in model_config[model]['params']:
            if param in keys_tmp:
                theta[cnt] = np.array(self.parents.value[param]).astype(np.float32)
            cnt += 1

        #print(rt)
        #print(response)
        #print(response.shape)
        #print(rt.shape)
        # response = 
        #pdf_fun = hddm.wfpt.wiener_like_nn_ddm_pdf
        # model_config[] # TODO FILL THIS IN SO THAT WE CREATE THE APPROPRIATE ARRAY AS INPUT TO THE SIMULATOR
        #out = pdf_multiplier * hddm.wfpt.wiener_pdf_cnn_2(x = rt, response = response, network = kwargs['network'], parameters = theta)# **kwargs) # This may still be buggy !
        out = hddm.wfpt.wiener_pdf_cnn_2(x = rt, response = response, network = kwargs['network'], parameters = theta)# **kwargs) # This may still be buggy !
        return out

    def cdf(self, x):
        # TODO: Implement the CDF method for neural networks
        return 'Not yet implemented'

    if model == 'ddm': # or model == 'weibull':
        def wienernn_like_ddm(x, 
                              v,
                              a,
                              z,
                              t,
                              p_outlier = 0,
                              w_outlier = 0,
                              **kwargs): #theta

            return hddm.wfpt.wiener_like_cnn_2(x['rt'].values,
                                               x['response'].values, 
                                               np.array([v, a, z, t], dtype = np.float32), 
                                               p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
                                               w_outlier = w_outlier,
                                               **kwargs)


        # Create wfpt class
        wfpt_nn = stochastic_from_dist('Wienernn_' + model, partial(wienernn_like_ddm, **kwargs))

        wfpt_nn.pdf = pdf
        wfpt_nn.cdf_vec = None # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
        wfpt_nn.cdf = cdf
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

            return hddm.wfpt.wiener_like_cnn_2(x['rt'].values,
                                               x['response'].values,
                                               np.array([v, a, z, t, alpha, beta], dtype = np.float32),
                                               p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
                                               w_outlier = w_outlier,
                                               **kwargs)
        # Create wfpt class
        wfpt_nn = stochastic_from_dist('Wienernn_' + model, partial(wienernn_like_weibull, **kwargs))

        wfpt_nn.pdf = pdf
        wfpt_nn.cdf_vec = None # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
        wfpt_nn.cdf = cdf
        wfpt_nn.random = random
        return wfpt_nn

    if model == 'levy':
        def wienernn_like_levy(x, 
                               v, 
                               a, 
                               alpha,
                               z,
                               t,
                               p_outlier = 0,
                               w_outlier = 0,
                               **kwargs): #theta

            return hddm.wfpt.wiener_like_cnn_2(x['rt'].values,
                                               x['response'].values,
                                               np.array([v, a, z, alpha, t], dtype = np.float32),
                                               p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
                                               w_outlier = w_outlier,
                                               **kwargs)
        
        # Create wfpt class
        wfpt_nn = stochastic_from_dist('Wienernn_' + model, partial(wienernn_like_levy, **kwargs))

        wfpt_nn.pdf = pdf
        wfpt_nn.cdf_vec = None # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
        wfpt_nn.cdf = cdf
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
    
            return hddm.wfpt.wiener_like_cnn_2(x['rt'].values,
                                               x['response'].values, 
                                               np.array([v, a, z, g, t], dtype = np.float32),
                                               p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
                                               w_outlier = w_outlier,
                                               **kwargs)
        # Create wfpt class
        wfpt_nn = stochastic_from_dist('Wienernn_' + model, partial(wienernn_like_ornstein, **kwargs))

        wfpt_nn.pdf = pdf
        wfpt_nn.cdf_vec = None # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
        wfpt_nn.cdf = cdf
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

            return hddm.wfpt.wiener_like_cnn_2(x['rt'].values,
                                               x['response'].values,
                                               np.array([v, a, z, t, sz, sv, st], dtype = np.float32),
                                               p_outlier = p_outlier,
                                               w_outlier = w_outlier,
                                               **kwargs)

        # Create wfpt class
        wfpt_nn = stochastic_from_dist('Wienernn_' + model, partial(wienernn_like_full_ddm, **kwargs))

        wfpt_nn.pdf = pdf
        wfpt_nn.cdf_vec = None # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
        wfpt_nn.cdf = cdf
        wfpt_nn.random = random
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

            return hddm.wfpt.wiener_like_cnn_2(x['rt'].values,
                                               x['response'].values,
                                               np.array([v, a, z, t, theta], dtype = np.float32),
                                               p_outlier = p_outlier,
                                               w_outlier = w_outlier,
                                               **kwargs)

        # Create wfpt class
        wfpt_nn = stochastic_from_dist('Wienernn_' + model, partial(wienernn_like_angle, **kwargs))

        wfpt_nn.pdf = pdf
        wfpt_nn.cdf_vec = None # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
        wfpt_nn.cdf = cdf
        wfpt_nn.random = random
        return wfpt_nn
    else:
        return 'Not implemented errror: Failed to load likelihood because the model specified is not implemented'


def generate_wfpt_nn_ddm_reg_stochastic_class(model = None,
                                              **kwargs):
    """Defines the regressor likelihoods for the CNN networks.

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
    def random(self):
        param_dict = deepcopy(self.parents.value)
        #print('param dict')
        #print(param_dict)
        del param_dict['reg_outcomes']
        sampled_rts = self.value.copy()
        #print('sampled rts')
        #print(sampled_rts)

        size = sampled_rts.shape[0]
        n_params = model_config[model]['n_params']
        param_data = np.zeros((size, n_params), dtype = np.float32)

        cnt = 0
        for tmp_str in model_config[model]['params']: #['v', 'a', 'z', 't']:
            if tmp_str in self.parents['reg_outcomes']:
                #print('param dict values')
                #print(param_dict[tmp_str].values[:, 0])
                param_data[:, cnt] = param_dict[tmp_str].values[:, 0]
            else:
                param_data[:, cnt] = param_dict[tmp_str]
            cnt += 1

        # for i in self.value.index:
        #     #get current params
        #     for p in self.parents['reg_outcomes']:
        #         param_dict[p] = np.asscalar(self.parents.value[p].loc[i])
        #     #sample
        #     samples = hddm.generate.gen_rts(method=sampling_method,
        #                                     size=1, dt=sampling_dt, **param_dict)

        #     sampled_rts.loc[i, 'rt'] = hddm.utils.flip_errors(samples).rt

        sim_out = simulator(theta = param_data, n_trials = size, model = model, n_samples = 1, max_t = 20)
        # sim_out_copy = []
        # sim_out_copy.append(np.squeeze(sim_out[0], axis = 0))
        # sim_out_copy.append(np.squeeze(sim_out[1], axis = 0))
        # sim_out_copy.append(sim_out[2])
        return hddm_preprocess(sim_out, keep_negative_responses = True)

    if model == 'ddm':
        def wiener_multi_like_nn_ddm(value, 
                                     v,
                                     a,
                                     z,
                                     t,
                                     reg_outcomes,
                                     p_outlier = 0,
                                     w_outlier = 0,
                                     **kwargs): #theta

            params = {'v': v, 'a': a, 'z': z, 't': t}
            n_params = 4 #model_config[model]['n_params']
            size = int(value.shape[0])
            data = np.zeros((size, n_params), dtype = np.float32)

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

            #print(value['rt'].values.astype(np.int_))
            return hddm.wfpt.wiener_like_reg_cnn_2(value['rt'].values.astype(np.int_),
                                                   value['response'].values.astype(np.int_), 
                                                   data, 
                                                   p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
                                                   w_outlier = w_outlier,
                                                   **kwargs)
    
        stoch = stochastic_from_dist('wfpt_reg', partial(wiener_multi_like_nn_ddm, **kwargs))
        stoch.random = random
        
    if model == 'full_ddm' or model == 'full_ddm2':
        def wiener_multi_like_nn_full_ddm(value, v, sv, a, z, sz, t, st, 
                                         reg_outcomes, 
                                         p_outlier = 0, 
                                         w_outlier = 0.1,
                                         **kwargs):

            params = {'v': v, 'a': a, 'z': z, 't': t, 'sz': sz, 'sv': sv, 'st': st}

            n_params = int(7)
            size = int(value.shape[0])
            data = np.zeros((size, n_params), dtype = np.float32)

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

            # THIS IS NOT YET FINISHED !
            return hddm.wfpt.wiener_like_reg_cnn_2(value['rt'].values.astype(np.int_),
                                                   value['response'].values.astype(np.int_), 
                                                   data, 
                                                   p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
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

            """Log-likelihood for the full DDM using the interpolation method"""

            params = {'v': v, 'a': a, 'z': z, 't': t, 'theta': theta}
            n_params = int(5)
            size = int(value.shape[0])
            data = np.zeros((size, n_params), dtype = np.float32)

            cnt = 0
            for tmp_str in ['v', 'a', 'z', 't', 'theta']:

                if tmp_str in reg_outcomes:
                    data[:, cnt] = params[tmp_str].loc[value['rt'].index].values[:, 0]
                    if (data[:, cnt].min() < model_config[model]['param_bounds'][0][cnt]) or (data[:, cnt].max() > model_config[model]['param_bounds'][1][cnt]):
                        print('boundary violation of regressor part')
                        return - np.inf
                else:
                    data[:, cnt] = params[tmp_str]

                cnt += 1

            # THIS IS NOT YET FINISHED !
            # THIS IS NOT YET FINISHED !
            return hddm.wfpt.wiener_like_reg_cnn_2(value['rt'].values.astype(np.int_),
                                                   value['response'].values.astype(np.int_), 
                                                   data, 
                                                   p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
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

            """Log-likelihood for the full DDM using the interpolation method"""
            params = {'v': v, 'a': a, 'z': z, 'alpha': alpha, 't': t}
            n_params = int(5)
            size = int(value.shape[0])
            data = np.zeros((size, n_params), dtype = np.float32)

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

            # THIS IS NOT YET FINISHED !
            return hddm.wfpt.wiener_like_reg_cnn_2(value['rt'].values.astype(np.int_),
                                                   value['response'].values.astype(np.int_), 
                                                   data, 
                                                   p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
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

            params = {'v': v, 'a': a, 'z': z, 'g': g, 't': t}
            n_params = int(5)
            size = int(value.shape[0])
            data = np.zeros((size, n_params), dtype = np.float32)

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

            # THIS IS NOT YET FINISHED !
            return hddm.wfpt.wiener_like_reg_cnn_2(value['rt'].values.astype(np.int_),
                                                   value['response'].values.astype(np.int_), 
                                                   data, 
                                                   p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
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

            params = {'v': v, 'a': a, 'z': z, 't': t, 'alpha': alpha, 'beta': beta}
            n_params = int(6)
            size = int(value.shape[0])
            data = np.zeros((size, n_params), dtype = np.float32)

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

            # THIS IS NOT YET FINISHED !
            return hddm.wfpt.wiener_like_reg_cnn_2(value['rt'].values.astype(np.int_),
                                                   value['response'].values.astype(np.int_), 
                                                   data, 
                                                   p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
                                                   w_outlier = w_outlier,
                                                   **kwargs)

        stoch = stochastic_from_dist('wfpt_reg', partial(wiener_multi_like_nn_weibull, **kwargs))
        stoch.random = random
    
    return stoch

