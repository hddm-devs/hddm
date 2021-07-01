import pandas as pd
import numpy as np
from copy import deepcopy
#import re
import argparse
import sys
import pickle
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import truncnorm
from patsy import dmatrix
from collections import OrderedDict
from hddm.simulators.basic_simulator import *

# Helper
def hddm_preprocess(simulator_data = None, subj_id = 'none', keep_negative_responses = False, add_model_parameters = False, keep_subj_idx = True):
    #print(simulator_data)
    # Define dataframe if simulator output is normal (comes out as list tuple [rts, choices, metadata])
    if len(simulator_data) == 3:
        df = pd.DataFrame(simulator_data[0].astype(np.double), columns = ['rt'])
        df['response'] = simulator_data[1].astype(int)
    # Define dataframe if simulator output is binned pointwise (comes out as tuple [np.array, metadata])
    
    # I think this part is never called !
    if len(simulator_data) == 2:
        df = pd.DataFrame(simulator_data[0][:, 0], columns = ['rt'])
        df['response'] = simulator_data[0][:, 1].astype(int)

    #df['nn_response'] = df['response']
    if not keep_negative_responses:
        df.loc[df['response'] == -1.0, 'response'] = 0.0
    
    if keep_subj_idx:
        df['subj_idx'] = subj_id

    # Add ground truth parameters to dataframe
    if add_model_parameters:
        for param in model_config[simulator_data[2]['model']]['params']:
            if len(simulator_data[2][param]) > 1:
                df[param] = simulator_data[2][param]
            else:
                df[param] = simulator_data[2][param][0]
    return df

def str_to_num(string = '', n_digits = 3):
    new_str = ''
    leading = 1
    for digit in range(n_digits):
        if string[digit] == '0' and leading and (digit < n_digits - 1):
            pass
        else:
            new_str += string[digit]
            leading = 0
    return int(new_str)

def num_to_str(num = 0, n_digits = 3):
    new_str = ''
    for i in range(n_digits - 1, -1, -1):
        if num < np.power(10, i):
            new_str += '0'
    if num != 0:
        new_str += str(num)
    return new_str

def pad_subj_id(in_str):
    # Make subj ids have three digits by prepending 0s if necessary
    stridx = in_str.find('.') # get index of 'subj.' substring
    subj_idx_len = len(in_str[(stridx + len('.')):]) # check how many letters remain after 'subj.' is enocuntered
    out_str = ''
    prefix_str = ''
    for i in range(3 - subj_idx_len):
        prefix_str += '0' # add zeros to pad subject id to have three digits

    out_str = in_str[:stridx + len('.')] + prefix_str + in_str[stridx + len('.'):] #   
    # print(out_str)
    return out_str

def _add_outliers(sim_out = None, 
                  p_outlier = None,
                  n_samples = None,
                  max_rt_outlier = 0.05,
                  ):

    if p_outlier == 0:
        return sim_out
    else:
        # Sample number of outliers from appropriate binomial
        n_outliers = np.random.binomial(n = n_samples, p = p_outlier)

        # Only if the sampled number of outliers is above 0,
        # do we bother generating and storing them
        if n_outliers > 0:
            # Initialize the outlier data
            outlier_data = np.zeros((n_outliers, 2))

            # Generate outliers
            # Reaction times are uniform between 0 and 1/max_rt_outlier (default 1 / 0.1)
            # Choice are random with equal probability among the valid choice options
            outlier_data[:, 0] = np.random.uniform(low = 0.0, high = max_rt_outlier, size = n_outliers)
            outlier_data[:, 1] = np.random.choice(sim_out[2]['possible_choices'], size = n_outliers)

            # Exchange the last parts of the simulator data for the outliers
            sim_out[0][-n_outliers:, 0] = outlier_data[:, 0]
            sim_out[1][-n_outliers:, 0] = outlier_data[:, 1]
    return 

# -------------------------------------------------------------------------------------
# Parameter set generator
def make_parameter_vectors_nn(model = 'angle',
                              param_dict = None,
                              n_parameter_vectors = 10):
    """Generates a (number of) parameter vector(s) for a given model. 

    :Arguments:

        model: str <default='angle'>
            String that specifies the model to be simulated. 
            Current options include, 'angle', 'ornstein', 'levy', 'weibull', 'full_ddm'
        param_dict: dict <default=None>
            Dictionary of parameter values that you would like to pre-specify. The dictionary takes the form (for the simple examples of the ddm),
            {'v': [0], 'a': [1.5]} etc.. For a given key supply either a list of length 1, or a list of 
            length equal to the n_parameter_vectors argument supplied.
        n_parameter_vectors: int <default=10>
            Nuber of parameter vectors you want to generate

    Return: pandas.DataFrame
            Columns are parameter names and rows fill the parameter values.
    """
    
    parameter_data = np.zeros((n_parameter_vectors, len(model_config[model]['params'])))
    
    if param_dict is not None:
        cnt = 0
        for param in model_config[model]['params']:

            if param in param_dict.keys():

                if (len(param_dict[param]) == n_parameter_vectors) or (len(param_dict[param]) == 1):
                    # Check if parameters are properly in bounds
                    if np.sum(np.array(param_dict[param]) < model_config[model]['param_bounds'][0][cnt]) > 0 \
                    or np.sum(np.array(param_dict[param]) > model_config[model]['param_bounds'][1][cnt]) > 0:
                        
                        print('The parameter: ', 
                              param, 
                              ', is out of the accepted bounds [', 
                              model_config[model]['param_bounds'][0][cnt], 
                              ',', 
                              model_config[model]['param_bounds'][1][cnt], ']')
                        return 
                    else:
                        parameter_data[:, cnt] = param_dict[param]
                else:
                    print('Param dict not specified correctly. Lengths of parameter lists needs to be 1 or equal to n_param_sets')

            else:
                parameter_data[:, cnt] = np.random.uniform(low = model_config[model]['param_bounds'][0][cnt],
                                                           high = model_config[model]['param_bounds'][1][cnt], 
                                                           size = n_parameter_vectors)
            cnt += 1
    else:
        parameter_data = np.random.uniform(low = model_config[model]['param_bounds'][0],
                                           high = model_config[model]['param_bounds'][1],
                                           size = (n_parameter_vectors, len(model_config[model]['params'])))
                                           
    return pd.DataFrame(parameter_data, columns = model_config[model]['params'])

# Dataset generators
def simulator_single_subject(parameters = [0, 0, 0],
                             p_outlier = 0.0,
                             max_rt_outlier = 10.0,
                             model = 'angle',
                             n_samples = 1000,
                             delta_t = 0.001,
                             max_t = 20,
                             bin_dim = None,
                             bin_pointwise = False):
    """Generate a hddm-ready dataset from a single set of parameters

    :Arguments:
        parameters: dict, list or numpy array
            Model parameters with which to simulate. Dict is preferable for informative error messages.
            If you know the order of parameters for your model of choice, you can also directly supply a
            list or nump.array which needs to have the parameters in the correct order.
        p_outlier: float between 0 and 1 <default=0>
            Probability of generating outlier datapoints. An outlier is defined 
            as a random choice from a uniform RT distribution
        max_rt_outlier: float > 0 <default=10.0>
            Using max_rt_outlier (which is commonly defined for hddm models) here as an imlicit maximum 
            on the RT of outliers. Outlier RTs are sampled uniformly from [0, max_rt_outlier]
        model: str <default='angle'>
            String that specifies the model to be simulated. 
            Current options include, 'angle', 'ornstein', 'levy', 'weibull', 'full_ddm'
        n_samples: int <default=1000>
            Number of samples to simulate.
        delta_t: float <default=0.001>
            Size fo timesteps in simulator (conceptually measured in seconds)
        max_t: float <default=20>
            Maximum reaction the simulator can reach
        bin_dim: int <default=None>
            If simulator output should be binned, this specifies the number of bins to use
        bin_pointwise: bool <default=False>
            Determines whether to bin simulator output pointwise. Pointwise here is in contrast to producing binned output
            in the form of a histogram. Binning pointwise gives each trial's RT and index which is the respective bin-number. 
            This is expected when you are using the 'cnn' network to fit the dataset later. If pointwise is not chosen,
            then the takes the form of a histogram, with bin-wise frequencies.

    Return: tuple of (pandas.DataFrame, dict, list)
        The first part of the tuple holds a DataFrame with a 'reaction time' column and a 'response' column. Ready to be fit with hddm.
        The second part of the tuple hold a dict with parameter names as keys and parameter values as values.
        The third part gives back the parameters supplied in array form.
        This return is consistent with the returned objects in other data generators under hddm.simulators
    """
    
    # Sanity checks
    assert p_outlier >= 0 and p_outlier <= 1, 'p_outlier is not between 0 and 1'
    assert max_rt_outlier > 0, 'max_rt__outlier needs to be > 0'

    print('Model: ', model)
    print('Parameters needed: ', model_config[model]['params'])
    if type(parameters) != dict:
        gt = {}
        for param in model_config[model]['params']:
            id_tmp = model_config[model]['params'].index(param)
            gt[param] = parameters[id_tmp]      
    else:  
        gt = parameters.copy()
        parameters = []
        for param in model_config[model]['params']:
            if param in gt.keys():
                parameters.append(gt[param])
            else:
                print('The parameter ', param, ' was not supplied to the function.')
                print('Taking default', param, ' from hddm.model_config.')
                parameters.append(model_config[model]['default_params'][model_config[model]['params'].index(param)])

    x = simulator(theta = parameters,
                  model = model,
                  n_samples = n_samples,
                  delta_t = delta_t,
                  max_t = max_t,
                  bin_dim = bin_dim,
                  bin_pointwise = bin_pointwise)

    # Add outliers 
    # (Potentially 0 outliers)
    x = _add_outliers(sim_out = x,
                      p_outlier = p_outlier,
                      n_samples = n_samples,
                      max_rt_outlier = max_rt_outlier)
        
    data_out = hddm_preprocess(x, add_model_parameters = True)

    return (data_out, gt)

# TD: DIDN'T GO OVER THIS ONE YET !
def simulator_stimcoding(model = 'angle',
                         split_by = 'v',
                         p_outlier = 0.0,
                         max_rt_outlier = 10.0,
                         drift_criterion = 0.0,
                         n_samples_by_condition = 1000,
                         delta_t = 0.001,
                         prespecified_params = {},
                         bin_pointwise = False,
                         bin_dim = None,
                         max_t = 20.0):

    """Generate a dataset as expected by Hddmstimcoding. Essentially it is a specific way to parameterize two condition data.

    :Arguments:
        parameters: list or numpy array
            Model parameters with which to simulate.
        model: str <default='angle'>
            String that specifies the model to be simulated. 
            Current options include, 'angle', 'ornstein', 'levy', 'weibull', 'full_ddm'
        split_by: str <default='v'>
            You can split by 'v' or 'z'. If splitting by 'v' one condition's v_0 = drift_criterion + 'v', the other 
            condition's v_1 = drift_criterion - 'v'.
            Respectively for 'z', 'z_0' = 'z' and 'z_1' = 1 - 'z'.
        p_outlier: float between 0 and 1 <default=0>
            Probability of generating outlier datapoints. An outlier is defined 
            as a random choice from a uniform RT distribution
        max_rt_outlier: float > 0 <default=10.0>
            Using max_rt_outlier (which is commonly defined for hddm models) here as an imlicit maximum 
            on the RT of outliers. Outlier RTs are sampled uniformly from [0, max_rt_outlier]
        drift_criterion: float <default=0.0>
            Parameter that can be treated as the 'bias part' of the slope, in case we split_by 'v'.
        n_samples_by_condition: int <default=1000>
            Number of samples to simulate per condition (here 2 condition by design).
        delta_t: float <default=0.001>
            Size fo timesteps in simulator (conceptually measured in seconds)
        prespecified_params: dict <default = {}>
            A dictionary with parameter names keys. Values are list of either length 1, or length equal to the number of conditions (here 2).
        max_t: float <default=20>
            Maximum reaction the simulator can reach
        bin_dim: int <default=None>
            If simulator output should be binned, this specifies the number of bins to use
        bin_pointwise: bool <default=False>
            Determines whether to bin simulator output pointwise. Pointwise here is in contrast to producing binned output
            in the form of a histogram. Binning pointwise gives each trial's RT and index which is the respective bin-number. 
            This is expected when you are using the 'cnn' network to fit the dataset later. If pointwise is not chosen,
            then the takes the form of a histogram, with bin-wise frequencies.

    Return: pandas.DataFrame holding a 'reaction time' column and a 'response' column. Ready to be fit with hddm.
    """
    
    param_base = np.tile(np.random.uniform(low = model_config[model]['param_bounds'][0],
                                           high = model_config[model]['param_bounds'][1], 
                                           size = (1, len(model_config[model]['params']))),
                                           (2, 1))

    # Fill in prespecified parameters if supplied
    if prespecified_params is not None:
        if type(prespecified_params) == dict:
            for param in prespecified_params:
                id_tmp = model_config[model]['params'].index(param)
                param_base[:, id_tmp] = prespecified_params[param]
        else:
            print('prespecified_params is not supplied as a dictionary, please reformat the input')
            return

    
    if type(split_by) == list:
        pass
    elif type(split_by) == str:
        split_by = [split_by]
    else:
        print('Can not recognize data-type of argument: split_by, provided neither a list nor a string')
        return
    gt = {}

    for i in range(2):
        
        if i == 0:
#             param_base[i, id_tmp] = np.random.uniform(low = model_config[model]['param_bounds'][0][id_tmp], 
#                                                       high = model_config[model]['param_bounds'][1][id_tmp])
            if 'v' in split_by:
                id_tmp = model_config[model]['params'].index('v')
                param_base[i, id_tmp] = drift_criterion - param_base[i, id_tmp]
                gt['v'] = param_base[i, id_tmp]
                gt['dc'] = drift_criterion
   
        if i == 1:
            
            if 'v' in split_by:
                id_tmp = model_config[model]['params'].index('v')
                param_base[i, id_tmp] = drift_criterion + param_base[i, id_tmp]
            if 'z' in split_by:
                id_tmp = model_config[model]['params'].index('z')
                param_base[i, id_tmp] = 1 - param_base[i, id_tmp]
            
    #print(param_base)
    dataframes = []
    for i in range(2):
        
        sim_out = simulator(param_base[i, :], 
                            model = model, 
                            n_samples = n_samples_by_condition,
                            bin_dim = bin_dim,
                            bin_pointwise = bin_pointwise,
                            max_t = max_t,
                            delta_t = delta_t)
        
        sim_out = _add_outliers(sim_out = sim_out,
                                p_outlier = p_outlier,
                                n_samples = n_samples_by_condition,
                                max_rt_outlier = max_rt_outlier)
        

        dataframes.append(hddm_preprocess(simulator_data = sim_out, subj_id = i + 1), keep_model_parameters = True)
    
    data_out = pd.concat(dataframes)
    data_out = data_out.rename(columns = {'subj_idx': "stim"})
    data_out['subj_idx'] = 'none'
    # print(param_base.shape)
    return (data_out, gt)

def simulator_condition_effects(n_conditions = 4,
                                n_samples_by_condition = 1000,
                                condition_effect_on_param = None,
                                prespecified_params = None,
                                p_outlier = 0.0,
                                max_rt_outlier = 10.0,
                                model = 'angle',
                                bin_dim = None,
                                bin_pointwise = False,
                                max_t = 20.0,
                                delta_t = 0.001,
                                ):

    """ Generate a dataset with multiple conditions.

    :Arguments:
        n_conditions: int <default=4>

        parameters: list or numpy array
            Model parameters with which to simulate.
        model: str <default='angle'>
            String that specifies the model to be simulated. 
            Current options include, 'angle', 'ornstein', 'levy', 'weibull', 'full_ddm'
        n_samples_by_condition: int <default=1000>
            Number of samples to simulate per condition (here 2 condition by design).
        condition_effect_on_param: list of strings <default=None>
            List containing the parameters which will be affected by the condition.
        delta_t: float <default=0.001>
            Size fo timesteps in simulator (conceptually measured in seconds)
        prespecified_params: dict <default = {}>
            A dictionary with parameter names keys. Values are list of either length 1, or length equal to the number of conditions.
        p_outlier: float between 0 and 1 <default=0>
            Probability of generating outlier datapoints. An outlier is defined 
            as a random choice from a uniform RT distribution
        max_rt_outlier: float > 0 <default=10.0>
            Using max_rt_outlier (which is commonly defined for hddm models) here as an imlicit maximum 
            on the RT of outliers. Outlier RTs are sampled uniformly from [0, max_rt_outlier]
        max_t: float <default=20>
            Maximum reaction the simulator can reach
        bin_dim: int <default=None>
            If simulator output should be binned, this specifies the number of bins to use
        bin_pointwise: bool <default=False>
            Determines whether to bin simulator output pointwise. Pointwise here is in contrast to producing binned output
            in the form of a histogram. Binning pointwise gives each trial's RT and index which is the respective bin-number. 
            This is expected when you are using the 'cnn' network to fit the dataset later. If pointwise is not chosen,
            then the takes the form of a histogram, with bin-wise frequencies.

    Returns: 
       tuple (pandas.DataFrame, dictionary): The DataFrame holds 'reaction time', 'response', 'stim' columns as well as trial by trial parameters. Ready to be fit with hddm.
       The dictionary holds the ground truth parameters with names as one expects from hddm model traces.
    """

    # Sanity checks
    assert p_outlier >= 0 and p_outlier <= 1, 'p_outlier is not between 0 and 1'
    assert max_rt_outlier > 0, 'max_rt__outlier needs to be > 0'

    # Get list of keys in prespecified_params and return if it is not a dict when it is in fact not None
    if prespecified_params is not None:
        if type(prespecified_params) == dict:
            prespecified_params_names = list(prespecified_params.keys())
        else:
            print('prespecified_params is not a dictionary')
            return
               
    
    # Randomly assign values to every parameter and then copy across rows = number of conditions
    param_base = np.tile(np.random.uniform(low = model_config[model]['param_bounds'][0],
                                            high = model_config[model]['param_bounds'][1], 
                                            size = (1, len(model_config[model]['params']))),
                                            (n_conditions, 1))
    
         
    # Reassign parameters according to the information in prespecified params and condition_effect_on_param
    gt = {}

    # Loop over valid model parameters
    for param in model_config[model]['params']:
        id_tmp = model_config[model]['params'].index(param)
        
        # Check if parameter is affected by condition
        if param in condition_effect_on_param:
            
            # If parameter is affected by condition we loop over conditions
            for i in range(n_conditions):
                # Assign randomly
                param_base[i, id_tmp] = np.random.uniform(low = model_config[model]['param_bounds'][0][id_tmp], 
                                                          high = model_config[model]['param_bounds'][1][id_tmp])
                
                # But if we actually specified it for each condition
                if prespecified_params is not None:
                    if param in prespecified_params_names:
                        # We assign it from prespecified dictionary
                        param_base[i, id_tmp] = prespecified_params[param][i] 
                
                # Add info to ground truth dictionary
                gt[param + '(' + str(i) + ')'] = param_base[i, id_tmp]
        
        # If the parameter is not affected by condition     
        else:
            # But prespecified
            if prespecified_params is not None:
                
                if param in prespecified_params_names:
                    # We assign prespecifided param
                    tmp_param = prespecified_params[param]
                    param_base[:, id_tmp] = tmp_param   

            # If it wasn't prespecified we just keep the random assignment that was generated above before the loops
            gt[param] = param_base[0, id_tmp]
    
    dataframes = []
    for i in range(n_conditions):
        sim_out = simulator(param_base[i, :],
                            model = model, 
                            n_samples = n_samples_by_condition,
                            bin_dim = bin_dim,
                            bin_pointwise = bin_pointwise,
                            max_t = max_t,
                            delta_t = delta_t)
        
        sim_out = _add_outliers(sim_out = sim_out,
                                p_outlier = p_outlier,
                                n_samples = n_samples_by_condition,
                                max_rt_outlier = max_rt_outlier)
        
        dataframes.append(hddm_preprocess(simulator_data = sim_out, subj_id = i, add_model_parameters = True))
    
    data_out = pd.concat(dataframes)
    
    # Change 'subj_idx' column name to 'condition' ('subj_idx' is assigned automatically by hddm_preprocess() function)
    data_out = data_out.rename(columns = {'subj_idx': "condition"})
    data_out['subj_idx'] = 0
    data_out.reset_index(drop = True, inplace = True)

    if bin_pointwise:
        data_out['rt'] = data_out['rt'].astype(np.int_)
        data_out['response'] = data_out['response'].astype(np.int_)
        #data_out['nn_response'] = data_out['nn_response'].astype(np.int_)

    return (data_out, gt)

def simulator_covariate(dependent_params = ['v'],
                        model = 'angle',
                        n_samples = 1000,
                        betas = {'v': 0.1},
                        covariate_magnitudes = {'v': 1.0},
                        prespecified_params = None,
                        p_outlier = 0.0,
                        max_rt_outlier = 10.0,
                        subj_id = 'none',
                        bin_dim = None, 
                        bin_pointwise = False,
                        max_t = 20.0,
                        delta_t = 0.001,
                        ):

    """ Generate a dataset which includes covariates. Some parameters are now a function (regression) covariates.

    :Arguments:
        dependent_params: list of strings <default=['v']>
            Parameters which will be treated as a deterministic function of a covariate
        prespecified_params: dict <default=None>
            Dictionary of parameters to prespecify. These parameters can not be functions of covariates.
            Example (e.g. for 'ddm' model): {'v': 0, 'a': 1.5, 'z': 0.5, 't':1.0}
        p_outlier: float between 0 and 1 <default=0>
            Probability of generating outlier datapoints. An outlier is defined 
            as a random choice from a uniform RT distribution
        max_rt_outlier: float > 0 <default=10.0>
            Using max_rt_outlier (which is commonly defined for hddm models) here as an imlicit maximum 
            on the RT of outliers. Outlier RTs are sampled uniformly from [0, max_rt_outlier]
        model: str <default='angle'>
            String that specifies the model to be simulated. 
            Current options include, 'angle', 'ornstein', 'levy', 'weibull', 'full_ddm'
        betas: dict <default={'v': 0.1}>
            Ground truth regression betas for the parameters which are functions of covariates.
        covariates_magnitudes: dict <default={'v': 1.0}>
            A dict which holds magnitudes of the covariate vectors (value), by for each parameters (key).
        subj_id: str <default='none'>
            Hddm expects a subject column in the dataset. This supplies a specific label if so desired.
        delta_t: float <default=0.001>
            Size fo timesteps in simulator (conceptually measured in seconds)
        max_t: float <default=20>
            Maximum reaction the simulator can reach
        bin_dim: int <default=None>
            If simulator output should be binned, this specifies the number of bins to use
        bin_pointwise: bool <default=False>
            Determines whether to bin simulator output pointwise. Pointwise here is in contrast to producing binned output
            in the form of a histogram. Binning pointwise gives each trial's RT and index which is the respective bin-number. 
            This is expected when you are using the 'cnn' network to fit the dataset later. If pointwise is not chosen,
            then the takes the form of a histogram, with bin-wise frequencies.

    Returns: 
      tuple (pandas.DataFrame, dictionary): The DataFrame holds 'reaction time', 'response', 'BOLD' (the covariate) columns as well as trial by trial parameters. Ready to be fit with hddm.
       The dictionary holds the ground truth parameters with names as one expects from hddm model traces.
    """

    # Sanity checks
    assert p_outlier >= 0 and p_outlier <= 1, 'p_outlier is not between 0 and 1'
    assert max_rt_outlier > 0, 'max_rt__outlier needs to be > 0'

    if betas == None:
        betas = {}
    if covariate_magnitudes == None:
        covariate_magnitudes = {}
    if len(dependent_params) < 1:
        print('If there are no dependent variables, no need for the simulator which includes covariates')
        return

    # sanity check that prespecified parameters do not clash with parameters that are supposed to derive from trial-wise regression
    if prespecified_params is not None:
        for param in prespecified_params:
            if param in covariate_magnitudes.keys() or param in betas.keys():
                'Parameters that have covariates are Prespecified, this should not be intented'
                return

    # Fill parameter matrix
    param_base = np.tile(np.random.uniform(low = model_config[model]['param_bounds'][0],
                                           high = model_config[model]['param_bounds'][1], 
                                           size = (1, len(model_config[model]['params']))),
                                           (n_samples, 1))

    # Adjust any parameters that where prespecified
    if prespecified_params is not None:
        for param in prespecified_params.keys():
            id_tmp = model_config[model]['params'].index(param)
            param_base[:, id_tmp] = prespecified_params[param]

    # TD: Be more clever about covariate magnitude (maybe supply?)
    # Parameters that have a
    param_base_before_adj = param_base.copy()

    for covariate in dependent_params:
        id_tmp = model_config[model]['params'].index(covariate)

        if covariate in covariate_magnitudes.keys():
            tmp_covariate_by_sample = np.random.uniform(low = - covariate_magnitudes[covariate], 
                                                        high = covariate_magnitudes[covariate], 
                                                        size = n_samples)
        else:
            tmp_covariate_by_sample = np.random.uniform(low = - 1, 
                                                        high = 1, 
                                                        size = n_samples)

        # If the current covariate has a beta parameter attached to it 
        if covariate in betas.keys():

            param_base[:, id_tmp] = param_base[:, id_tmp] + (betas[covariate] * tmp_covariate_by_sample)
        else: 
            param_base[:, id_tmp] = param_base[:, id_tmp] + (0.1 * tmp_covariate_by_sample)
    
    # TD: IMPROVE THIS SIMULATOR SO THAT WE CAN PASS MATRICES OF PARAMETERS
    # WAY TOO SLOW RIGHT NOW
    
    sim_out = simulator(param_base,
                        model = model,
                        n_samples = 1,
                        n_trials = n_samples,
                        bin_dim = bin_dim,
                        bin_pointwise = bin_pointwise,
                        max_t = max_t,
                        delta_t = delta_t)

    sim_out = _add_outliers(sim_out = sim_out,
                            p_outlier = p_outlier,
                            n_samples = n_samples,
                            max_rt_outlier = max_rt_outlier)
    
    # Preprocess 
    data = hddm_preprocess(sim_out, subj_id, add_model_parameters = True)
    # data = hddm_preprocess([rts, choices], subj_id)
    
    # Call the covariate BOLD (unnecessary but in style)
    data['BOLD'] = tmp_covariate_by_sample

    # Make ground truth dict
    gt = {}
    
    for param in model_config[model]['params']:
        id_tmp = model_config[model]['params'].index(param)
        
        # If a parameter actually had a covariate attached then we add the beta coefficient as a parameter as well
        # Now intercept, beta
        if param in dependent_params:
            if param in betas.keys():
            # gt[param + '_Intercept'] = param_base[:, ]
                gt[param + '_BOLD'] = betas[param]
            else:
                gt[param + '_BOLD'] = 0.1
            gt[param + '_Intercept'] = param_base_before_adj[0, id_tmp]
            
        else:
            gt[param] = param_base[0, id_tmp]
    
    return (data, gt)

# ALEX TD: Change n_samples_by_subject --> n_trials_per_subject (but apply consistently)
def simulator_hierarchical(n_subjects = 5,
                           n_samples_by_subject = 500,
                           prespecified_param_means = None, # {'v': 2},
                           prespecified_param_stds = None, # {'v': 0.3},
                           p_outlier = 0.0,
                           max_rt_outlier = 10.0,
                           model = 'angle',
                           bin_dim = None,
                           bin_pointwise = True,
                           max_t = 20.0,
                           delta_t = 0.001):

    """Generate a dataset which includes covariates. Some parameters are now a function (regression) covariates.

    :Arguments:
        n_subjects: int <default=5>
            Number of subjects in the datasets
        n_samples_by_subject: int <default=500>
            Number of trials for each subject
        prspecified_param_means: dict <default={'v': 2}>
            Prespeficied group means 
        prespecified_param_stds: dict <default={'v': 0.3}
            Prespeficied group standard deviations
        p_outlier: float between 0 and 1 <default=0>
            Probability of generating outlier datapoints. An outlier is defined 
            as a random choice from a uniform RT distribution
        max_rt_outlier: float > 0 <default=10.0>
            Using max_rt_outlier (which is commonly defined for hddm models) here as an imlicit maximum 
            on the RT of outliers. Outlier RTs are sampled uniformly from [0, max_rt_outlier]
        model: str <default='angle'>
            String that specifies the model to be simulated. 
            Current options include, 'angle', 'ornstein', 'levy', 'weibull', 'full_ddm'
        delta_t: float <default=0.001>
            Size fo timesteps in simulator (conceptually measured in seconds)
        max_t: float <default=20>
            Maximum reaction the simulator can reach
        bin_dim: int <default=None>
            If simulator output should be binned, this specifies the number of bins to use
        bin_pointwise: bool <default=False>
            Determines whether to bin simulator output pointwise. Pointwise here is in contrast to producing binned output
            in the form of a histogram. Binning pointwise gives each trial's RT and index which is the respective bin-number. 
            This is expected when you are using the 'cnn' network to fit the dataset later. If pointwise is not chosen,
            then the takes the form of a histogram, with bin-wise frequencies.

    Returns: 
        (pandas.DataFrame, dict, np.array): The Dataframe holds a 'reaction time' column, a 'response' column and a 'BOLD' column (for the covariate). The dictionary holds the groundtruth parameter (values) and parameter names (keys).
                                           Ready to be fit with hddm.
    """

    # Sanity checks
    assert p_outlier >= 0 and p_outlier <= 1, 'p_outlier is not between 0 and 1'
    assert max_rt_outlier > 0, 'max_rt__outlier needs to be > 0'

    # AF TD: Why is this unused ?!
    # param_ranges_half = (np.array(model_config[model]['param_bounds'][1]) - np.array(model_config[model]['param_bounds'][0])) / 2
    # Fill in some global parameter vectors
    global_means = np.random.uniform(low = model_config[model]['param_bounds'][0],
                                     high = model_config[model]['param_bounds'][1],
                                     size = (1, len(model_config[model]['param_bounds'][0])))                     

    global_stds = np.random.uniform(low = 0.001, 
                                    high = np.minimum(abs(global_means - model_config[model]['param_bounds'][0]), 
                                                      abs(model_config[model]['param_bounds'][1] - global_means)) / 3, # previously param_ranges_half / 6,
                                    size = (1, len(model_config[model]['param_bounds'][0])))
    
    dataframes = []
    subject_parameters = np.zeros((n_subjects, 
                                   len(model_config[model]['param_bounds'][0])))
    gt = {}
    
    # Update global parameter vectors according to what was pre-specified
    for param in model_config[model]['params']:
        id_tmp = model_config[model]['params'].index(param)
        if prespecified_param_means is not None:
            if param in prespecified_param_means.keys():
                global_means[0, id_tmp] = prespecified_param_means[param]

        if prespecified_param_stds is not None:
            if param in prespecified_param_stds.keys():
                global_stds[0, id_tmp] = prespecified_param_stds[param]
        
        gt[param] = global_means[0, id_tmp]
        gt[param + '_std'] = global_stds[0, id_tmp]
    
    # For each subject get subject level parameters by sampling from a truncated gaussian as speficied by the global parameters above
    for i in range(n_subjects):
        subj_id = num_to_str(i)
        
        # Get subject parameters
        a = (model_config[model]['param_bounds'][0] - global_means[0, :]) / global_stds[0, :]
        b = (model_config[model]['param_bounds'][1] - global_means[0, :]) / global_stds[0, :]
        
        subject_parameters[i, :] = np.float32(global_means[0, :] + (truncnorm.rvs(a, b, size = global_stds.shape[1]) * global_stds[0, :]))
        
        sim_out = simulator(subject_parameters[i, :],
                            model = model,
                            n_samples = n_samples_by_subject,
                            bin_dim = bin_dim,
                            bin_pointwise = bin_pointwise,
                            max_t = max_t,
                            delta_t = delta_t)

        sim_out = _add_outliers(sim_out = sim_out,
                                p_outlier = p_outlier,
                                n_samples = n_samples_by_subject,
                                max_rt_outlier = max_rt_outlier)
        
        dataframes.append(hddm_preprocess(simulator_data = sim_out, 
                                          subj_id = subj_id,
                                          add_model_parameters = True))
        
        for param in model_config[model]['params']:
            id_tmp = model_config[model]['params'].index(param)
            gt[param + '_subj.' + subj_id] = subject_parameters[i, id_tmp]
        
    data_out = pd.concat(dataframes)
    
    return (data_out, gt)

### NEW
def simulator_h_c(n_subjects = 10,
                   n_samples_by_subject = 100,
                   model = 'ddm_vanilla',
                   conditions = {'c_one': ['high', 'low'], 'c_two': ['high', 'low'], 'c_three': ['high', 'medium', 'low']},
                   depends_on = {'v': ['c_one', 'c_two']},
                   regression_models = ['z ~ covariate_name'],
                   regression_covariates = {'covariate_name': {'type': 'categorical', 'range': (0, 4)}}, # need this to make initial covariate matrix from which to use dmatrix (patsy)
                   group_only_regressors = True,
                   group_only = ['z'],
                   fixed_at_default = ['t'],
                   p_outlier = 0.0,
                   outlier_max_t = 10.0):

    """Flexible simulator that allows specification of models very similar to the hddm model classes.

    :Arguments:
        n_subjects: int <default=5>
            Number of subjects in the datasets
        n_samples_by_subject: int <default=500>
            Number of trials for each subject
        model: str <default = 'ddm_vanilla'>
            Model to sample from. For traditional hddm supported models, append '_vanilla' to the model. Omitting 'vanilla' 
            imposes constraints on the parameter sets to not violate the trained parameter space of our LANs.
        conditions: dict <default={'c_one': ['high', 'low'], 'c_two': ['high', 'low'], 'c_three': ['high', 'medium', 'low']}>
            Keys represent condition relevant columns, and values are lists of unique items for each condition relevant column.
        depends_on: dict <default={'v': ['c_one', 'c_two']}>
            Keys specify model parameters that depend on the values --> lists of condition relevant columns.
        regression_models: list or strings <default = ['z ~ covariate_name']>
            Specify regression model formulas for one or more dependent parameters in a list.
        regression_covariates: dict <default={'covariate_name': {'type': 'categorical', 'range': (0, 4)}}>
            Dictionary in dictionary. Specify the name of the covariate column as keys, and for each key supply the 'type' (categorical, continuous) and 
            'range' ((lower bound, upper bound)) of the covariate.
        group_only_regressors: bin <default=True>
            Should regressors only be specified at the group level? If true then only intercepts are specified subject wise. 
            Other covariates act globally.
        group_only: list <default = ['z']>
            List of parameters that are specified only at the group level.
        fixed_at_default: list <default=['t']>
            List of parameters for which defaults are to be used. These defaults are specified in the model_config dictionary, which you can access via: hddm.simulators.model_config.
        p_outlier: float <default = 0.0>
            Specifies the proportion of outliers in the data.
        outlier_max_t: float <default = 10.0>
            Outliers are generated from np.random.uniform(low = 0, high = outlier_max_t) with random choices.
    Returns: 
        (pandas.DataFrame, dict): The Dataframe holds the generated dataset, ready for constuction of an hddm model. The dictionary holds the groundtruth parameter (values) and parameter names (keys). Keys match 
                                  the names of traces when fitting the equivalent hddm model. The parameter dictionary is useful for some graphs, otherwise not neccessary.
    """

    def check_params(data = None, model = None, is_nn = True):
        """
            Function checks if parameters are within legal bounds
        """
        for key in data.keys():
            if key in model_config[model]['params']:
                if np.sum(data[key] < model_config[model]['param_bounds'][0][model_config[model]['params'].index(key)]) > 0:
                    return 0
                elif np.sum(data[key] > model_config[model]['param_bounds'][1][model_config[model]['params'].index(key)]) > 0:
                    return 0
        return 1

    def get_parameter_remainder(regression_models = None, 
                        group_only = None, 
                        depends_on = None, 
                        fixed_at_default = None):
        
        """
            The arguments supplied to the simulator implicitly specify how we should handle a bunch of model parameters.
            If there remain model parameters that did not receive implicit instructions, we call these 'remainder' parameters
            and sample them randomly for our simulations. 
        """
        
        # Add subject parameters to full_parameter_dict
        total_param_list = model_config[model]['params']
        params_utilized = []

        # Regression Part
        #reg_df = make_covariate_df(regression_covariates, n_samples_by_subject)
        if regression_models is not None:
            for regression_model in regression_models:
                separator = regression_model.find('~')
                assert separator != -1, 'No outcome variable specified.'
                params_utilized += regression_model[:separator].strip(' ')

        # Group only Part
        if group_only is not None:
            params_utilized += group_only
        
        # Fixed Part
        if fixed_at_default is not None:
            params_utilized += fixed_at_default
        
        # Depends on Part
        if depends_on is not None:
            for depends_on_key in depends_on.keys():
                params_utilized += [depends_on_key]

        params_utilized = list(set(params_utilized))

        # Rest of Params
        #print(total_param_list)
        #print(params_utilized)
        remainder = set(total_param_list) - set(params_utilized)
        
        return remainder

    def make_covariate_df(regression_covariates, 
                          n_samples_by_subject):
        """ 
            Goes through the supplied covariate data, and turns it into a dataframe, with randomly generated covariate values.
            Each column refers to one covariate.
        """

        cov_df = pd.DataFrame(np.zeros((n_samples_by_subject, len(list(regression_covariates.keys())))), columns = [key for key in regression_covariates.keys()])
        #print(cov_df)
        for covariate in regression_covariates.keys():
            tmp = regression_covariates[covariate]
            if tmp['type'] == 'categorical':
                #print(cov_df)
                #print(covariate)
                #print(np.random.choice(np.arange(tmp['range'][0], tmp['range'][1], 1), replace =  True, size = n_samples_by_subject))
                cov_df[covariate] = np.random.choice(np.arange(tmp['range'][0], tmp['range'][1] + 1, 1), replace =  True, size = n_samples_by_subject) / (tmp['range'][1])
            else:
                cov_df[covariate] = np.random.uniform(low = tmp['range'][0], high = tmp['range'][1], size = n_samples_by_subject) / (tmp['range'][1])
        
        return cov_df
    
    def make_conditions_df(conditions = None):
        """ 
            Makes a dataframe out of the supplied condition dictionary, that stores each combination as a row.
        """
        arg_tuple = tuple([conditions[key] for key in conditions.keys()])
        condition_rows = np.meshgrid(*arg_tuple)
        return pd.DataFrame(np.column_stack([x_tmp.flatten() for x_tmp in condition_rows]), columns = [key for key in conditions.keys()])

    def make_single_sub_cond_df(conditions_df,
                                depends_on, 
                                regression_models, 
                                regression_covariates, 
                                group_only_regressors, 
                                group_only,
                                fixed_at_default, 
                                remainder,
                                model, 
                                group_level_parameter_dict,
                                n_subjects,
                                n_samples_by_subject):

        # Construct subject data
        full_parameter_dict = group_level_parameter_dict.copy()
  
        # Subject part -----------------------
        full_data = []
        # Condition --------------------------
        if conditions_df is None:
            n_conditions = 1
        else:
            n_conditions = conditions_df.shape[0]
        
        for condition_id in range(n_conditions):
            remainder_set = 0
            regressor_set = 0
            
            for subj_idx in range(n_subjects):
                # Parameter vector
                subj_data = pd.DataFrame(index = np.arange(0, n_samples_by_subject, 1))
                subj_data['subj_idx'] = str(subj_idx)

                # Fixed part
                if fixed_at_default is not None:
                    for fixed_tmp in fixed_at_default:
                        subj_data[fixed_tmp] = group_level_parameter_dict[fixed_tmp]

                # Group only part
                if group_only is not None:
                    for group_only_tmp in group_only:
                        if group_only_tmp in list(depends_on.keys()):
                            pass
                        else:
                            subj_data[group_only_tmp] = group_level_parameter_dict[group_only_tmp]
                    
                # Remainder part
                if remainder is not None:
                    for remainder_tmp in remainder:
                        if not remainder_set:
                            tmp_mean = group_level_parameter_dict[remainder_tmp]
                            tmp_std = group_level_parameter_dict[remainder_tmp + '_std']
                            full_parameter_dict[remainder_tmp + '_subj.' + str(subj_idx)] = np.random.normal(loc = tmp_mean, scale = tmp_std)
                            subj_data[remainder_tmp] = full_parameter_dict[remainder_tmp + '_subj.' + str(subj_idx)]
                        
                        # AF-TODO: IS THIS NECESSARY?
                        if remainder_set:
                            subj_data[remainder_tmp] = full_parameter_dict[remainder_tmp + '_subj.' + str(subj_idx)]

                # Depends on part
                if depends_on is not None:
                    # conditions_tmp = conditions_df.iloc[condition_id]
                    for depends_tmp in depends_on.keys():
                        conditions_df_tmp = conditions_df[depends_on[depends_tmp]].iloc[condition_id]
                        condition_elem = '.'.join(conditions_df_tmp)
                
                        if depends_tmp not in group_only:
                            tmp_mean = group_level_parameter_dict[depends_tmp + '(' + condition_elem + ')']
                            tmp_std = group_level_parameter_dict[depends_tmp + '_std']

                            full_parameter_dict[depends_tmp + '_subj(' + condition_elem + ').' + str(subj_idx)] = np.random.normal(loc = tmp_mean, scale = tmp_std)
                            subj_data[depends_tmp] = full_parameter_dict[depends_tmp + '_subj(' + condition_elem + ').' + str(subj_idx)]
                        else:
                            subj_data[depends_tmp] =  full_parameter_dict[depends_tmp + '(' + condition_elem + ')']

                        for condition_key_tmp in conditions_df_tmp.keys():  
                            subj_data[condition_key_tmp] = conditions_df_tmp[condition_key_tmp] ##############################################################

                # Regressor part
                if regression_covariates is not None:
                    cov_df = make_covariate_df(regression_covariates, n_samples_by_subject)
                
                    # Add cov_df to subject data
                    for key_tmp in cov_df.keys():
                        subj_data[key_tmp] = cov_df[key_tmp].copy()
                
                if regression_models is not None:
                    for reg_model in regression_models:
                        separator = reg_model.find('~')
                        outcome = reg_model[:separator].strip(' ')
                        reg_model_stripped = reg_model[(separator + 1):]
                        design_matrix = dmatrix(reg_model_stripped, cov_df)
                        
                        reg_params_tmp = []
                        reg_param_names_tmp = []
                        
                        for reg_param_key in group_level_parameter_dict[outcome + '_reg'].keys():
                            if (group_only_regressors and 'Intercept' in reg_param_key) or (not group_only_regressors):
                                reg_params_tmp.append(np.random.normal(loc = group_level_parameter_dict[outcome + '_reg'][reg_param_key], 
                                                                       scale = group_level_parameter_dict[outcome + '_reg_std'][reg_param_key + '_std']))

                                reg_param_names_tmp.append(reg_param_key + '_subj.' + str(subj_idx)) #########################################################
                            else: 
                                reg_params_tmp.append(group_level_parameter_dict[outcome + '_reg'][reg_param_key])
                                reg_param_names_tmp.append(reg_param_key)

                        reg_params_tmp = np.array(reg_params_tmp)

                        for key in group_level_parameter_dict[outcome + '_reg'].keys():
                            full_parameter_dict[key] = group_level_parameter_dict[outcome + '_reg'][key]
                        for key in group_level_parameter_dict[outcome + '_reg_std'].keys():
                            full_parameter_dict[key] = group_level_parameter_dict[outcome + '_reg_std'][key]
                        
                        if not regressor_set:
                            for k in range(len(reg_param_names_tmp)):
                                full_parameter_dict[reg_param_names_tmp[k]] = reg_params_tmp[k]

                        subj_data[outcome] = (design_matrix * reg_params_tmp).sum(axis = 1)
                        
                # Append full data:
                full_data.append(subj_data.copy())
                # print(full_data)
                 
            remainder_set = 1
            regressor_set = 1

        full_data = pd.concat(full_data)
        parameters = full_data[model_config[model]['params']]

        # Run the actual simulations
        sim_data = hddm.simulators.simulator(theta = parameters.values,
                                             model = model,
                                             n_samples = 1,
                                             delta_t = 0.001, 
                                             max_t = 20,
                                             no_noise = False,
                                             bin_dim = None,
                                             bin_pointwise = False)

        # Post-processing
        full_data['rt'] = sim_data[0].astype(np.float64)
        full_data['response'] = sim_data[1].astype(np.float64)
        full_data.loc[full_data['response'] < 0, ['response']] = 0.0

        # Add in outliers
        if p_outlier > 0:
            outlier_idx = np.random.choice(list(data.index), replace = False, size = int(p_outlier * len(list(data.index))))
            outlier_data = np.zeros((outlier_idx.shape[0], 2))
            
            # Outlier rts
            outlier_data[:, 0] = np.random.uniform(low = 0.0, high = outlier_max_t, size = outlier_data.shape[0])

            # Outlier choices
            outlier_data[:, 1] = np.random.choice(sim_data[2]['possible_choices'], size = outlier_data.shape[0])

            # Exchange data for outliers
            full_data.iloc[outlier_idx, [list(full_data.keys()).index('rt'), list(full_data.keys()).index('response')]] = outlier_data
        
            # Identify outliers in dataframe
            full_data['outlier']  = 0
            full_data[outlier_idx, [list(full_data.keys()).index('outlier')]] = 1

        full_data_cols = ['rt', 'response', 'subj_idx']
        
        if regression_covariates is not None:
            full_data_cols += [key for key in regression_covariates.keys()]
        if conditions is not None:
            full_data_cols += [key for key in conditions.keys()]

        full_data_cols += model_config[model]['params']
        full_data = full_data[full_data_cols]
        full_data.reset_index(drop = True, inplace = True)
        return full_data, full_parameter_dict

    def make_group_level_params(conditions_df,
                                group_only, 
                                depends_on, 
                                model, 
                                fixed_at_default, 
                                remainder, 
                                group_only_regressors,
                                regression_models, 
                                regression_covariates):
        """ 
            Make group level parameters from the information supplied.
        """

        group_level_parameter_dict = {}

        # Fixed part
        if fixed_at_default is not None:
            for fixed_tmp in fixed_at_default:
                group_level_parameter_dict[fixed_tmp] = np.random.uniform(low = model_config[model]['param_bounds'][0][model_config[model]['params'].index(fixed_tmp)],
                                                                   high = model_config[model]['param_bounds'][1][model_config[model]['params'].index(fixed_tmp)])
        # Group only part (excluding depends on)
        if (len(group_only) > 0):
            for group_only_tmp in group_only:
                if group_only_tmp in list(depends_on.keys()):
                    pass
                else:
                    group_level_parameter_dict[group_only_tmp] = np.random.uniform(low = model_config[model]['param_bounds'][0][model_config[model]['params'].index(group_only_tmp)],
                                                                            high = model_config[model]['param_bounds'][1][model_config[model]['params'].index(group_only_tmp)])
        # Remainder part
        if remainder is not None:
            for remainder_tmp in remainder:
                group_level_parameter_dict[remainder_tmp] = np.random.uniform(low = model_config[model]['param_bounds'][0][model_config[model]['params'].index(remainder_tmp)],
                                                                       high = model_config[model]['param_bounds'][1][model_config[model]['params'].index(remainder_tmp)])
                group_level_parameter_dict[remainder_tmp + '_std'] = np.random.uniform(low = 0, 
                                                                              high = (1 / 10) * (model_config[model]['param_bounds'][1][model_config[model]['params'].index(remainder_tmp)] - model_config[model]['param_bounds'][0][model_config[model]['params'].index(remainder_tmp)]))
        
        # Depends on part
        if depends_on is not None:
            for depends_tmp in depends_on.keys():
                conditions_df_tmp = conditions_df[depends_on[depends_tmp]]

                # Get unique elements:
                unique_elems = []
                for i in range(conditions_df_tmp.shape[0]):
                    unique_elems.append('.'.join(conditions_df_tmp.iloc[i]))
                unique_elems = np.unique(np.array(unique_elems))

                for unique_elem in unique_elems:
                    group_level_parameter_dict[depends_tmp + '(' + unique_elem + ')'] = np.random.uniform(low = model_config[model]['param_bounds'][0][model_config[model]['params'].index(depends_tmp)],
                                                                                                   high = model_config[model]['param_bounds'][1][model_config[model]['params'].index(depends_tmp)])
                
                if depends_tmp not in group_only:
                    bound_to_bound_tmp = (model_config[model]['param_bounds'][1][model_config[model]['params'].index(depends_tmp)] - model_config[model]['param_bounds'][0][model_config[model]['params'].index(depends_tmp)])
                    group_level_parameter_dict[depends_tmp + '_std'] = np.random.uniform(low = 0,
                                                                                         high = (1 / 10) * bound_to_bound_tmp)
            
        # Regressor part
        if regression_covariates is not None:
            cov_df = make_covariate_df(regression_covariates, n_samples_by_subject)

        if regression_models is not None:
            for reg_model in regression_models:
                separator = reg_model.find('~')
                outcome = reg_model[:separator].strip(' ')
                reg_model_stripped = reg_model[(separator + 1):]
                covariate_names = dmatrix(reg_model_stripped, cov_df).design_info.column_names

                reg_trace_dict = OrderedDict()
                reg_std_trace_dict = OrderedDict()
                
                for covariate in covariate_names:
                    if 'Intercept' in covariate:
                        bound_to_bound_tmp = model_config[model]['param_bounds'][1][model_config[model]['params'].index(outcome)] - model_config[model]['param_bounds'][0][model_config[model]['params'].index(outcome)]

                        reg_trace_dict[outcome + '_' + covariate] = np.random.uniform(low =  model_config[model]['param_bounds'][0][model_config[model]['params'].index(outcome)] + 0.3 *  bound_to_bound_tmp,
                                                                                      high = model_config[model]['param_bounds'][0][model_config[model]['params'].index(outcome)] + 0.7 *  bound_to_bound_tmp)
                        print(reg_trace_dict[outcome + '_' + covariate])

                        # Intercept is always fit subject wise
                        reg_std_trace_dict[outcome + '_' + covariate + '_' + 'std'] = np.random.uniform(low =  0,
                                                                                                        high = bound_to_bound_tmp / 10)

                    else:
                        reg_trace_dict[outcome + '_' + covariate] = np.random.uniform(low = -.1,
                                                                                      high = .1)
                        if not group_only_regressors:
                            reg_std_trace_dict[outcome + '_' + covariate + '_' + 'std'] = np.random.uniform(low =  0,
                                                                                                            high = bound_to_bound_tmp / 10)
                
                group_level_parameter_dict[outcome + '_reg'] = reg_trace_dict.copy()
                
                #if not group_only_regressors:
                group_level_parameter_dict[outcome + '_reg' + '_std'] = reg_std_trace_dict.copy()

        return group_level_parameter_dict

    # MAIN PART OF THE FUNCTION

    # Some checks
    if group_only is None:
        group_only = []

    # Specify 'remainder' parameters --> will be sampled randomly from the allowed range
    remainder = get_parameter_remainder(regression_models = regression_models,
                                        group_only = group_only,
                                        depends_on = depends_on,
                                        fixed_at_default = fixed_at_default)
    print(fixed_at_default)
    print(remainder)

    # Make conditions df
    if depends_on is not None:
        conditions_df = make_conditions_df(conditions = conditions)
        print('Conditions created...')
        print(conditions_df)
    else:
        conditions_df = None
    
    params_ok_all = 0
    cnt = 0   
    while params_ok_all == 0:
        if cnt > 0:
            print('new round of data simulation because parameter bounds where violated')

        group_level_param_dict = make_group_level_params(conditions_df = conditions_df,
                                                         group_only = group_only,
                                                         depends_on = depends_on,
                                                         model = model,
                                                         fixed_at_default = fixed_at_default,
                                                         remainder = remainder,
                                                         group_only_regressors = group_only_regressors,
                                                         regression_models = regression_models,
                                                         regression_covariates = regression_covariates)

        data, full_parameter_dict = make_single_sub_cond_df(conditions_df = conditions_df,
                                                            group_only = group_only,
                                                            depends_on = depends_on,
                                                            model = model,
                                                            fixed_at_default = fixed_at_default,
                                                            remainder = remainder,
                                                            regression_models = regression_models,
                                                            regression_covariates = regression_covariates,
                                                            group_only_regressors = group_only_regressors,
                                                            group_level_parameter_dict = group_level_param_dict,
                                                            n_samples_by_subject = n_samples_by_subject,
                                                            n_subjects = n_subjects)
        
        params_ok_all = check_params(data = data, model = model)
        cnt += 1

    return data, full_parameter_dict