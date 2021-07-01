from tensorflow import keras
import hddm.keras_models

def load_mlp(model = 'ddm'):
    """Loads the MLP for a specified generative model

    :Arguments:
        model: str <default='ddm'>
            String that specifies the model for which we should load the MLP network.

    Returns:
        keras.model: Returns a pretrained keras model which we can use to get pointwise log-likelihoods for the model supplied in the model string.
    """
    if model == 'ddm':
        return keras.models.load_model(hddm.keras_models.__path__[0] + '/model_final_ddm.h5', compile = False)
    if model == 'ddm_analytic':
        return keras.models.load_model(hddm.keras_models.__path__[0] + '/model_final_ddm_analytic.h5', compile = False)
    
    if model == 'weibull_cdf':
        return keras.models.load_model(hddm.keras_models.__path__[0] + '/model_final_new.h5', compile = False)
    
    if model == 'angle':
        return keras.models.load_model(hddm.keras_models.__path__[0] + '/model_final_angle.h5', compile = False)
    
    if model == 'ornstein':
        return keras.models.load_model(hddm.keras_models.__path__[0] + '/model_final_ornstein.h5', compile = False)

    if model == 'levy':
        return keras.models.load_model(hddm.keras_models.__path__[0] + '/model_final_levy.h5', compile = False)
    
    if model == 'full_ddm' or model == 'full_ddm2':
        return keras.models.load_model(hddm.keras_models.__path__[0] + '/model_final_full_ddm.h5', compile = False)
    
    if model == 'ddm_sdv':
        return keras.models.load_model(hddm.keras_models.__path__[0] + '/model_final_ddm_sdv.h5', compile = False)
    
    if model == 'ddm_sdv_analytic':
        return keras.models.load_model(hddm.keras_models.__path__[0] + '/model_final_ddm_sdv_analytic.h5', compile = False)

    if model == 'ddm_sdv_analytic:':
        return keras.models.load_model(hddm.keras_models.__path__[0] + '/model_final_ddm_sdv_analytic.h5', compile = False)
    
    else:
        return 'Model is not known'