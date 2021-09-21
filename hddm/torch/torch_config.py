# Add all the torch configs here
import hddm
import pickle
import os

class TorchConfig(object):
    def __init__(self, model = None):
        self.network_files = {"ddm": "d27193a4153011ecb76ca0423f39a3e6_ddm_torch_state_dict.pt",
                              "angle": "eba53550128911ec9fef3cecef056d26_angle_torch_state_dict.pt",
                              "levy": "80dec298152e11ec88b8ac1f6bfea5a4_levy_torch_state_dict.pt",
                              "ornstein": "1f496b50127211ecb6943cecef057438_ornstein_torch_state_dict.pt",
                              "weibull": "44deb16a127f11eca325a0423f39b436_weibull_torch_state_dict.pt",
                              "par2": "495136c615d311ecb5fda0423f39a3e6_par2_no_bias_torch_state_dict.pt",
                              "seq2": "247aa76015d311ec922d3cecef057012_seq2_no_bias_torch_state_dict.pt",
                              "mic2": "259784b0160011ec822da0423f3e9b4e_mic2_angle_no_bias_torch_state_dict.pt",
                              "par2_angle": "c611f4ca15dd11ec89543cecef056d26_par2_angle_no_bias_torch_state_dict.pt",
                              "seq2_angle": "7821dc4c15f811eca1553cecef057012_seq2_angle_no_bias_torch_state_dict.pt",
                              "mic2_angle": "259784b0160011ec822da0423f3e9b4e_mic2_angle_no_bias_torch_state_dict.pt",
                              "par2_weibull": "a5e6bbc2160f11ec88173cecef056d26_par2_weibull_no_bias_torch_state_dict.pt",
                              "seq2_weibull": "74ca5d6c161b11ec9ebb3cecef056d26_seq2_weibull_no_bias_torch_state_dict.pt",
                              "mic2_weibull": "4a420fec161911eca5d63cecef057012_mic2_weibull_no_bias_torch_state_dict.pt"}
        
        self.network_config_files = {"ddm": "d27193a4153011ecb76ca0423f39a3e6_ddm_torch__network_config.pickle",
                                     "angle": "eba53550128911ec9fef3cecef056d26_angle_torch__network_config.pickle",
                                     "levy": "80dec298152e11ec88b8ac1f6bfea5a4_levy_torch__network_config.pickle",
                                     "ornstein": "1f496b50127211ecb6943cecef057438_ornstein_torch__network_config.pickle",
                                     "weibull": "44deb16a127f11eca325a0423f39b436_weibull_torch__network_config.pickle",
                                     "par2": "495136c615d311ecb5fda0423f39a3e6_par2_no_bias_torch__network_config.pickle",
                                     "seq2": "247aa76015d311ec922d3cecef057012_seq2_no_bias_torch__network_config.pickle",
                                     "mic2": "259784b0160011ec822da0423f3e9b4e_mic2_angle_no_bias_torch__network_config.pickle",
                                     "par2_angle": "c611f4ca15dd11ec89543cecef056d26_par2_angle_no_bias_torch__network_config.pickle",
                                     "seq2_angle": "7821dc4c15f811eca1553cecef057012_seq2_angle_no_bias_torch__network_config.pickle",
                                     "mic2_angle": "259784b0160011ec822da0423f3e9b4e_mic2_angle_no_bias_torch__network_config.pickle",
                                     "par2_weibull": "a5e6bbc2160f11ec88173cecef056d26_par2_weibull_no_bias_torch__network_config.pickle", 
                                     "seq2_weibull": "74ca5d6c161b11ec9ebb3cecef056d26_seq2_weibull_no_bias_torch__network_config.pickle",
                                     "mic2_weibull": "4a420fec161911eca5d63cecef057012_mic2_weibull_no_bias_torch__network_config.pickle"}
        
        self.network_config = self.get_network_config(file_name = self.network_config_files[model])
        self.network_path = os.path.join(hddm.__path__[0], "torch_models", self.network_files[model])

    def get_network_config(self, file_name = None):
        return pickle.load(open(os.path.join(hddm.__path__[0], "torch_models", file_name), 'rb'))

#network_configs = 
# network_configs['ddm'] = {'layer_types': ['dense', 'dense', 'dense', 'dense'],
#                           'layer_sizes': [100, 100, 100, 1],
#                           'activations': ['tanh', 'tanh', 'tanh', 'linear'],
#                           'loss': 'huber',
#                           'callbacks': ['checkpoint', 'earlystopping', 'reducelr'],
#                           'model_id': 'ddm'}
#return network_configs[model]