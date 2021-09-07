# Add all the torch configs here
import hddm
import pickle
import os

class TorchConfig(object):
    def __init__(self, model = None):
        self.network_files = {"ddm": "db415ca6008311ec8d90a0423f3e9b42_ddm_torch_state_dict.pt"}
        self.network_config_files = {"ddm": "db415ca6008311ec8d90a0423f3e9b42_ddm_torch__network_config.pickle"}
        self.network_config = self.get_network_config(file_name = self.network_config_files[model])
        self.network_path = os.path.join(hddm.__path__[0], "torch_models", network_files[model])

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