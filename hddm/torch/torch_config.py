# Add all the torch configs here
import hddm
import pickle
import os


class TorchConfig(object):
    def __init__(self, model=None):
        self.network_files = {
            "ddm": "d27193a4153011ecb76ca0423f39a3e6_ddm_torch_state_dict.pt",
            # "angle": "eba53550128911ec9fef3cecef056d26_angle_torch_state_dict.pt",
            "angle": "248c94cca33e11ecb947ac1f6bfea5a4_training_data_angle_torch_state_dict.pt",
            "levy": "80dec298152e11ec88b8ac1f6bfea5a4_levy_torch_state_dict.pt",
            "ornstein": "1f496b50127211ecb6943cecef057438_ornstein_torch_state_dict.pt",
            "weibull": "44deb16a127f11eca325a0423f39b436_weibull_torch_state_dict.pt",
            "weibull_cdf": "44deb16a127f11eca325a0423f39b436_weibull_torch_state_dict.pt",
            "ddm_par2_no_bias": "495136c615d311ecb5fda0423f39a3e6_par2_no_bias_torch_state_dict.pt",
            "ddm_seq2_no_bias": "247aa76015d311ec922d3cecef057012_seq2_no_bias_torch_state_dict.pt",
            "ddm_mic2_adj_no_bias": "049de8c23cba11ecb507a0423f3e9a4a_ddm_mic2_adj_no_bias_torch_state_dict.pt",
            "ddm_par2_angle_no_bias": "c611f4ca15dd11ec89543cecef056d26_par2_angle_no_bias_torch_state_dict.pt",
            "ddm_seq2_angle_no_bias": "7821dc4c15f811eca1553cecef057012_seq2_angle_no_bias_torch_state_dict.pt",
            "ddm_mic2_adj_angle_no_bias": "122c7ca63cc411eca89b3cecef05595c_ddm_mic2_adj_angle_no_bias_torch_state_dict.pt",
            "ddm_par2_weibull_no_bias": "a5e6bbc2160f11ec88173cecef056d26_par2_weibull_no_bias_torch_state_dict.pt",
            "ddm_seq2_weibull_no_bias": "74ca5d6c161b11ec9ebb3cecef056d26_seq2_weibull_no_bias_torch_state_dict.pt",
            "ddm_mic2_adj_weibull_no_bias": "151a6c363ccc11ecbf89a0423f3e9b68_ddm_mic2_adj_weibull_no_bias_torch_state_dict.pt",
            "lca_no_bias_4": "0d9f0e94175b11eca9e93cecef057438_lca_no_bias_4_torch_state_dict.pt",
            "lca_no_bias_angle_4": "362f8656175911ecbe8c3cecef057438_lca_no_bias_angle_4_torch_state_dict.pt",
            "race_no_bias_4": "ff29c116173611ecbdba3cecef05595c_race_no_bias_4_torch_state_dict.pt",
            "race_no_bias_angle_4": "179c5a6e175111ec93f13cecef056d26_race_no_bias_angle_4_torch_state_dict.pt",
        }

        self.network_config_files = {
            "ddm": "d27193a4153011ecb76ca0423f39a3e6_ddm_torch__network_config.pickle",
            # "angle": "eba53550128911ec9fef3cecef056d26_angle_torch__network_config.pickle",
            "angle": "248c94cca33e11ecb947ac1f6bfea5a4_training_data_angle_torch__network_config.pickle",
            "levy": "80dec298152e11ec88b8ac1f6bfea5a4_levy_torch__network_config.pickle",
            "ornstein": "1f496b50127211ecb6943cecef057438_ornstein_torch__network_config.pickle",
            "weibull": "44deb16a127f11eca325a0423f39b436_weibull_torch__network_config.pickle",
            "weibull_cdf": "44deb16a127f11eca325a0423f39b436_weibull_torch_state_dict.pt",
            "ddm_par2_no_bias": "495136c615d311ecb5fda0423f39a3e6_par2_no_bias_torch__network_config.pickle",
            "ddm_seq2_no_bias": "247aa76015d311ec922d3cecef057012_seq2_no_bias_torch__network_config.pickle",
            "ddm_mic2_adj_no_bias": "049de8c23cba11ecb507a0423f3e9a4a_ddm_mic2_adj_no_bias_torch__network_config.pickle",
            "ddm_par2_angle_no_bias": "c611f4ca15dd11ec89543cecef056d26_par2_angle_no_bias_torch__network_config.pickle",
            "ddm_seq2_angle_no_bias": "7821dc4c15f811eca1553cecef057012_seq2_angle_no_bias_torch__network_config.pickle",
            "ddm_mic2_adj_angle_no_bias": "122c7ca63cc411eca89b3cecef05595c_ddm_mic2_adj_angle_no_bias_torch__network_config.pickle",
            "ddm_par2_weibull_no_bias": "a5e6bbc2160f11ec88173cecef056d26_par2_weibull_no_bias_torch__network_config.pickle",
            "ddm_seq2_weibull_no_bias": "74ca5d6c161b11ec9ebb3cecef056d26_seq2_weibull_no_bias_torch__network_config.pickle",
            "ddm_mic2_adj_weibull_no_bias": "151a6c363ccc11ecbf89a0423f3e9b68_ddm_mic2_adj_weibull_no_bias_torch__network_config.pickle",
            "lca_no_bias_4": "0d9f0e94175b11eca9e93cecef057438_lca_no_bias_4_torch__network_config.pickle",
            "lca_no_bias_angle_4": "362f8656175911ecbe8c3cecef057438_lca_no_bias_angle_4_torch__network_config.pickle",
            "race_no_bias_4": "ff29c116173611ecbdba3cecef05595c_race_no_bias_4_torch__network_config.pickle",
            "race_no_bias_angle_4": "179c5a6e175111ec93f13cecef056d26_race_no_bias_angle_4_torch__network_config.pickle",
        }

        self.network_config = self.get_network_config(
            file_name=self.network_config_files[model]
        )
        self.network_path = os.path.join(
            hddm.__path__[0], "torch_models", self.network_files[model]
        )

    def get_network_config(self, file_name=None):
        with open(os.path.join(hddm.__path__[0], "torch_models", file_name), "rb") as f:
            network_config = pickle.load(f)
        return network_config
