import unittest
import hddm
import os
import shutil
import numpy as np


class NetworkInspectorTest(unittest.TestCase):
    def setUp(self):
        self.models = list(
            hddm.torch.torch_config.TorchConfig(model="ddm").network_files.keys()
        )
        self.n_samples_per_trial = 1000
        self.nmcmc = 200
        self.nburn = 100
        self.filepath = hddm.__path__[0] + "/tests/torch_models/"
        self.cav_data = hddm.load_csv(
            hddm.__path__[0] + "/examples/cavanagh_theta_nn.csv"
        )

        # Make model test folder if it is not there
        if not os.path.isdir(self.filepath):
            os.mkdir(self.filepath)
            pass

    def tearDown(self):
        pass

    def test_caterpillar(self):
        for model in self.models:
            # Make Data
            print("Make Data")
            (
                data,
                gt_params,
            ) = hddm.simulators.hddm_dataset_generators.simulator_single_subject(
                parameters=hddm.model_config.model_config[model]["params_default"],
                n_samples=self.n_samples_per_trial,
                p_outlier=0.01,
                max_rt_outlier=10.0,
                model=model,
            )

            # Fit Model
            print("Fit Model")
            model_ = hddm.HDDMnn(
                data,
                model=model,
                informative=False,
                include=hddm.model_config.model_config[model]["hddm_include"],
                is_group_model=False,
                depends_on={},
                p_outlier=0.00,
            )

            model_.sample(self.nmcmc, burn=self.nburn)

            # Plot
            print("Caterpillar plot")
            hddm.plotting.caterpillar_plot(
                model_,
                ground_truth_parameter_dict=gt_params,
                drop_sd=True,
                keep_key=None,
                x_limits=[-2, 2],
                aspect_ratio=2,
                figure_scale=1.0,
                save=False,
                show=False,
                tick_label_size_x=22,
                tick_label_size_y=14,
            )


if __name__ == "__main__":
    unittest.main()
