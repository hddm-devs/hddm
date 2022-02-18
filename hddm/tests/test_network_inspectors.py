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
        self.rt_range = (0, 20)
        self.rt_steps = 0.01

    def tearDown(self):
        pass

    def test_load_torch_model_and_run_prediction(self):
        for model in self.models:
            # Load network
            print("Load Network for model: ", model)
            torch_model = hddm.network_inspectors.load_torch_mlp(model=model)

            # Define model dependent data
            n_choices = len(hddm.model_config.model_config[model]["choices"])
            choice_options = hddm.model_config.model_config[model]["choices"]
            theta = hddm.model_config.model_config[model]["params_default"]

            rts = np.expand_dims(
                np.concatenate(
                    [
                        np.arange(self.rt_range[0], self.rt_range[1], self.rt_steps)
                        for i in range(n_choices)
                    ]
                ),
                axis=1,
            )
            choices = np.concatenate(
                [
                    [
                        c
                        for i in range(
                            int((self.rt_range[1] - self.rt_range[0]) / self.rt_steps)
                        )
                    ]
                    for c in choice_options
                ]
            )
            thetas = np.tile(np.array(theta), reps=(rts.shape[0], 1))
            tmp_data = np.column_stack([thetas, rts, choices]).astype(np.float32)

            # Predict likelihood from model
            print(torch_model.predict_on_batch(tmp_data).shape)
            pass


if __name__ == "__main__":
    unittest.main()
