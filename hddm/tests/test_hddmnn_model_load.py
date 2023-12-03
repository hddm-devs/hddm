import unittest
import hddm
import os
import shutil


class ModelLoad(unittest.TestCase):
    def get_data_single_subj(self, model=None):
        (
            data,
            gt_parms__,
        ) = hddm.simulators.hddm_dataset_generators.simulator_single_subject(
            parameters=hddm.model_config.model_config[model]["params_default"],
            p_outlier=0.01,
            max_rt_outlier=10.0,
            model=model,
        )
        print(data)
        return data

    def setUp(self):
        self.models = list(
            hddm.torch.torch_config.TorchConfig(model="ddm").network_files.keys()
        )
        self.n_samples_per_trial = 1000
        self.nmcmc = 100
        self.nburn = 50
        self.filepath = hddm.__path__[0] + "/tests/torch_models/"
        self.cav_data = hddm.load_csv(
            hddm.__path__[0] + "/examples/cavanagh_theta_nn.csv"
        )

        # Make model test folder if it is not there
        if not os.path.isdir(self.filepath):
            os.mkdir(self.filepath)
            pass

    def tearDown(self):
        shutil.rmtree(hddm.__path__[0] + "/tests/torch_models")
        pass

    def test_init_sample_save_load_single_subj(self):
        for model in self.models:
            # Get simulations
            data = self.get_data_single_subj(model=model)

            # Initialize HDDMnn Model
            print("Loading Model: " + model)
            model_ = hddm.HDDMnn(
                data,
                model=model,
                informative=False,
                include=hddm.model_config.model_config[model]["hddm_include"],
                is_group_model=False,
                depends_on={},
                p_outlier=0.00,
            )

            # Sample
            print("Sampling: ")
            model_.sample(
                self.nmcmc,
                burn=self.nburn,
                dbname=self.filepath + "test_" + model + ".db",
                db="pickle",
            )

            # Save Model
            print("Saving Model: ")
            model_.save(self.filepath + "test_" + model + ".pickle")

            # Load Model
            print("Loading Model: ")
            model__ = hddm.load(self.filepath + "test_" + model + ".pickle")
            self.assertTrue(model__.nn == True)

            del model_
            del model__

    def test_init_sample_save_load_stimcoding(self):
        for model in self.models:
            if len(hddm.model_config.model_config[model]["choices"]) == 2:
                # Generate Data
                data, gt = hddm.simulators.hddm_dataset_generators.simulator_stimcoding(
                    model=model,
                    n_samples_by_condition=self.n_samples_per_trial,
                    split_by="v",
                )
                # Initialize HDDM Model
                model_ = hddm.HDDMnnStimCoding(
                    data,
                    model=model,
                    split_param="v",
                    drift_criterion=True,
                    stim_col="stim",
                )
                # Sample

                # Sample
                model_.sample(
                    self.nmcmc,
                    burn=self.nburn,
                    dbname=self.filepath + "test_" + model + ".db",
                    db="pickle",
                )

                # Save Model
                print("Saving Model: ")
                model_.save(self.filepath + "test_" + model + ".pickle")

                # Load Model
                print("Loading Model: ")
                model__ = hddm.load(self.filepath + "test_" + model + ".pickle")
                self.assertTrue(model__.nn == True)

                del model_
                del model__

            else:
                print("Skipping n > 2 choice models for this test for now !")
        pass

        pass

    def test_init_sample_save_load_regressor(self):
        for model in self.models:
            if len(hddm.model_config.model_config[model]["choices"]) == 2:
                # Define Link Function
                def id_link(x):
                    return x

                # Define Regression Model
                v_reg = {"model": "v ~ 1 + theta", "link_func": id_link}

                # Initialize HDDM Model (using cavanagh data)
                model_ = hddm.HDDMnnRegressor(
                    self.cav_data,
                    [v_reg],
                    include=hddm.model_config.model_config[model]["hddm_include"],
                    model=model,
                    group_only_regressors=True,
                )

                # Sample
                model_.sample(
                    self.nmcmc,
                    burn=self.nburn,
                    dbname=self.filepath + "test_" + model + ".db",
                    db="pickle",
                )

                # Save Model
                print("Saving Model: ")
                model_.save(self.filepath + "test_" + model + ".pickle")

                # Load Model
                print("Loading Model: ")
                model__ = hddm.load(self.filepath + "test_" + model + ".pickle")
                self.assertTrue(model__.nn == True)

                # Check if id_link is preserved correctly
                print("Checking if link func is correctly recovered")
                self.assertTrue(
                    model__.model_descrs[0]["model"]["link_func"] == id_link
                )

                del model_
                del model__

            else:
                print("Skipping n > 2 choice models for this test for now !")
        pass


if __name__ == "__main__":
    unittest.main()


# Arrange


# Act


# Assert
