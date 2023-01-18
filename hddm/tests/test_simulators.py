import unittest
import numpy as np
import pandas as pd
import hddm

# Help on function simulator_h_c in module hddm.simulators.hddm_dataset_generators:

# simulator_h_c(data=None, n_subjects=10, n_trials_per_subject=100, model='ddm_hddm_base', conditions={'c_one': ['high', 'low'], 'c_two': ['high', 'low'], 'c_three': ['high', 'medium', 'low']}, depends_on={'v': ['c_one', 'c_two']}, regression_models=['z ~ covariate_name'], regression_covariates={'covariate_name': {'type': 'categorical', 'range': (0, 4)}}, group_only_regressors=True, group_only=['z'], fixed_at_default=['t'], p_outlier=0.0, outlier_max_t=10.0, **kwargs)
#     Flexible simulator that allows specification of models very similar to the hddm model classes.

#     :Arguments:
#         data: pd.DataFrame <default=None>
#             Actual covariate dataset. If data is supplied its covariates are used instead of generated.
#         n_subjects: int <default=5>
#             Number of subjects in the datasets
#         n_trials_per_subject: int <default=500>
#             Number of trials for each subject
#         model: str <default = 'ddm_hddm_base'>
#             Model to sample from. For traditional hddm supported models, append '_hddm_base' to the model. Omitting 'hddm_base'
#             imposes constraints on the parameter sets to not violate the trained parameter space of our LANs.
#         conditions: dict <default={'c_one': ['high', 'low'], 'c_two': ['high', 'low'], 'c_three': ['high', 'medium', 'low']}>
#             Keys represent condition relevant columns, and values are lists of unique items for each condition relevant column.
#         depends_on: dict <default={'v': ['c_one', 'c_two']}>
#             Keys specify model parameters that depend on the values --> lists of condition relevant columns.
#         regression_models: list or strings <default = ['z ~ covariate_name']>
#             Specify regression model formulas for one or more dependent parameters in a list.
#         regression_covariates: dict <default={'covariate_name': {'type': 'categorical', 'range': (0, 4)}}>
#             Dictionary in dictionary. Specify the name of the covariate column as keys, and for each key supply the 'type' (categorical, continuous) and
#             'range' ((lower bound, upper bound)) of the covariate.
#         group_only_regressors: bin <default=True>
#             Should regressors only be specified at the group level? If true then only intercepts are specified subject wise.
#             Other covariates act globally.
#         group_only: list <default = ['z']>
#             List of parameters that are specified only at the group level.
#         fixed_at_default: list <default=['t']>
#             List of parameters for which defaults are to be used. These defaults are specified in the model_config dictionary, which you can access via: hddm.simulators.model_config.
#         p_outlier: float <default = 0.0>
#             Specifies the proportion of outliers in the data.
#         outlier_max_t: float <default = 10.0>
#             Outliers are generated from np.random.uniform(low = 0, high = outlier_max_t) with random choices.
#     Returns:
#         (pandas.DataFrame, dict): The Dataframe holds the generated dataset, ready for constuction of an hddm model. The dictionary holds the groundtruth parameter (values) and parameter names (keys). Keys match
#                                   the names of traces when fitting the equivalent hddm model. The parameter dictionary is useful for some graphs, otherwise not neccessary.


class SimulatorTests(unittest.TestCase):
    def setUp(self):
        print("Passing setUp")
        self.models = list(
            hddm.torch.torch_config.TorchConfig(model="ddm").network_files.keys()
        ) + ["ddm_hddm_base", "full_ddm_hddm_base"]
        self.n_samples_per_subject = 100
        self.n_trials = 10

        self.n_subjects = 5
        self.cav_data = hddm.load_csv(
            hddm.__path__[0] + "/examples/cavanagh_theta_nn.csv"
        )
        pass

    def tearDown(self):
        print("Passing tearDown")
        pass

    def test_basic_simulator(self):
        print("Testing basic simulators")
        for model in self.models:
            print("Now testing model: ", model)
            out = hddm.simulators.simulator(
                theta=np.tile(
                    hddm.model_config.model_config[model]["params_default"],
                    reps=(self.n_trials, 1),
                ),
                model=model,
                n_samples=self.n_samples_per_subject,
            )

            print("Testing output shapes")
            self.assertEqual(out[0].shape, out[1].shape)
            self.assertEqual(
                out[0].shape, (self.n_samples_per_subject, self.n_trials, 1)
            )
            self.assertEqual(type(out[2]), dict)

            print("Check if all parameters are in output dictionary of simulator")
            for param in hddm.model_config.model_config[model]["params"]:
                self.assertTrue(param in out[2].keys())

            # Potentially add some simulator behavior tests

    def test_simulator_h_c_depends(self):
        # print(hddm.__path__ + '/examples/cavanagh_theta_nn.csv')

        # Sample some metaparameters here

        for model in self.models:
            print("Testing simulator_h_c() for model: ", model)
            depends_param_id = np.random.choice(
                len(hddm.model_config.model_config[model]["params"])
            )
            depends_param_tmp = hddm.model_config.model_config[model]["params"][
                depends_param_id
            ]
            depends_on = {depends_param_tmp: ["stim"]}
            print("depends_on : ", depends_on)
            conditions = {"stim": list(self.cav_data["stim"].unique())}

            fixed_at_default_id = depends_param_id
            while fixed_at_default_id == depends_param_id:
                fixed_at_default_id = np.random.choice(
                    len(hddm.model_config.model_config[model]["params"])
                )
                fixed_at_default_tmp = [
                    hddm.model_config.model_config[model]["params"][fixed_at_default_id]
                ]

            print("fixed_at_default_tmp: ", fixed_at_default_tmp)

            print("Testing simulator_h_c() with depends_on on: Cavanagh Data")
            (
                data,
                full_parameter_dict,
            ) = hddm.simulators.hddm_dataset_generators.simulator_h_c(
                data=self.cav_data.copy(),
                model=model,
                p_outlier=0.00,
                conditions=None,
                depends_on=depends_on,
                regression_models=None,
                regression_covariates=None,
                group_only_regressors=False,
                group_only=None,  # ['vh', 'vl1', 'vl2', 'a', 't'],
                fixed_at_default=fixed_at_default_tmp,
            )  # ['z'])
            self.assertTrue(np.unique(data[fixed_at_default_tmp]).shape[0], 1)
            self.assertEqual(data.shape[0], self.cav_data.shape[0])

            print("Testing simulator_h_c() with depends_on on: Artificial Data")
            (
                data,
                full_parameter_dict,
            ) = hddm.simulators.hddm_dataset_generators.simulator_h_c(
                data=None,
                n_subjects=self.n_subjects,
                n_trials_per_subject=self.n_smples_per_subject,
                model=model,
                p_outlier=0.00,
                conditions=conditions,
                depends_on=depends_on,
                regression_models=None,
                regression_covariates=None,
                group_only_regressors=False,
                group_only=None,  # ['vh', 'vl1', 'vl2', 'a', 't'],
                fixed_at_default=fixed_at_default_tmp,
            )  # ['z'])
            self.assertTrue(np.unique(data[fixed_at_default_tmp]).shape[0], 1)


if __name__ == "__main__":
    unittest.main()

# Arrange


# Act


# Assert

# Mocks, Fakes, Stubs
