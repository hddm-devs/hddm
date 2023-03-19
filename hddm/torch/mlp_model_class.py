try:
    import torch
    import torch.nn as nn
    import uuid

    # import torch.optim as optim
    # import torch.nn.functional as F
    # from torch import optim

    class TorchMLP(nn.Module):
        def __init__(
            self,
            network_config=None,
            input_shape=10,
            save_folder=None,
            generative_model_id="ddm",
        ):
            super(TorchMLP, self).__init__()
            if generative_model_id is not None:
                self.model_id = uuid.uuid1().hex + "_" + generative_model_id
            else:
                self.model_id = None

            self.save_folder = save_folder
            self.input_shape = input_shape
            self.network_config = network_config
            self.activations = {"relu": torch.nn.ReLU(), "tanh": torch.nn.Tanh()}
            self.layers = nn.ModuleList()

            self.layers.append(
                nn.Linear(input_shape, self.network_config["layer_sizes"][0])
            )
            self.layers.append(self.activations[self.network_config["activations"][0]])
            # print(self.network_config['activations'][0])
            for i in range(len(self.network_config["layer_sizes"]) - 1):
                self.layers.append(
                    nn.Linear(
                        self.network_config["layer_sizes"][i],
                        self.network_config["layer_sizes"][i + 1],
                    )
                )
                # print(self.network_config['activations'][i + 1])
                if i < (len(self.network_config["layer_sizes"]) - 2):
                    self.layers.append(
                        self.activations[self.network_config["activations"][i + 1]]
                    )
                else:
                    # skip last activation since
                    pass
            self.len_layers = len(self.layers)

        def forward(self, x):
            for i in range(self.len_layers - 1):
                x = self.layers[i](x)
            return self.layers[-1](x)

except:
    print(
        "Error loading pytorch capabilities. Neural network functionality cannot be used."
    )
