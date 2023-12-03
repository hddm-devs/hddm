import numpy as np

model_config_rl = {
    "RWupdate": {
        "doc": "Rescorla-Wagner update rule.",
        "params": ["rl_alpha"],
        "params_trans": [0],
        "params_std_upper": [10],
        "param_bounds": [[0.0], [1.0]],
        "params_default": [0.5],
    },
    "RWupdate_dual": {
        "doc": "Rescorla-Wagner update with two learning rates",
        "params": ["rl_alpha", "rl_pos_alpha"],
        "params_trans": [0, 0],
        "params_std_upper": [10, 10],
        "param_bounds": [[0.0, 0.0], [1.0, 1.0]],
        "params_default": [0.5, 0.5],
    },
}
