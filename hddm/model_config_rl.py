import numpy as np

model_config_rl = {
    "qlearn": {
        "doc": "Q-learning",
        "params": ["rl_alpha"],
        "params_trans": [0],
        "params_std_upper": [None],
        "param_bounds": [[0.0], [1.0]],
        "params_default": [0.5],
    },

    "qlearn_dual": {
        "doc": "Q-learning with two learning rates",
        "params": ["rl_alpha", "rl_pos_alpha"],
        "params_trans": [0, 0],
        "params_std_upper": [None, None],
        "param_bounds": [[0.0, 0.0], [1.0, 1.0]],
        "params_default": [0.5, 0.5],
    }
} 
