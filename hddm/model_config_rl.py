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
    "RLWM_v1": {
        "doc": "RLWM model v1.",
        "params": ["rl_alpha", "rl_phi", "rl_rho"],
        "params_trans": [1, 1, 1],
        "params_std_upper": [10, 10, 10],
        "param_bounds": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        "params_default": [0.002, 0.2, 0.5], # for param_trans = 1
        #"params_default": [-8, -1, 0], # for param_trans = 0
        "hddm_include": ['rl_alpha', 'rl_phi', 'rl_rho'],
        # "slice_widths": {
        #     "rl_alpha": 0.2,
        #     "rl_alpha_std": 0.2,
        #     "rl_phi": 0.2,
        #     "rl_phi_std": 0.2,
        #     "rl_rho": 0.2,
        #     "rl_rho_std": 0.2,
            
        # },
    },
    "RLWM_v2": {
        "doc": "RLWM model v2.",
        "params": ["rl_alpha", "rl_phi", "rl_rho", "rl_gamma", "rl_epsilon"],
        "params_trans": [1, 1, 1, 1],
        "params_std_upper": [10, 10, 10, 10],
        "param_bounds": [[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]],
        "params_default": [0.05, 0.2, 0.5, 0.8, 0.02],
        "hddm_include": ['rl_alpha', 'rl_phi', 'rl_rho', 'rl_gamma', 'rl_epsilon'],
        "slice_widths": {
            "rl_alpha": 0.2,
            "rl_alpha_std": 0.2,
            "rl_phi": 0.2,
            "rl_phi_std": 0.2,
            "rl_rho": 0.2,
            "rl_rho_std": 0.2,
            "rl_gamma": 0.2,
            "rl_epsilon": 0.2,
        },
    },
}

'''
"RLWM": {
        "doc": "RLWM model.",
        "params": ["v", "rl_alpha", "rl_phi", "rl_rho"],
        "params_trans": [0, 0, 0, 0],
        "params_std_upper": [2, 10, 10, 10],
        "param_bounds": [[1.0, 0.0, 0.0, 0.0], [3.0, 1.0, 1.0, 1.0]],
        "params_default": [1.0, -5.325, -2.56, 0],
        "hddm_include": ['v', 'rl_alpha', 'rl_phi', 'rl_rho'],
        "slice_widths": {
            "v": 1,
            "v_std": 1,
            "a": 1,
            "a_std": 1,
            "t": 0.01,
            "t_std": 0.15,
            "rl_alpha": 0.2,
            "rl_alpha_std": 0.2,
            "rl_phi": 0.2,
            "rl_phi_std": 0.2,
            "rl_rho": 0.2,
            "rl_rho_std": 0.2,
            
        },
    },




======
    "RLWM": {
        "doc": "RLWM model.",
        "params": ["v", "rl_alpha", "rl_gamma", "rl_phi", "rl_rho"],
        "params_trans": [0, 0, 0, 0, 0],
        "params_std_upper": [2, 10, 10, 10, 10],
        "param_bounds": [[-3.0, 0.0, 0.0, 0.0, 0.0], [3.0, 1.0, 1.0, 1.0, 1.0]],
        "params_default": [1.0, -5.325, -1.15, -2.56, 0],
        "hddm_include": ['v', 'rl_alpha', 'rl_gamma', 'rl_phi', 'rl_rho'],
        "slice_widths": {
            "v": 1,
            "v_std": 1,
            "a": 1,
            "a_std": 1,
            "t": 0.01,
            "t_std": 0.15,
            "rl_alpha": 0.2,
            "rl_alpha_std": 0.2,
            "rl_gamma": 0.2,
            "rl_gamma_std": 0.2,
            "rl_phi": 0.2,
            "rl_phi_std": 0.2,
            "rl_rho": 0.2,
            "rl_rho_std": 0.2,
            
        },
'''