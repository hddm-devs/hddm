# import pandas as pd
# import numpy as np
# from copy import deepcopy
# from data_simulators import ddm_flexbound
# from data_simulators import levy_flexbound
# from data_simulators import ornstein_uhlenbeck
# from data_simulators import full_ddm
# from data_simulators import ddm_sdv
# from data_simulators import ddm
# from data_simulators import full_ddm_hddm_base
# from data_simulators import ddm_flex

# from data_simulators import ddm_flexbound_pre
# from data_simulators import race_model
# from data_simulators import lca
# from data_simulators import ddm_flexbound_seq2
# from data_simulators import ddm_flexbound_par2
# from data_simulators import ddm_flexbound_mic2_adj
# from data_simulators import ddm_flexbound_tradeoff
#import ssms.basic_simulators.boundary_functions as bf
#import ssms.basic_simulators.drift_functions as df
#from . import boundary_functions as bf
#from . import drift_functions as df

import ssms
from hddm.model_config import model_config

def simulator(**kwargs):
    """Basic data simulator for the models included in HDDM.

        :Arguments:
            theta : list or numpy.array or panda.DataFrame
                Parameters of the simulator. If 2d array, each row is treated as a 'trial'
                and the function runs n_sample * n_trials simulations.
            model: str <default='angle'>
                Determines the model that will be simulated.
            n_samples: int <default=1000>
                Number of simulation runs (for each trial if supplied n_trials > 1)
            delta_t: float
                Size fo timesteps in simulator (conceptually measured in seconds)
            max_t: float
                Maximum reaction the simulator can reach
            no_noise: bool <default=False>
                Turn noise of (useful for plotting purposes mostly)

        :Return: tuple
            can be (rts, responses, metadata)
            or     (rt-response histogram, metadata)
            or     (rts binned pointwise, responses, metadata)

        """

    data_tmp = ssms.basic_simulators.simulator(**kwargs)
    return (data_tmp['rts'], data_tmp['choices'], data_tmp['metadata'])