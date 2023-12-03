import ssms
from hddm.model_config import model_config
from ssms.basic_simulators import boundary_functions


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
    # Fix weibull issue with ssms
    if "model" in kwargs:
        if kwargs["model"] == "weibull":
            kwargs["model"] = "weibull_cdf"

    data_tmp = ssms.basic_simulators.simulator(**kwargs)
    return (data_tmp["rts"], data_tmp["choices"], data_tmp["metadata"])
