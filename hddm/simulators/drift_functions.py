# External
import scipy as scp
from scipy.stats import gamma
import numpy as np

def constant(t = np.arange(0, 20, 0.1)):
    return np.zeros(t.shape[0])

def gamma_drift(t = np.arange(0, 20, 0.1),
                shape = 2,
                scale = 0.01,
                c = 1.5):
    """Drift function that follows a scaled gamma distribution

    :Arguments:
        t: np.ndarray <default=np.arange(0, 20, 0.1)>
            Timepoints at which to evaluate the drift. Usually np.arange() of some sort. 
        shape: float <default=2>
            Shape parameter of the gamma distribution
        scale: float <default=0.01>
            Scale parameter of the gamma distribution
        c: float <default=1.5> 
            Scalar parameter that scales the peak of the gamma distribution 
            (Note this function follows a gamma distribution but does not integrate to 1)

    :Return: np.ndarray
         The gamma drift evaluated at the supplied timepoints t.

    """

    num_ = np.power(t, shape - 1) * np.exp(np.divide(-t, scale))
    div_ = np.power(shape - 1, shape - 1) * np.power(scale, shape - 1) * np.exp(- (shape - 1))
    return c * np.divide(num_, div_)
