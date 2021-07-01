# External
#import scipy as scp
from scipy.stats import gamma
import numpy as np
import hddm.simulators


# Collection of boundary functions

# Constant: (multiplicative)
def constant(t = 0):
    return 1

# Angle (additive)
def angle(t = 1,
          theta = 1):
    return np.multiply(t, (- np.sin(theta) / np.cos(theta)))

# Generalized logistic bound (additive)
def generalized_logistic_bnd(t = 1,
                             B = 2.,
                             M = 3.,
                             v = 0.5):
    return 1 - (1 / np.power(1 + np.exp(- B * (t - M)), 1 / v))

# Weibull survival fun (multiplicative)
def weibull_cdf(t = 1,
                alpha  = 1,
                beta = 1):
    return np.exp( - np.power( np.divide(t, beta), alpha))

# # Gamma shape: (additive)
# def gamma_bnd(t = 1,
#               node = 1,
#               shape = 1.01,
#               scale = 1,
#               theta = 0):
#     return gamma.pdf(t - node, a = shape, scale = scale)

# Exponential decay with decay starting point (multiplicative)
# def exp_c1_c2(t = 1,
#               c1 = 1,
#               c2 = 1):
#
#     b = np.exp(- c2*(t-c1))
#
#     if t >= c1:
#
#         return b
#
#     else:
#         return 1

# # Logistic (additive)
# def logistic_bound(t = 1,
#                    node = 1,
#                    k = 1,
#                    midpoint = 1,
#                    max_val  = 3):

#     return - (max_val / (1 + np.exp(- k * ((t - midpoint)))))

# # Linear collapse (additive)
# def linear_collapse(t = 1,
#                     node = 1,
#                     theta = 1):
#     if t >= node:
#         return (t - node) * (- np.sin(theta) / np.cos(theta))
#     else:
#         return 0
