# # External
# # import scipy as scp
# from scipy.stats import gamma
# import numpy as np

# # import hddm.simulators


# # Collection of boundary functions

# # Constant: (multiplicative)
# def constant(t=0):
#     """constant boundary"""
#     return 1


# # Angle (additive)
# def angle(t=1, theta=1):
#     """angle boundary

#     :Arguments:
#         t: np.array or float <default = 1>
#             Time/s (with arbitrary measure, but in HDDM it is used as seconds) at which to evaluate the bound.
#         theta: float <default = 1>
#             Angle of the bound in radians.

#     """
#     return np.multiply(t, (-np.sin(theta) / np.cos(theta)))


# # Generalized logistic bound (additive)
# def generalized_logistic_bnd(t=1, B=2.0, M=3.0, v=0.5):
#     """generalized logistic bound

#     :Arguments:
#         t: np.array or float <default = 1>
#             Time/s (with arbitrary measure, but in HDDM it is used as seconds) at which to evaluate the bound.
#         B: float <default = 2.0>
#             Scale parameter
#         M: float <default = 3.0>
#             Mean shift parameter
#         v: float <default = 0.5>
#             Shape parameter

#     """
#     return 1 - (1 / np.power(1 + np.exp(-B * (t - M)), 1 / v))


# # Weibull survival fun (multiplicative)
# def weibull_cdf(t=1, alpha=1, beta=1):
#     """generalized logistic bound

#     :Arguments:
#         t: np.array or float <default = 1>
#             Time/s (with arbitrary measure, but in HDDM it is used as seconds) at which to evaluate the bound.
#         alpha: float <default = 1.0>
#             Shape parameter
#         beta: float <default = 1.0>
#             Shape parameter

#     """

#     return np.exp(-np.power(np.divide(t, beta), alpha))


# def conflict_gamma_bound(
#     a=0.5,
#     theta=0.5,
#     scale=1,
#     alpha_gamma=1.01,
#     scale_gamma=0.3,
#     t=np.arange(0, 20, 0.1),
# ):
#     """conflict bound that allows initial divergence then collapse

#     :Arguments:
#         t: np.array or float <default = 1>
#             Time/s (with arbitrary measure, but in HDDM it is used as seconds) at which to evaluate the bound.
#         theta: float <default = 0.5>
#             Collapse angle
#         scale: float <default = 1.0>
#             Scaling the gamma distribution of the boundary (since bound does not have to integrate to one)
#         a: float <default = 0.5>
#             Initial boundary separation
#         alpha_gamma: float <default = 1.01>
#             alpha parameter for a gamma in scale shape parameterization
#         scale_gamma: float <default = 0.3>
#             scale parameter for a gamma in scale shape paraemterization

#     """

#     return np.maximum(
#         a
#         + scale * gamma.pdf(t, a=alpha_gamma, loc=0, scale=scale_gamma)
#         + np.multiply(t, (-np.sin(theta) / np.cos(theta))),
#         0,
#     )


# # Gamma shape: (additive)
# def gamma_bnd(t=1, node=1, shape=1.01, scale=1, theta=0):
#     return gamma.pdf(t - node, a=shape, scale=scale)


# # Logistic (additive)
# def logistic_bound(t=1, node=1, k=1, midpoint=1, max_val=3):

#     return -(max_val / (1 + np.exp(-k * ((t - midpoint)))))


# # Linear collapse (additive)
# def linear_collapse(t=1, node=1, theta=1):
#     if t >= node:
#         return (t - node) * (-np.sin(theta) / np.cos(theta))
#     else:
#         return 0
