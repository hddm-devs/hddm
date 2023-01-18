# # External
# import scipy as scp
# from scipy.stats import gamma
# import numpy as np


# def constant(t=np.arange(0, 20, 0.1)):
#     return np.zeros(t.shape[0])


# def gamma_drift(t=np.arange(0, 20, 0.1), shape=2, scale=0.01, c=1.5):
#     """Drift function that follows a scaled gamma distribution

#     :Arguments:
#         t: np.ndarray <default=np.arange(0, 20, 0.1)>
#             Timepoints at which to evaluate the drift. Usually np.arange() of some sort.
#         shape: float <default=2>
#             Shape parameter of the gamma distribution
#         scale: float <default=0.01>
#             Scale parameter of the gamma distribution
#         c: float <default=1.5>
#             Scalar parameter that scales the peak of the gamma distribution
#             (Note this function follows a gamma distribution but does not integrate to 1)

#     :Return: np.ndarray
#          The gamma drift evaluated at the supplied timepoints t.

#     """

#     num_ = np.power(t, shape - 1) * np.exp(np.divide(-t, scale))
#     div_ = (
#         np.power(shape - 1, shape - 1)
#         * np.power(scale, shape - 1)
#         * np.exp(-(shape - 1))
#     )
#     return c * np.divide(num_, div_)


# def ds_support_analytic(t=np.arange(0, 10, 0.001), init_p=0, fix_point=1, slope=2):

#     """Solution to differential equation of the form: x' = slope*(fix_point - x), with initial
#        condition init_p. The solution takes the form: (init_p - fix_point) * exp(-slope * t) + fix_point

#     :Arguments:
#         t: np.ndarray <default=np.arange(0, 20, 0.1)>
#             Timepoints at which to evaluate the drift. Usually np.arange() of some sort.
#         init_p: float <default=0>
#             Initial condition of dynamical system
#         fix_point: float <default=1>
#             Fixed point of dynamical system
#         slope: float <default=0.01>
#             Coefficient in exponent of the solution.
#     :Return: np.ndarray
#          The gamma drift evaluated at the supplied timepoints t.

#     """

#     return (init_p - fix_point) * np.exp(-(slope * t)) + fix_point


# def ds_conflict_drift(
#     t=np.arange(0, 10, 0.001),
#     init_p_t=0,
#     init_p_d=0,
#     slope_t=1,
#     slope_d=1,
#     fixed_p_t=1,
#     coherence_t=1.5,
#     coherence_d=1.5,
# ):
#     """This drift is inspired by a conflict task which involves a target and a distractor stimuli both presented
#        simultaneously. Two drift timecourses are linearly combined weighted by the coherence in the respective target
#        and distractor stimuli. Each timecourse follows a dynamical system as described in the ds_support_analytic() function.

#     :Arguments:
#         t: np.ndarray <default=np.arange(0, 20, 0.1)>
#             Timepoints at which to evaluate the drift. Usually np.arange() of some sort.
#         init_p_t: float <default=0>
#             Initial condition of target drift timecourse
#         init_p_d: float <default=0>
#             Initial condition of distractor drift timecourse
#         slope_t: float <default=1>
#             Slope parameter for target drift timecourse
#         slope_d: float <default=1>
#             Slope parameter for distractor drift timecourse
#         fixed_p_t: float <default=1>
#             Fixed point for target drift timecourse
#         coherence_t: float <default=1.0>
#             Coefficient for the target drift timecourse
#         coherence_d: float <default=-1.0>
#             Coefficient for the distractor drift timecourse
#     :Return: np.ndarray
#          The full drift timecourse evaluated at the supplied timepoints t.

#     """

#     w_t = ds_support_analytic(t=t, init_p=init_p_t, fix_point=fixed_p_t, slope=slope_t)

#     w_d = ds_support_analytic(t=t, init_p=init_p_d, fix_point=0, slope=slope_d)

#     v_t = (w_t * coherence_t) + (w_d * coherence_d)

#     return v_t, w_t, w_d
