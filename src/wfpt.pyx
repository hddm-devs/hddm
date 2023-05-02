# cython: embedsignature=True
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# distutils: language = c++
#
# Cython version of the Navarro & Fuss, 2009 DDM PDF. Based on the following code by Navarro & Fuss:
# http://www.psychocmath.logy.adelaide.edu.au/personalpages/staff/danielnavarro/resources/wfpt.m
#
# This implementation is about 170 times faster than the matlab
# reference version.
#
# Copyleft Thomas Wiecki (thomas_wiecki[at]brown.edu) & Imri Sofer, 2011
# GPLv3

import hddm
#from hddm.model_config import model_config

import scipy.integrate as integrate
from copy import copy
import numpy as np

cimport numpy as np
cimport cython

from cython.parallel import *
# cimport openmp

# include "pdf.pxi"
include 'integrate.pxi'

def pdf_array(np.ndarray[double, ndim=1] x, double v, double sv, double a, double z, double sz,
              double t, double st, double err=1e-4, bint logp=0, int n_st=2, int n_sz=2, bint use_adaptive=1,
              double simps_err=1e-3, double p_outlier=0, double w_outlier=0):

    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef np.ndarray[double, ndim = 1] y = np.empty(size, dtype=np.double)

    for i in prange(size, nogil=True):
        y[i] = full_pdf(x[i], v, sv, a, z, sz, t, st, err,
                        n_st, n_sz, use_adaptive, simps_err)

    y = y * (1 - p_outlier) + (w_outlier * p_outlier)
    if logp == 1:
        return np.log(y)
    else:
        return y

cdef inline bint p_outlier_in_range(double p_outlier):
    return (p_outlier >= 0) & (p_outlier <= 1)


def wiener_like(np.ndarray[double, ndim=1] x, double v, double sv, double a, double z, double sz, double t,
                double st, double err, int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-8,
                double p_outlier=0, double w_outlier=0.1):
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef double p
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier

    if not p_outlier_in_range(p_outlier):
        return -np.inf

    for i in range(size):
        p = full_pdf(x[i], v, sv, a, z, sz, t, st, err,
                     n_st, n_sz, use_adaptive, simps_err)
        # If one probability = 0, the log sum will be -Inf
        p = p * (1 - p_outlier) + wp_outlier
        if p == 0:
            return -np.inf

        sum_logp += log(p)

    return sum_logp

def wiener_like_rlddm(np.ndarray[double, ndim=1] x,
                      np.ndarray[long, ndim=1] response,
                      np.ndarray[double, ndim=1] feedback,
                      np.ndarray[long, ndim=1] split_by,
                      double q, double alpha, double pos_alpha, double v, 
                      double sv, double a, double z, double sz, double t,
                      double st, double err, int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-8,
                      double p_outlier=0, double w_outlier=0):
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i, j
    cdef Py_ssize_t s_size
    cdef int s
    cdef double p
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier
    cdef double alfa
    cdef double pos_alfa
    cdef np.ndarray[double, ndim=1] qs = np.array([q, q])
    cdef np.ndarray[double, ndim=1] xs
    cdef np.ndarray[double, ndim=1] feedbacks
    cdef np.ndarray[long, ndim=1] responses
    cdef np.ndarray[long, ndim=1] unique = np.unique(split_by)

    if not p_outlier_in_range(p_outlier):
        return -np.inf

    if pos_alpha==100.00:
        pos_alfa = alpha
    else:
        pos_alfa = pos_alpha

    # unique represent # of conditions
    for j in range(unique.shape[0]):
        s = unique[j]
        # select trials for current condition, identified by the split_by-array
        feedbacks = feedback[split_by == s]
        responses = response[split_by == s]
        xs = x[split_by == s]
        s_size = xs.shape[0]
        qs[0] = q
        qs[1] = q

        # don't calculate pdf for first trial but still update q
        if feedbacks[0] > qs[responses[0]]:
            alfa = (2.718281828459**pos_alfa) / (1 + 2.718281828459**pos_alfa)
        else:
            alfa = (2.718281828459**alpha) / (1 + 2.718281828459**alpha)

        # qs[1] is upper bound, qs[0] is lower bound. feedbacks is reward
        # received on current trial.
        qs[responses[0]] = qs[responses[0]] + \
            alfa * (feedbacks[0] - qs[responses[0]])

        # loop through all trials in current condition
        for i in range(1, s_size):
            p = full_pdf(xs[i], ((qs[1] - qs[0]) * v), sv, a, z,
                         sz, t, st, err, n_st, n_sz, use_adaptive, simps_err)
            # If one probability = 0, the log sum will be -Inf
            p = p * (1 - p_outlier) + wp_outlier
            if p == 0:
                return -np.inf
            sum_logp += log(p)

            # get learning rate for current trial. if pos_alpha is not in
            # include it will be same as alpha so can still use this
            # calculation:
            if feedbacks[i] > qs[responses[i]]:
                alfa = (2.718281828459**pos_alfa) / (1 + 2.718281828459**pos_alfa)
            else:
                alfa = (2.718281828459**alpha) / (1 + 2.718281828459**alpha)

            # qs[1] is upper bound, qs[0] is lower bound. feedbacks is reward
            # received on current trial.
            qs[responses[i]] = qs[responses[i]] + \
                alfa * (feedbacks[i] - qs[responses[i]])
    return sum_logp


def softmax(np.ndarray[double, ndim=1] q_val, double beta):
    q_val = np.array(q_val)*beta
    q_val = np.exp(q_val)
    q_val = q_val / np.sum(q_val)
    return q_val



def wiener_like_rlssm_nn_rlwm(str model, 
                      np.ndarray[long, ndim=1] block_num,
                      np.ndarray[long, ndim=1] set_size,
                      np.ndarray[double, ndim=1] stim,
                      np.ndarray[double, ndim=1] rt,
                      np.ndarray[long, ndim=1] response,
                      np.ndarray[double, ndim=1] feedback,
                      np.ndarray[double, ndim=1] params_ssm,
                      np.ndarray[double, ndim=1] params_rl,
                      np.ndarray[double, ndim=2] params_bnds,
                      double p_outlier=0, double w_outlier=0, network = None):

    cdef long[:] mv_block_num = block_num
    cdef long[:] mv_set_size = set_size
    cdef double[:] mv_stim = stim
    cdef double[:] mv_rt = rt
    cdef long[:] mv_response = response
    cdef double[:] mv_feedback = feedback
    cdef double[:] mv_params_ssm = params_ssm
    cdef double[:] mv_params_rl = params_rl
    cdef double[:, :] mv_params_bnds = params_bnds

    #cdef double v = params_ssm[0]
    cdef double a = params_ssm[0]
    cdef double z = params_ssm[1]
    cdef double theta = params_ssm[2]

    cdef double rl_alpha = params_rl[0]
    cdef double rl_gamma = 1
    cdef double rl_phi = params_rl[1] 
    cdef double rl_rho = params_rl[2]
    cdef double rl_beta = 50

    cdef Py_ssize_t size = rt.shape[0]
    cdef Py_ssize_t tr

    cdef int num_actions = 3
    cdef int C = 3

    cdef double log_p = 0
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier

    cdef double[:] stims
    cdef long[:] responses
    cdef double[:] feedbacks
    cdef double[:] rts

    cdef Py_ssize_t n_params = num_actions + 3 # this should be num of params in ssm 
    cdef np.ndarray[float, ndim=2] data = np.zeros((size, n_params + 2), dtype = np.float32)
    cdef long[:] bl_unique = np.unique(mv_block_num)

    cdef double weight
    cdef np.ndarray[double, ndim=2] q_RL
    cdef np.ndarray[double, ndim=2] q_WM
    cdef float[:] pol_RL = np.ones(num_actions, dtype = np.float32)
    cdef float[:] pol_WM = np.ones(num_actions, dtype = np.float32)
    cdef float[:] pol = np.ones(num_actions, dtype = np.float32)

    cdef int cumm_s_size = 0
    cdef int bl_size
    cdef int const
    cdef int curr_block_index
    cdef int block_ns
    cdef int state
    cdef int action
    cdef float reward
    cdef float ll_min = -16.11809
    
    cdef float [:, :] mv_data = data
    cdef double [:, :] mv_q_RL
    cdef double [:, :] mv_q_WM
    
    cdef int i_loop, j_loop
    cdef double sum_exp_RL = 0
    cdef double sum_exp_WM = 0


    if not p_outlier_in_range(p_outlier):
        return -np.inf
    
    # if params_ssm[1] < params_bnds[0][num_actions] or params_ssm[1] > params_bnds[1][num_actions]:
    #         return -np.inf
    # if params_ssm[2] < params_bnds[0][num_actions+2] or params_ssm[2] > params_bnds[1][num_actions+2]:
    #         return -np.inf 

    rl_alpha = (2.718281828459**rl_alpha) / (1 + 2.718281828459**rl_alpha)
    rl_gamma = 1 #(2.718281828459**rl_gamma) / (1 + 2.718281828459**rl_gamma)
    rl_phi = (2.718281828459**rl_phi) / (1 + 2.718281828459**rl_phi)
    rl_rho = (2.718281828459**rl_rho) / (1 + 2.718281828459**rl_rho)
    

    if a < mv_params_bnds[0][num_actions] or a > mv_params_bnds[1][num_actions]:
        return -np.inf
    if z < mv_params_bnds[0][num_actions+1] or z > mv_params_bnds[1][num_actions+1]:
        return -np.inf
    if theta < mv_params_bnds[0][num_actions+2] or theta > mv_params_bnds[1][num_actions+2]:
        return -np.inf
    if rl_alpha < mv_params_bnds[0][6] or rl_alpha > mv_params_bnds[1][6]:
        return -np.inf
    if rl_phi < mv_params_bnds[0][7] or rl_phi > mv_params_bnds[1][7]:
        return -np.inf
    if rl_rho < mv_params_bnds[0][8] or rl_rho > mv_params_bnds[1][8]:
        return -np.inf
    
    #print("incoming- ", params_ssm, " | ", params_rl)
    curr_block_index = 0

    for j in range(bl_unique.shape[0]):
        bl = bl_unique[j]
        #block_ns = len(np.unique(np.asarray(stim)[block_num == bl])) # THIS IS INEFFICIENT
        block_ns = mv_set_size[curr_block_index]
        #tp_stims = np.asarray(stim)[block_num == bl]

        # responses = np.asarray(response)[block_num == bl]
        # feedbacks = np.asarray(feedback)[block_num == bl]
        # rts = np.asarray(rt)[block_num == bl]
        const = block_ns*15
        stims = mv_stim[curr_block_index:curr_block_index+const]
        responses = mv_response[curr_block_index:curr_block_index+const]
        feedbacks = mv_feedback[curr_block_index:curr_block_index+const]
        rts = mv_rt[curr_block_index:curr_block_index+const]
        
        #print(">>> ", tp_stims[0:10], np.asarray(stims[0:10]), tp_stims[-5:], np.asarray(stims[-5:]))

        bl_size = rts.shape[0]
        
        q_RL = np.ones((block_ns, num_actions)) * 1/num_actions
        q_WM = np.ones((block_ns, num_actions)) * 1/num_actions
        weight = rl_rho * min(1, C/block_ns)
        
        mv_q_RL = q_RL
        mv_q_WM = q_WM

        # loop through all trials in current condition
        for tr in range(0, bl_size):

            state = int(stims[tr])
            action = responses[tr]
            reward = feedbacks[tr]

            # pol_RL = softmax(np.asarray(q_RL[state]), rl_beta)
            # pol_WM = softmax(np.asarray(q_WM[state]), rl_beta)

            # pol = weight * np.asarray(pol_WM) + (1-weight) * np.asarray(pol_RL)

            sum_exp_RL = 0
            sum_exp_WM = 0

            for i_loop in range(num_actions):
                sum_exp_RL += 2.71828**(mv_q_RL[state, i_loop]*rl_beta)
                sum_exp_WM += 2.71828**(mv_q_WM[state, i_loop]*rl_beta)
            
            for i_loop in range(num_actions):
                pol_RL[i_loop] = 2.71828**(mv_q_RL[state, i_loop]*rl_beta) / sum_exp_RL
                pol_WM[i_loop] = 2.71828**(mv_q_WM[state, i_loop]*rl_beta) / sum_exp_WM

            for i_loop in range(num_actions):
                pol[i_loop] = weight * pol_WM[i_loop] + (1-weight) * pol_RL[i_loop]


            for a_idx in range(num_actions):
                mv_data[cumm_s_size + tr, a_idx] = pol[a_idx]
                if pol[a_idx] < 0 or pol[a_idx] > 1:
                    print("ERROR")

            # Check for boundary violations -- if true, return -np.inf
            # for a_idx in range(num_actions):
            #     if mv_data[cumm_s_size + tr, a_idx] < mv_params_bnds[0,0] or mv_data[cumm_s_size + tr, a_idx] > mv_params_bnds[1,0]:
            #         return -np.inf
            
            #print("\tbefore- ", mv_q_RL[state, action], mv_q_WM[state, action])
            if reward == 1:
                mv_q_RL[state, action] = mv_q_RL[state, action] + rl_alpha * (reward - mv_q_RL[state, action])
                mv_q_WM[state, action] = mv_q_WM[state, action] + 1 * (reward - mv_q_WM[state, action])
            elif reward == 0:
                mv_q_RL[state, action] = mv_q_RL[state, action] + rl_gamma * rl_alpha * (reward - mv_q_RL[state, action])
                mv_q_WM[state, action] = mv_q_WM[state, action] + rl_gamma * 1 * (reward - mv_q_WM[state, action])
            #print("\tafter- ", mv_q_RL[state, action], mv_q_WM[state, action])

            #q_WM = q_WM + rl_phi * ((1/num_actions) - np.asarray(q_WM))
            for i_loop in range(block_ns):
                for j_loop in range(num_actions):
                    mv_q_WM[i_loop, j_loop] = mv_q_WM[i_loop, j_loop] + rl_phi * ((1/num_actions) - mv_q_WM[i_loop, j_loop])
            
            # print(">>> ", np.asarray(mv_q_WM[0:5, :]), q_WM[0:5, :])
            # print("@@@ ", np.asarray(mv_q_RL[0:5, :]), q_RL[0:5, :])
            # print(" --- ", np.asarray(pol_RL), np.asarray(pol_WM), np.asarray(pol))

        cumm_s_size += bl_size
        curr_block_index += const

    mv_data[:, num_actions] = a #np.tile(params_ssm[0], (size,)).astype(np.float32) # a
    mv_data[:, num_actions+1] = z #np.tile(params_ssm[1], (size,)).astype(np.float32) # z
    mv_data[:, num_actions+2] = theta #np.tile(params_ssm[2], (size,)).astype(np.float32) # theta
    data[:, n_params:] = np.stack([rt, response], axis = 1)

    #print("\n\ndata = ", data[0:3, :])

    # Call to network:
    if p_outlier == 0:
        sum_logp = np.sum(np.core.umath.maximum(network.predict_on_batch(data), ll_min))
    else:
        sum_logp = np.sum(np.log(np.exp(np.core.umath.maximum(network.predict_on_batch(data), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))

    return sum_logp


# def wiener_like_rlssm_nn_rlwm(str model, 
#                       np.ndarray[long, ndim=1] block_num,
#                       np.ndarray[double, ndim=1] stim,
#                       np.ndarray[double, ndim=1] rt,
#                       np.ndarray[long, ndim=1] response,
#                       np.ndarray[double, ndim=1] feedback,
#                       np.ndarray[double, ndim=1] params_ssm,
#                       np.ndarray[double, ndim=1] params_rl,
#                       np.ndarray[double, ndim=2] params_bnds,
#                       double p_outlier=0, double w_outlier=0, network = None):

#     #cdef double v = params_ssm[0]
#     cdef double a = params_ssm[0]
#     cdef double z = params_ssm[1]
#     cdef double theta = params_ssm[2]

#     cdef double rl_alpha = params_rl[0]
#     #cdef double rl_gamma = params_rl[1]
#     cdef double rl_phi = params_rl[1] 
#     cdef double rl_rho = params_rl[2]
#     cdef double rl_beta = 50

#     cdef Py_ssize_t size = rt.shape[0]
#     cdef Py_ssize_t tr

#     cdef int num_actions = 3
#     cdef int C = 3

#     cdef double log_p = 0
#     cdef double sum_logp = 0
#     cdef double wp_outlier = w_outlier * p_outlier

#     cdef np.ndarray[double, ndim=1] stims
#     cdef np.ndarray[long, ndim=1] responses
#     cdef np.ndarray[double, ndim=1] feedbacks
#     cdef np.ndarray[double, ndim=1] rts

#     cdef Py_ssize_t n_params = num_actions + 3 # this should be num of params in ssm 
#     cdef np.ndarray[float, ndim=2] data = np.zeros((size, n_params + 2), dtype = np.float32)
#     cdef np.ndarray[long, ndim=1] bl_unique = np.unique(block_num)

#     cdef double weight
#     cdef np.ndarray[double, ndim=2] q_RL
#     cdef np.ndarray[double, ndim=2] q_WM
#     cdef np.ndarray[double, ndim=1] pol_RL
#     cdef np.ndarray[double, ndim=1] pol_WM
#     cdef np.ndarray[double, ndim=1] pol

#     cdef int cumm_s_size = 0
#     cdef float ll_min = -16.11809
    
#     cdef float [:, :] mv_data = data
#     cdef double [:, :] mv_q_RL
#     cdef double [:, :] mv_q_WM
    

#     if not p_outlier_in_range(p_outlier):
#         return -np.inf
    
#     if params_ssm[1] < params_bnds[0][num_actions] or params_ssm[1] > params_bnds[1][num_actions]:
#             return -np.inf
#     if params_ssm[2] < params_bnds[0][num_actions+2] or params_ssm[2] > params_bnds[1][num_actions+2]:
#             return -np.inf 

#     rl_alpha = (2.718281828459**rl_alpha) / (1 + 2.718281828459**rl_alpha)
#     rl_gamma = 1 #(2.718281828459**rl_gamma) / (1 + 2.718281828459**rl_gamma)
#     rl_phi = (2.718281828459**rl_phi) / (1 + 2.718281828459**rl_phi)
#     rl_rho = (2.718281828459**rl_rho) / (1 + 2.718281828459**rl_rho)
    
#     # if rl_alpha < params_bnds[0][7] or rl_alpha > params_bnds[1][7]:
#     #     return -np.inf
#     # if rl_gamma < params_bnds[0][7] or rl_gamma > params_bnds[1][7]:
#     #     return -np.inf
#     # if rl_phi < params_bnds[0][7] or rl_phi > params_bnds[1][7]:
#     #     return -np.inf
#     # if rl_rho < params_bnds[0][7] or rl_rho > params_bnds[1][7]:
#     #     return -np.inf
    
#     #print("incoming- ", params_ssm, " | ", params_rl)

#     for j in range(bl_unique.shape[0]):
#         bl = bl_unique[j]
#         stims = stim[block_num == bl]
#         responses = response[block_num == bl]
#         feedbacks = feedback[block_num == bl]
#         rts = rt[block_num == bl]

#         bl_size = rts.shape[0]
        
#         q_RL = np.ones((bl_size, num_actions)) * 1/num_actions
#         q_WM = np.ones((bl_size, num_actions)) * 1/num_actions
#         weight = rl_rho * min(1, C/bl_size)
        
#         mv_q_RL = q_RL
#         mv_q_WM = q_WM

#         # loop through all trials in current condition
#         for tr in range(0, bl_size):

#             state = int(stims[tr])
#             action = responses[tr]
#             reward = feedbacks[tr]

#             pol_RL = softmax(q_RL[state], rl_beta)
#             pol_WM = softmax(q_WM[state], rl_beta)

#             pol = weight * pol_WM + (1-weight) * pol_RL

#             if tr != 0:
#                 for a_idx in range(num_actions):
#                     mv_data[cumm_s_size + tr, a_idx] = pol[a_idx]

#             # Check for boundary violations -- if true, return -np.inf
#             for a_idx in range(num_actions):
#                 if mv_data[cumm_s_size + tr, a_idx] < params_bnds[0][0] or mv_data[cumm_s_size + tr, a_idx] > params_bnds[1][0]:
#                     return -np.inf
            
#             #print("\tbefore- ", mv_q_RL[state, action], mv_q_WM[state, action])
#             if reward == 1:
#                 mv_q_RL[state, action] = mv_q_RL[state, action] + rl_alpha * (reward - mv_q_RL[state, action])
#                 mv_q_WM[state, action] = mv_q_WM[state, action] + 1 * (reward - mv_q_WM[state, action])
#             elif reward == 0:
#                 mv_q_RL[state, action] = mv_q_RL[state, action] + rl_gamma * rl_alpha * (reward - mv_q_RL[state, action])
#                 mv_q_WM[state, action] = mv_q_WM[state, action] + rl_gamma * 1 * (reward - mv_q_WM[state, action])
#             #print("\tafter- ", mv_q_RL[state, action], mv_q_WM[state, action])

#             q_WM = q_WM + rl_phi * ((1/num_actions) - q_WM)

#         cumm_s_size += bl_size

#     data[:, num_actions] = params_ssm[0] #np.tile(params_ssm[1:2], (size, 1)).astype(np.float32) # a
#     data[:, num_actions+1] = params_ssm[1] # z
#     data[:, num_actions+2] = params_ssm[2] #np.tile(params_ssm[2:3], (size, 1)).astype(np.float32) # t
#     data[:, n_params:] = np.stack([rt, response], axis = 1)

#     #print("\n\ndata = ", data[0:3, :])

#     # Call to network:
#     if p_outlier == 0:
#         sum_logp = np.sum(np.core.umath.maximum(network.predict_on_batch(data), ll_min))
#     else:
#         sum_logp = np.sum(np.log(np.exp(np.core.umath.maximum(network.predict_on_batch(data), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))

#     return sum_logp


def wiener_like_rlssm_nn(str model, 
                      np.ndarray[double, ndim=1] x,
                      np.ndarray[long, ndim=1] response,
                      np.ndarray[double, ndim=1] feedback,
                      np.ndarray[long, ndim=1] split_by,
                      double q,
                      np.ndarray[double, ndim=1] params_ssm,
                      np.ndarray[double, ndim=1] params_rl,
                      np.ndarray[double, ndim=2] params_bnds,
                      double p_outlier=0, double w_outlier=0, network = None):

    cdef double v = params_ssm[0]
    cdef double rl_alpha = params_rl[0]

    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i, j, i_p
    cdef Py_ssize_t s_size
    cdef int s
    cdef double log_p = 0
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier
    cdef double alfa
    cdef double pos_alfa
    cdef np.ndarray[double, ndim=1] qs = np.array([q, q])
    cdef np.ndarray[double, ndim=1] xs
    cdef np.ndarray[double, ndim=1] feedbacks
    cdef np.ndarray[long, ndim=1] responses
    cdef np.ndarray[long, ndim=1] responses_qs
    cdef np.ndarray[long, ndim=1] unique = np.unique(split_by)
    cdef Py_ssize_t n_params = params_ssm.shape[0] #+ params_rl.shape[0]
    cdef np.ndarray[float, ndim=2] data = np.zeros((size, n_params + 2), dtype = np.float32)
    cdef float ll_min = -16.11809
    cdef int cumm_s_size = 0

    if not p_outlier_in_range(p_outlier):
        return -np.inf
    
    # Check for boundary violations -- if true, return -np.inf
    for i_p in np.arange(1, len(params_ssm)):
        lower_bnd = params_bnds[0][i_p]
        upper_bnd = params_bnds[1][i_p]

        if params_ssm[i_p] < lower_bnd or params_ssm[i_p] > upper_bnd:
            return -np.inf


    if len(params_rl) == 2:
        pos_alfa = params_rl[1]
    else:
        pos_alfa = params_rl[0]
    
    # unique represent # of conditions
    for j in range(unique.shape[0]):
        s = unique[j]
        # select trials for current condition, identified by the split_by-array
        feedbacks = feedback[split_by == s]
        responses = response[split_by == s]
        xs = x[split_by == s]
        s_size = xs.shape[0]
        qs[0] = q
        qs[1] = q

        responses_qs = responses
        responses_qs[responses_qs == -1] = 0

        # don't calculate pdf for first trial but still update q
        if feedbacks[0] > qs[responses_qs[0]]:
            alfa = (2.718281828459**pos_alfa) / (1 + 2.718281828459**pos_alfa)
        else:
            alfa = (2.718281828459**rl_alpha) / (1 + 2.718281828459**rl_alpha)
        

        # qs[1] is upper bound, qs[0] is lower bound. feedbacks is reward
        # received on current trial.
        qs[responses_qs[0]] = qs[responses_qs[0]] + \
            alfa * (feedbacks[0] - qs[responses_qs[0]])


        data[0, 0] = 0.0
        # loop through all trials in current condition
        for i in range(1, s_size):
            data[cumm_s_size + i, 0] = (qs[1] - qs[0]) * v
            # Check for boundary violations -- if true, return -np.inf
            if data[cumm_s_size + i, 0] < params_bnds[0][0] or data[cumm_s_size + i, 0] > params_bnds[1][0]:
                return -np.inf

            # get learning rate for current trial. if pos_alpha is not in
            # include it will be same as alpha so can still use this
            # calculation:
            if feedbacks[i] > qs[responses_qs[i]]:
                alfa = (2.718281828459**pos_alfa) / (1 + 2.718281828459**pos_alfa)
            else:
                alfa = (2.718281828459**rl_alpha) / (1 + 2.718281828459**rl_alpha)

            # qs[1] is upper bound, qs[0] is lower bound. feedbacks is reward
            # received on current trial.
            qs[responses_qs[i]] = qs[responses_qs[i]] + \
                alfa * (feedbacks[i] - qs[responses_qs[i]])
        cumm_s_size += s_size


    data[:, 1:n_params] = np.tile(params_ssm[1:], (size, 1)).astype(np.float32)
    data[:, n_params:] = np.stack([x, response], axis = 1)

    # Call to network:
    if p_outlier == 0:
        sum_logp = np.sum(np.core.umath.maximum(network.predict_on_batch(data), ll_min))
    else:
        sum_logp = np.sum(np.log(np.exp(np.core.umath.maximum(network.predict_on_batch(data), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))

    return sum_logp


def wiener_like_rl(np.ndarray[long, ndim=1] response,
                   np.ndarray[double, ndim=1] feedback,
                   np.ndarray[long, ndim=1] split_by,
                   double q, double alpha, double pos_alpha, double v, double z,
                   double err=1e-4, int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-8,
                   double p_outlier=0, double w_outlier=0):
    cdef Py_ssize_t size = response.shape[0]
    cdef Py_ssize_t i, j
    cdef Py_ssize_t s_size
    cdef int s
    cdef double drift
    cdef double p
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier
    cdef double alfa
    cdef double pos_alfa
    cdef np.ndarray[double, ndim=1] qs = np.array([q, q])
    cdef np.ndarray[double, ndim=1] feedbacks
    cdef np.ndarray[long, ndim=1] responses
    cdef np.ndarray[long, ndim=1] unique = np.unique(split_by)

    if not p_outlier_in_range(p_outlier):
        return -np.inf

    if pos_alpha==100.00:
        pos_alfa = alpha
    else:
        pos_alfa = pos_alpha
        
    # unique represent # of conditions
    for j in range(unique.shape[0]):
        s = unique[j]
        # select trials for current condition, identified by the split_by-array
        feedbacks = feedback[split_by == s]
        responses = response[split_by == s]
        s_size = responses.shape[0]
        qs[0] = q
        qs[1] = q

        # don't calculate pdf for first trial but still update q
        if feedbacks[0] > qs[responses[0]]:
            alfa = (2.718281828459**pos_alfa) / (1 + 2.718281828459**pos_alfa)
        else:
            alfa = (2.718281828459**alpha) / (1 + 2.718281828459**alpha)

        # qs[1] is upper bound, qs[0] is lower bound. feedbacks is reward
        # received on current trial.
        qs[responses[0]] = qs[responses[0]] + \
            alfa * (feedbacks[0] - qs[responses[0]])

        # loop through all trials in current condition
        for i in range(1, s_size):

            drift = (qs[1] - qs[0]) * v

            if drift == 0:
                p = 0.5
            else:
                if responses[i] == 1:
                    p = (2.718281828459**(-2 * z * drift) - 1) / \
                        (2.718281828459**(-2 * drift) - 1)
                else:
                    p = 1 - (2.718281828459**(-2 * z * drift) - 1) / \
                        (2.718281828459**(-2 * drift) - 1)

            # If one probability = 0, the log sum will be -Inf
            p = p * (1 - p_outlier) + wp_outlier
            if p == 0:
                return -np.inf

            sum_logp += log(p)

            # get learning rate for current trial. if pos_alpha is not in
            # include it will be same as alpha so can still use this
            # calculation:
            if feedbacks[i] > qs[responses[i]]:
                alfa = (2.718281828459**pos_alfa) / (1 + 2.718281828459**pos_alfa)
            else:
                alfa = (2.718281828459**alpha) / (1 + 2.718281828459**alpha)

            # qs[1] is upper bound, qs[0] is lower bound. feedbacks is reward
            # received on current trial.
            qs[responses[i]] = qs[responses[i]] + \
                alfa * (feedbacks[i] - qs[responses[i]])
    return sum_logp


def wiener_like_multi(np.ndarray[double, ndim=1] x, v, sv, a, z, sz, t, st, double err, multi=None,
                      int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-3,
                      double p_outlier=0, double w_outlier=0):
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef double p = 0
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier

    if multi is None:
        return full_pdf(x, v, sv, a, z, sz, t, st, err)
    else:
        params = {'v': v, 'z': z, 't': t, 'a': a, 'sv': sv, 'sz': sz, 'st': st}
        params_iter = copy(params)
        for i in range(size):
            for param in multi:
                params_iter[param] = params[param][i]
            if abs(x[i]) != 999.:
                p = full_pdf(x[i], params_iter['v'],
                             params_iter['sv'], params_iter['a'], params_iter['z'],
                             params_iter['sz'], params_iter['t'], params_iter['st'],
                             err, n_st, n_sz, use_adaptive, simps_err)
                p = p * (1 - p_outlier) + wp_outlier
            elif x[i] == 999.:
                p = prob_ub(params_iter['v'], params_iter['a'], params_iter['z'])
            else: # x[i] == -999.
                p = 1 - prob_ub(params_iter['v'], params_iter['a'], params_iter['z'])

            sum_logp += log(p)

        return sum_logp


def wiener_like_multi_rlddm(np.ndarray[double, ndim=1] x, 
                      np.ndarray[long, ndim=1] response,
                      np.ndarray[double, ndim=1] feedback,
                      np.ndarray[long, ndim=1] split_by,
                      double q, v, sv, a, z, sz, t, st, alpha, double err, multi=None,
                      int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-3,
                      double p_outlier=0, double w_outlier=0):
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t ij
    cdef Py_ssize_t s_size
    cdef double p = 0
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier
    cdef int s
    cdef np.ndarray[double, ndim=1] qs = np.array([q, q])

    if multi is None:
        return full_pdf(x, v, sv, a, z, sz, t, st, err)
    else:
        params = {'v': v, 'z': z, 't': t, 'a': a, 'sv': sv, 'sz': sz, 'st': st, 'alpha':alpha}
        params_iter = copy(params)
        qs[0] = q
        qs[1] = q
        for i in range(size):
            for param in multi:
                params_iter[param] = params[param][i]

            if (i != 0):
                if (split_by[i] != split_by[i-1]):
                    qs[0] = q
                    qs[1] = q

            p = full_pdf(x[i], params_iter['v'] * (qs[1] - qs[0]),
                         params_iter['sv'], params_iter['a'], params_iter['z'],
                         params_iter['sz'], params_iter[
                             't'], params_iter['st'],
                         err, n_st, n_sz, use_adaptive, simps_err)
            p = p * (1 - p_outlier) + wp_outlier
            sum_logp += log(p)

            alfa = (2.718281828459**params_iter['alpha']) / (1 + 2.718281828459**params_iter['alpha'])   
            qs[response[i]] = qs[response[i]] + alfa * (feedback[i] - qs[response[i]])

        return sum_logp


def wiener_like_rlssm_nn_reg(np.ndarray[float, ndim=2] data,
                      np.ndarray[float, ndim=2] rl_arr,
                      np.ndarray[double, ndim=1] x,
                      np.ndarray[long, ndim=1] response,
                      np.ndarray[double, ndim=1] feedback,
                      np.ndarray[long, ndim=1] split_by,
                      double q,
                      np.ndarray[double, ndim=2] params_bnds,
                      double p_outlier=0, double w_outlier=0, network = None):
    cdef double rl_alpha 
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i, j, i_p
    cdef Py_ssize_t s_size
    cdef int s
    cdef double log_p = 0
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier
    cdef double alfa
    cdef np.ndarray[double, ndim=1] qs = np.array([q, q])
    cdef np.ndarray[double, ndim=1] xs
    cdef np.ndarray[double, ndim=1] feedbacks
    cdef np.ndarray[long, ndim=1] responses
    cdef np.ndarray[long, ndim=1] responses_qs
    cdef np.ndarray[long, ndim=1] unique = np.unique(split_by)
    cdef np.ndarray[float, ndim=2] data_copy = data
    cdef float ll_min = -16.11809
    cdef int cumm_s_size = 0

    if not p_outlier_in_range(p_outlier):
        return -np.inf
    
    # Check for boundary violations -- if true, return -np.inf
    for i_p in np.arange(1, data.shape[1]-2):
        lower_bnd = params_bnds[0][i_p]
        upper_bnd = params_bnds[1][i_p]

        if data[:,i_p].min() < lower_bnd or data[:,i_p].max() > upper_bnd:
            return -np.inf

    # unique represent # of conditions
    for j in range(unique.shape[0]):
        s = unique[j]
        # select trials for current condition, identified by the split_by-array
        feedbacks = feedback[split_by == s]
        responses = response[split_by == s]
        xs = x[split_by == s]
        s_size = xs.shape[0]
        qs[0] = q
        qs[1] = q

        responses_qs = responses
        responses_qs[responses_qs == -1] = 0

        # loop through all trials in current condition
        for i in range(0, s_size):
            tp_scale = data[cumm_s_size + i, 0]
            if tp_scale < 0:
                return -np.inf

            data_copy[cumm_s_size + i, 0] = (qs[1] - qs[0]) * tp_scale 

            # Check for boundary violations -- if true, return -np.inf
            if data_copy[cumm_s_size + i, 0] < params_bnds[0][0] or data_copy[cumm_s_size + i, 0] > params_bnds[1][0]:
                return -np.inf

            rl_alpha = rl_arr[cumm_s_size + i, 0]
            alfa = (2.718281828459**rl_alpha) / (1 + 2.718281828459**rl_alpha)

            qs[responses_qs[i]] = qs[responses_qs[i]] + \
                alfa * (feedbacks[i] - qs[responses_qs[i]])
        cumm_s_size += s_size

    # Call to network:
    if p_outlier == 0:
        sum_logp = np.sum(np.core.umath.maximum(network.predict_on_batch(data_copy), ll_min))
    else:
        sum_logp = np.sum(np.log(np.exp(np.core.umath.maximum(network.predict_on_batch(data_copy), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))

    return sum_logp


def gen_rts_from_cdf(double v, double sv, double a, double z, double sz, double t,
                     double st, int samples=1000, double cdf_lb=-6, double cdf_ub=6, double dt=1e-2):

    cdef np.ndarray[double, ndim = 1] x = np.arange(cdf_lb, cdf_ub, dt)
    cdef np.ndarray[double, ndim = 1] l_cdf = np.empty(x.shape[0], dtype=np.double)
    cdef double pdf, rt
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i, j
    cdef int idx

    l_cdf[0] = 0
    for i from 1 <= i < size:
        pdf = full_pdf(x[i], v, sv, a, z, sz, 0, 0, 1e-4)
        l_cdf[i] = l_cdf[i - 1] + pdf

    l_cdf /= l_cdf[x.shape[0] - 1]

    cdef np.ndarray[double, ndim = 1] rts = np.empty(samples, dtype=np.double)
    cdef np.ndarray[double, ndim = 1] f = np.random.rand(samples)
    cdef np.ndarray[double, ndim = 1] delay

    if st != 0:
        delay = (np.random.rand(samples) * st + (t - st / 2.))
    for i from 0 <= i < samples:
        idx = np.searchsorted(l_cdf, f[i])
        rt = x[idx]
        if st == 0:
            rt = rt + np.sign(rt) * t
        else:
            rt = rt + np.sign(rt) * delay[i]
        rts[i] = rt
    return rts


def wiener_like_contaminant(np.ndarray[double, ndim=1] x, np.ndarray[int, ndim=1] cont_x, double v,
                            double sv, double a, double z, double sz, double t, double st, double t_min,
                            double t_max, double err, int n_st=10, int n_sz=10, bint use_adaptive=1,
                            double simps_err=1e-8):
    """Wiener likelihood function where RTs could come from a
    separate, uniform contaminant distribution.

    Reference: Lee, Vandekerckhove, Navarro, & Tuernlinckx (2007)
    """
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef double p
    cdef double sum_logp = 0
    cdef int n_cont = np.sum(cont_x)
    cdef int pos_cont = 0

    for i in prange(size, nogil=True):
        if cont_x[i] == 0:
            p = full_pdf(x[i], v, sv, a, z, sz, t, st, err,
                         n_st, n_sz, use_adaptive, simps_err)
            if p == 0:
                with gil:
                    return -np.inf
            sum_logp += log(p)
        # If one probability = 0, the log sum will be -Inf

    # add the log likelihood of the contaminations
    sum_logp += n_cont * log(0.5 * 1. / (t_max - t_min))

    return sum_logp

def gen_cdf_using_pdf(double v, double sv, double a, double z, double sz, double t, double st, double err,
                      int N=500, double time=5., int n_st=2, int n_sz=2, bint use_adaptive=1, double simps_err=1e-3,
                      double p_outlier=0, double w_outlier=0):
    """
    generate cdf vector using the pdf
    """
    if (sv < 0) or (a <= 0 ) or (z < 0) or (z > 1) or (sz < 0) or (sz > 1) or (z + sz / 2. > 1) or \
            (z - sz / 2. < 0) or (t - st / 2. < 0) or (t < 0) or (st < 0) or not p_outlier_in_range(p_outlier):
        raise ValueError(
            "at least one of the parameters is out of the support")

    cdef np.ndarray[double, ndim = 1] x = np.linspace(-time, time, 2 * N + 1)
    cdef np.ndarray[double, ndim = 1] cdf_array = np.empty(x.shape[0], dtype=np.double)
    cdef int idx

    # compute pdf on the real line
    cdf_array = pdf_array(x, v, sv, a, z, sz, t, st, err, 0,
                          n_st, n_sz, use_adaptive, simps_err, p_outlier, w_outlier)

    # integrate
    cdf_array[1:] = integrate.cumtrapz(cdf_array)

    # normalize
    cdf_array /= cdf_array[x.shape[0] - 1]

    return x, cdf_array


def split_cdf(np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] data):

    # get length of data
    cdef int N = (len(data) - 1) / 2

    # lower bound is reversed
    cdef np.ndarray[double, ndim = 1] x_lb = -x[:N][::-1]
    cdef np.ndarray[double, ndim = 1] lb = data[:N][::-1]
    # lower bound is cumulative in the wrong direction
    lb = np.cumsum(np.concatenate([np.array([0]), -np.diff(lb)]))

    cdef np.ndarray[double, ndim = 1] x_ub = x[N + 1:]
    cdef np.ndarray[double, ndim = 1] ub = data[N + 1:]
    # ub does not start at 0
    ub -= ub[0]

    return (x_lb, lb, x_ub, ub)


# NEW WITH NN-EXTENSION

#############
# Basic MLP Likelihoods
def wiener_like_nn_mlp(np.ndarray[float, ndim = 1] rt,
                       np.ndarray[float, ndim = 1] response,
                       np.ndarray[float, ndim = 1] params,
                       double p_outlier = 0,
                       double w_outlier = 0,
                       network = None):

    cdef Py_ssize_t size = rt.shape[0]
    cdef Py_ssize_t n_params = params.shape[0]
    cdef float log_p = 0
    cdef float ll_min = -16.11809

    cdef np.ndarray[float, ndim = 2] data = np.zeros((size, n_params + 2), dtype = np.float32)
    data[:, :n_params] = np.tile(params, (size, 1)).astype(np.float32)
    data[:, n_params:] = np.stack([rt, response], axis = 1)

    # Call to network:
    if p_outlier == 0:
        log_p = np.sum(np.core.umath.maximum(network.predict_on_batch(data), ll_min))
    else:
        log_p = np.sum(np.log(np.exp(np.core.umath.maximum(network.predict_on_batch(data), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))

    return log_p

# Basic MLP Likelihoods
def wiener_like_nn_mlp_info(np.ndarray[float, ndim = 1] rt,
                            np.ndarray[float, ndim = 1] response,
                            np.ndarray[float, ndim = 1] params,
                            np.ndarray[float, ndim = 1] upper_bounds,
                            np.ndarray[float, ndim = 1] lower_bounds,
                            double p_outlier = 0,
                            double w_outlier = 0,
                            network = None):

    cdef Py_ssize_t size = rt.shape[0]
    cdef Py_ssize_t n_params = params.shape[0]
    cdef float log_p = 0
    cdef float ll_min = -16.11809
    cdef float[:] upper_bounds_view = upper_bounds
    cdef float[:] lower_bounds_view = lower_bounds
    cdef float[:] params_view = params

    cdef np.ndarray[float, ndim = 2] data = np.zeros((size, n_params + 2), dtype = np.float32)
    data[:, :n_params] = np.tile(params, (size, 1)).astype(np.float32)
    data[:, n_params:] = np.stack([rt, response], axis = 1)

    for i in range(n_params):
        if params_view[i] > upper_bounds_view[i]:
            return -np.inf
        elif params_view[i] < lower_bounds_view[i]:
            return -np.inf

    # Call to network:
    if p_outlier == 0:
        log_p = np.sum(np.core.umath.maximum(network.predict_on_batch(data), ll_min))
    else:
        log_p = np.sum(np.log(np.exp(np.core.umath.maximum(network.predict_on_batch(data), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))

    return log_p

def wiener_like_nn_mlp_pdf(np.ndarray[float, ndim = 1] rt,
                           np.ndarray[float, ndim = 1] response,
                           np.ndarray[float, ndim = 1] params,
                           double p_outlier = 0, 
                           double w_outlier = 0,
                           bint logp = 0,
                           network = None):
    
    cdef Py_ssize_t size = rt.shape[0]
    cdef Py_ssize_t n_params = params.shape[0]

    cdef np.ndarray[float, ndim = 1] log_p = np.zeros(size, dtype = np.float32)
    cdef float ll_min = -16.11809

    cdef np.ndarray[float, ndim = 2] data = np.zeros((size, n_params + 2), dtype = np.float32)
    data[:, :n_params] = np.tile(params, (size, 1)).astype(np.float32)
    data[:, n_params:] = np.stack([rt, response], axis = 1)

    # Call to network:
    if p_outlier == 0: # ddm_model
        log_p = np.squeeze(np.core.umath.maximum(network.predict_on_batch(data), ll_min))
    else: # ddm_model
        log_p = np.squeeze(np.log(np.exp(np.core.umath.maximum(network.predict_on_batch(data), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))
    if logp == 0:
        log_p = np.exp(log_p) # shouldn't be called log_p anymore but no need for an extra array here
    return log_p


################
# Regression style likelihoods: (Can prob simplify and make all mlp likelihoods of this form)

def wiener_like_multi_nn_mlp(np.ndarray[float, ndim = 2] data,
                             double p_outlier = 0, 
                             double w_outlier = 0,
                             network = None):
                             #**kwargs):
    
    cdef float ll_min = -16.11809
    cdef float log_p

    # Call to network:
    if p_outlier == 0: # previous ddm_model
        log_p = np.sum(np.core.umath.maximum(network.predict_on_batch(data), ll_min))
    else:
        log_p = np.sum(np.log(np.exp(np.core.umath.maximum(network.predict_on_batch(data), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))
    return log_p 

def wiener_like_multi_nn_mlp_pdf(np.ndarray[float, ndim = 2] data,
                                 double p_outlier = 0, 
                                 double w_outlier = 0,
                                 network = None):
                                 #**kwargs):
    
    cdef float ll_min = -16.11809
    cdef float log_p

    # Call to network:
    if p_outlier == 0: # previous ddm_model
        log_p = np.squeeze(np.core.umath.maximum(network.predict_on_batch(data), ll_min))
    else:
        log_p = np.squeeze(np.log(np.exp(np.core.umath.maximum(network.predict_on_batch(data), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))
    return log_p

###########
# Basic CNN likelihoods

#def wiener_like_cnn_2(np.ndarray[long, ndim = 1] x, 
#                      np.ndarray[long, ndim = 1] response, 
#                      np.ndarray[float, ndim = 1] parameters,
#                      double p_outlier = 0, 
#                      double w_outlier = 0,
#                      **kwargs):
#
#    cdef Py_ssize_t size = x.shape[0]
#    cdef Py_ssize_t i 
#    cdef float log_p = 0
#    cdef np.ndarray[float, ndim = 2] pred = kwargs['network'](parameters)
#    #log_p = 0
#    
#    for i in range(size):
#        if response[i] == 0:
#            log_p += np.log(pred[0, 2 * x[i]] * (1 - p_outlier) + w_outlier * p_outlier)
#        else: 
#            log_p += np.log(pred[0, 2 * x[i] + 1] * (1 - p_outlier) + w_outlier * p_outlier)
#
#    # Call to network:
#    return log_p
#
#def wiener_pdf_cnn_2(np.ndarray[long, ndim = 1] x, 
#                     np.ndarray[long, ndim = 1] response, 
#                     np.ndarray[float, ndim = 1] parameters,
#                     double p_outlier = 0, 
#                     double w_outlier = 0,
#                     bint logp = 0,
#                     **kwargs):
#
#    cdef Py_ssize_t size = x.shape[0]
#    cdef Py_ssize_t i
#    cdef np.ndarray[float, ndim = 1] log_p = np.zeros(size, dtype = np.float32)
#    cdef np.ndarray[float, ndim = 2] pred = kwargs['network'](parameters)
#    #print(pred.shape)
#    #log_p = 0
#    for i in range(size):
#        if response[i] == 0:
#            log_p[i] += np.log(pred[0, 2 * x[i]] * (1 - p_outlier) + w_outlier * p_outlier)
#        else: 
#            log_p[i] += np.log(pred[0, 2 * x[i] + 1] * (1 - p_outlier) + w_outlier * p_outlier)
#    
#    if logp == 0:
#        log_p = np.exp(log_p)
#
#    # Call to network:
#    return log_p
#
#def wiener_like_reg_cnn_2(np.ndarray[long, ndim = 1] x, 
#                          np.ndarray[long, ndim = 1] response, 
#                          np.ndarray[float, ndim = 2] parameters,
#                          double p_outlier = 0, 
#                          double w_outlier = 0,
#                          bint logp = 0,
#                          **kwargs):
#
#    cdef Py_ssize_t size = x.shape[0]
#    cdef Py_ssize_t i
#    cdef float log_p = 0
#    cdef np.ndarray[float, ndim = 2] pred = kwargs['network'](parameters)
#    #log_p = 0
#    #print(pred.shape)
#    #print(pred)
#    for i in range(size):
#        if response[i] == 0:
#            log_p += np.log(pred[i, 2 * x[i]] * (1 - p_outlier) + w_outlier * p_outlier)
#        else: 
#            log_p += np.log(pred[i, 2 * x[i] + 1] * (1 - p_outlier) + w_outlier * p_outlier)
#    
#    # Call to network:
#    return log_p
#
#