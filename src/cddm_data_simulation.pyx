# Globaly settings for cython
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: initializedcheck=False

# Functions for DDM data simulation
import cython
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log, sqrt, pow, fmax, atan, sin, cos, tan, M_PI, M_PI_2

import numpy as np
cimport numpy as np
from time import time
import inspect
import pickle

DTYPE = np.float32

# Method to draw random samples from a gaussian
cdef float random_uniform():
    cdef float r = rand()
    return r / RAND_MAX

cdef float random_exponential():
    return - log(random_uniform())

cdef float random_stable(float alpha_diff):
    cdef float eta, u, w, x
    # chi = - tan(M_PI_2 * alpha_diff)

    u = M_PI * (random_uniform() - 0.5)
    w = random_exponential()

    if alpha_diff == 1.0:
        eta = M_PI_2 # useless but kept to remain faithful to wikipedia entry
        x = (1.0 / eta) * ((M_PI_2) * tan(u))
    else:
        x = (sin(alpha_diff * u) / (pow(cos(u), 1 / alpha_diff))) * pow(cos(u - (alpha_diff * u)) / w, (1.0 - alpha_diff) / alpha_diff)
    return x

cdef float[:] draw_random_stable(int n, float alpha_diff):
    cdef int i
    cdef float[:] result = np.zeros(n, dtype = DTYPE)

    for i in range(n):
        result[i] = random_stable(alpha_diff)
    return result

cdef float random_gaussian():
    cdef float x1, x2, w
    w = 2.0

    while(w >= 1.0):
        x1 = 2.0 * random_uniform() - 1.0
        x2 = 2.0 * random_uniform() - 1.0
        w = x1 * x1 + x2 * x2

    w = ((-2.0 * log(w)) / w) ** 0.5
    return x1 * w

cdef int sign(float x):
    return (x > 0) - (x < 0)

cdef float csum(float[:] x):
    cdef int i
    cdef int n = x.shape[0]
    cdef float total = 0
    
    for i in range(n):
        total += x[i]
    
    return total

## @cythonboundscheck(False)
cdef void assign_random_gaussian_pair(float[:] out, int assign_ix):
    cdef float x1, x2, w
    w = 2.0

    while(w >= 1.0):
        x1 = (2.0 * random_uniform()) - 1.0
        x2 = (2.0 * random_uniform()) - 1.0
        w = (x1 * x1) + (x2 * x2)

    w = ((-2.0 * log(w)) / w) ** 0.5
    out[assign_ix] = x1 * w
    out[assign_ix + 1] = x2 * w # this was x2 * 2 ..... :0 

# @cythonboundscheck(False)
cdef float[:] draw_gaussian(int n):
    # Draws standard normal variables - need to have the variance rescaled
    cdef int i
    cdef float[:] result = np.zeros(n, dtype=DTYPE)
    for i in range(n // 2):
        assign_random_gaussian_pair(result, i * 2)
    if n % 2 == 1:
        result[n - 1] = random_gaussian()
    return result

# DUMMY TEST SIMULATOR ------------------------------------------------------------------------
# Simulate (rt, choice) tuples from: SIMPLE DDM -----------------------------------------------
# Simplest algorithm

def test(np.ndarray[float, ndim = 1] v, # drift by timestep 'delta_t'
         np.ndarray[float, ndim = 1] a, # boundary separation
         np.ndarray[float, ndim = 1] z,  # between 0 and 1
         np.ndarray[float, ndim = 1] t, # non-decision time
         float s = 1, # noise sigma
         float delta_t = 0.001, # timesteps fraction of seconds
         float max_t = 20, # maximum rt allowed
         int n_samples = 20000, # number of samples considered
         int n_trials = 10,
         ):

    # Param views
    cdef float[:] v_view = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] t_view = t

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t)
    cdef float sqrt_st = delta_t_sqrt * s

    cdef float y, t_particle

    #cdef int n
    cdef Py_ssize_t n, k
    cdef int m = 0
    cdef int num_draws = int(max_t / delta_t + 1)
    cdef float[:] gaussian_values = draw_gaussian(num_draws)
    
    for k in range(n_trials):
        # Loop over samples
        for n in range(n_samples):
            y = z_view[k] * a_view[k] # reset starting point
            t_particle = 0.0 # reset time

            # Random walker
            while y <= a_view[k] and y >= 0 and t <= max_t:
                y += v_view[k] * delta_t + sqrt_st * gaussian_values[m] # update particle position
                t_particle += delta_t
                m += 1
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            # Note that for purposes of consistency with Navarro and Fuss, 
            # the choice corresponding the lower barrier is +1, higher barrier is -1
            rts_view[n, k, 0] = t_particle + t_view[k] # store rt
            choices_view[n, k, 0] = (-1) * sign(y) # store choice
  
    return (rts, choices, {'v': v,
                           'a': a,
                           'z': z,
                           't': t,
                           's': s,
                           'delta_t': delta_t,
                           'max_t': max_t,
                           'n_samples': n_samples,
                           'simulator': 'ddm',
                           'boundary_fun_type': 'constant',
                           'possible_choices': [0, 1]})

# ---------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: Full DDM with flexible bounds --------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def full_ddm_hddm_base(np.ndarray[float, ndim = 1] v, # = 0,
                     np.ndarray[float, ndim = 1] a, # = 1,
                     np.ndarray[float, ndim = 1] z, # = 0.5,
                     np.ndarray[float, ndim = 1] t, # = 0.0,
                     np.ndarray[float, ndim = 1] sz, # = 0.05,
                     np.ndarray[float, ndim = 1] sv, # = 0.1,
                     np.ndarray[float, ndim = 1] st, # = 0.0,
                     float s = 1,
                     float delta_t = 0.001,
                     float max_t = 20,
                     int n_samples = 20000,
                     int n_trials = 1,
                     ):

    # cdef int cov_length = np.max([v.size, a.size, w.size, t.size]).astype(int)

    # Param views
    cdef float[:] v_view  = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] t_view = t
    cdef float[:] sz_view = sz
    cdef float[:] sv_view = sv
    cdef float[:] st_view = st

    # Data-structs for trajectory storage
    traj = np.zeros((int(max_t / delta_t) + 1, 1), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:, :] traj_view = traj

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)

    cdef float y, t_particle, t_tmp
    cdef Py_ssize_t n, ix, k
    cdef Py_ssize_t m = 0
    cdef float drift_increment = 0.0
    cdef float[:] gaussian_values = draw_gaussian(num_draws) 

    # Loop over trials
    for k in range(n_trials): 
        # Loop over samples
        for n in range(n_samples):
            # initialize starting point
            y = (z_view[k] * (a_view[k]))  # reset starting position
            
            # get drift by random displacement of v 
            drift_increment = (v_view[k] + sv_view[k] * gaussian_values[m]) * delta_t
            t_tmp = t_view[k] + (2 * (random_uniform() - 0.5) * st_view[k])
            
            # apply uniform displacement on y
            y += 2 * (random_uniform() - 0.5) * sz_view[k]
            
            # increment m appropriately
            m += 1
            if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0
            
            t_particle = 0.0 # reset time
            ix = 0 # reset boundary index
            
            if n == 0:
                if k == 0:
                    traj_view[0, 0] = y

            # Random walker
            while y >= 0 and y <= a_view[k] and t_particle <= max_t:
                y += drift_increment + (sqrt_st * gaussian_values[m])
                t_particle += delta_t
                ix += 1
                m += 1
                
                if n == 0:
                    if k == 0:
                        traj_view[ix, 0] = y
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            rts_view[n, k, 0] = t_particle + t_tmp # Store rt
            
            if y < 0:
                choices_view[n, k, 0] = 0 # Store choice
            else:
                choices_view[n, k, 0] = 1

    return (rts, choices,  {'v': v,
                            'a': a,
                            'z': z,
                            't': t,
                            'sz': sz,
                            'sv': sv,
                            'st': st,
                            's': s,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'simulator': 'full_ddm_hddm_base',
                            'possible_choices': [0, 1],
                            'trajectory': traj})
# -------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: SIMPLE DDM -----------------------------------------------
# Simplest algorithm
# delete random comment
# delete random comment 2
#@cython.boundscheck(False)
#@cython.wraparound(False)

def ddm(np.ndarray[float, ndim = 1] v, # drift by timestep 'delta_t'
        np.ndarray[float, ndim = 1] a, # boundary separation
        np.ndarray[float, ndim = 1] z,  # between 0 and 1
        np.ndarray[float, ndim = 1] t, # non-decision time
        float s = 1, # noise sigma
        float delta_t = 0.001, # timesteps fraction of seconds
        float max_t = 20, # maximum rt allowed
        int n_samples = 20000, # number of samples considered
        int n_trials = 10,
        ):

    # Param views
    cdef float[:] v_view = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] t_view = t

    # Data-structs for trajectory storage
    traj = np.zeros((int(max_t / delta_t) + 1, 1), dtype = DTYPE)
    traj[:, :] = -999
    cdef float[:, :] traj_view = traj

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t)
    cdef float sqrt_st = delta_t_sqrt * s

    cdef float y, t_particle

    #cdef int n
    cdef Py_ssize_t n, k
    cdef int m = 0
    cdef int num_draws = int(max_t / delta_t + 1)
    cdef float[:] gaussian_values = draw_gaussian(num_draws)
    
    for k in range(n_trials):
        # Loop over samples
        for n in range(n_samples):
            y = z_view[k] * a_view[k] # reset starting point
            t_particle = 0.0 # reset time

            if n == 0:
                if k == 0:
                    traj_view[0, 0] = y
            
            # Random walker
            while y <= a_view[k] and y >= 0 and t_particle <= max_t:
                y += v_view[k] * delta_t + sqrt_st * gaussian_values[m] # update particle position
                t_particle += delta_t
                m += 1

                if n == 0:
                    if k == 0:
                        traj_view[m, 0] = y

                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            # Note that for purposes of consistency with Navarro and Fuss, 
            # the choice corresponding the lower barrier is +1, higher barrier is -1
            rts_view[n, k, 0] = t_particle + t_view[k] # store rt
            if y < 0:
                choices_view[n, k, 0] = 0 # store choice
            else:
                choices_view[n, k, 0] = 1 # store choice
        
    return (rts, choices, {'v': v,
                           'a': a,
                           'z': z,
                           't': t,
                           's': s,
                           'delta_t': delta_t,
                           'max_t': max_t,
                           'n_samples': n_samples,
                           'simulator': 'ddm',
                           'boundary_fun_type': 'constant',
                           'possible_choices': [0, 1],
                           'trajectory': traj})

# Simulate (rt, choice) tuples from: SIMPLE DDM -----------------------------------------------
# Simplest algorithm
# delete random comment
# delete random comment 2
#@cython.boundscheck(False)
#@cython.wraparound(False)

def ddm_cov(np.ndarray[float, ndim = 1] v, # drift by timestep 'delta_t'
            np.ndarray[float, ndim = 1] a, # boundary separation
            np.ndarray[float, ndim = 1] z,  # between 0 and 1
            np.ndarray[float, ndim = 1] t, # non-decision time
            float s = 1, # noise sigma
            float delta_t = 0.001, # timesteps fraction of seconds
            float max_t = 20, # maximum rt allowed
            int n_samples = 1000, # number of samples considered
            int n_trials = 1,
            ):

    #cdef int n_trials = np.max([v.size, a.size, w.size, t.size]).astype(int)

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices
    cdef float[:] v_view = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] t_view = t

    cdef float delta_t_sqrt = sqrt(delta_t)
    cdef float sqrt_st = delta_t_sqrt * s

    cdef float y, t_particle

    # cdef int n
    cdef Py_ssize_t n
    cdef Py_ssize_t k
    cdef int m = 0
    cdef int num_draws = int(max_t / delta_t + 1)
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    # Loop over samples
    for n in range(n_samples):
        for k in range(n_trials):
            y = z_view[k] * a_view[k] # reset starting point
            t_particle = 0.0 # reset time

            # Random walker
            while y <= a[k] and y >= 0 and t_particle <= max_t:
                y += v_view[k] * delta_t + sqrt_st * gaussian_values[m] # update particle position
                t_particle += delta_t
                m += 1
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            # Note that for purposes of consistency with Navarro and Fuss, 
            # the choice corresponding the lower barrier is +1, higher barrier is -1
            rts_view[n, k, 0] = t_particle + t_view[k] # store rt
            choices_view[n, k, 0] = (-1) * sign(y) # store choice

    return (rts, choices, {'v': v,
                            'a': a,
                            'z': z,
                            't': t,
                            's': s,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'simulator': 'ddm',
                            'boundary_fun_type': 'constant',
                            'possible_choices': [-1, 1]})

# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES ------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_flexbound(np.ndarray[float, ndim = 1] v,
                  np.ndarray[float, ndim = 1] a,
                  np.ndarray[float, ndim = 1] z,
                  np.ndarray[float, ndim = 1] t,
                  float s = 1,
                  float delta_t = 0.001,
                  float max_t = 20,
                  int n_samples = 20000,
                  int n_trials = 1,
                  boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                  boundary_multiplicative = True,
                  boundary_params = {},
                  ):

    #cdef int cov_length = np.max([v.size, a.size, w.size, t.size]).astype(int)

    # Param views:
    cdef float[:] v_view  = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] t_view = t

    traj = np.zeros((int(max_t / delta_t) + 1, 1), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:,:] traj_view = traj

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)

    cdef float y, t_particle
    cdef Py_ssize_t n 
    cdef Py_ssize_t ix
    cdef Py_ssize_t m = 0
    cdef Py_ssize_t k
    cdef float[:] gaussian_values = draw_gaussian(num_draws)
    cdef float[:] boundary_view = boundary
    
    # Loop over samples
    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
        if boundary_multiplicative:
            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
    
        for n in range(n_samples):
            y = (-1) * boundary_view[0] + (z_view[k] * 2 * (boundary_view[0]))  # reset starting position 
            t_particle = 0.0 # reset time
            ix = 0 # reset boundary index
            
            # Can improve with less checks
            if n == 0:
                if k == 0:
                    traj_view[0, 0] = y

            while (y >= (-1) * boundary_view[ix]) and (y <= boundary_view[ix]) and (t_particle <= max_t):
                y += (v_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
                t_particle += delta_t
                ix += 1
                m += 1
                
                # Can improve with less checks
                if n == 0:
                    if k == 0:
                        traj_view[ix, 0] = y
                
                # Can improve with less checks
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            rts_view[n, k, 0] = t_particle + t_view[k] # Store rt
            choices_view[n, k, 0] = sign(y) # Store choice
    
    return (rts, choices,  {'v': v,
                            'a': a,
                            'z': z,
                            't': t,
                            's': s,
                            **boundary_params,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'simulator': 'ddm_flexbound',
                            'boundary_fun_type': boundary_fun.__name__,
                            'possible_choices': [-1, 1],
                            'trajectory': traj,
                            'boundary': boundary})
# ----------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES ------------------------------------
def ddm_flexbound_max(float v = 0.0,
                      float a = 1.0,
                      float z = 0.5,
                      float t = 0.5,
                      float s = 1,
                      float delta_t = 0.001,
                      float max_t = 20,
                      int n_samples = 20000,
                      boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                      boundary_multiplicative = True,
                      boundary_params = {},
                      ):

    rts = np.zeros((n_samples, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, 1), dtype = np.intc)

    cdef float[:, :] rts_view = rts
    cdef int[:, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)


    cdef float y, t_particle
    cdef Py_ssize_t n 
    cdef Py_ssize_t ix
    cdef Py_ssize_t m = 0
    cdef Py_ssize_t k
    cdef float[:] gaussian_values = draw_gaussian(num_draws)
    cdef float[:] boundary_view = boundary

    # Loop over samples
    # Precompute boundary evaluations
    if boundary_multiplicative:
        boundary[:] = np.multiply(a, boundary_fun(t = t_s, **boundary_params)).astype(DTYPE)
    else:
        boundary[:] = np.add(a, boundary_fun(t = t_s, **boundary_params)).astype(DTYPE)

    for n in range(n_samples):
        y = (-1) * boundary_view[0] + (z * 2 * (boundary_view[0]))  # reset starting position 
        t_particle = 0.0 # reset time
        ix = 0 # reset boundary index

        # Random walker
        while y >= (-1) * boundary_view[ix] and y <= boundary_view[ix] and t_particle <= max_t:
            y += (v * delta_t) + (sqrt_st * gaussian_values[m])
            t_particle += delta_t
            ix += 1
            m += 1
            
            # Can improve with less checks
            if m == num_draws:
                gaussian_values = draw_gaussian(num_draws)
                m = 0

        rts_view[n, 0] = t_particle + t # Store rt
        choices_view[n, 0] = sign(y) # Store choice
    

    return (rts, choices,  {'v': v,
                            'a': a,
                            'z': z,
                            't': t,
                            's': s,
                            **boundary_params,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'simulator': 'ddm_flexbound',
                            'boundary_fun_type': boundary_fun.__name__,
                            'possible_choices': [-1, 1],
                            'boundary': boundary})
# ----------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: Levy Flight with Flex Bound -------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def levy_flexbound(np.ndarray[float, ndim = 1] v,
                   np.ndarray[float, ndim = 1] a,
                   np.ndarray[float, ndim = 1] z,
                   np.ndarray[float, ndim = 1] alpha_diff,
                   np.ndarray[float, ndim = 1] t,
                   float s = 1, # strictly speaking this is a variance multiplier here, not THE variance !
                   float delta_t = 0.001,
                   float max_t = 20,
                   int n_samples = 20000,
                   int n_trials = 1,
                   boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                   boundary_multiplicative = True,
                   boundary_params = {}
                   ):

    # Param views:
    cdef float[:] v_view  = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] alpha_diff_view = alpha_diff
    cdef float[:] t_view = t

    # Data-struct for trajectory storage
    traj = np.zeros((int(max_t / delta_t) + 1, 1), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:,:] traj_view = traj

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:,:, :] rts_view = rts
    cdef int[:,:, :] choices_view = choices

    cdef float delta_t_alpha # = pow(delta_t, 1.0 / alpha_diff) # correct scalar so we can use standard normal samples for the brownian motion

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    cdef float y, t_particle
    cdef Py_ssize_t n 
    cdef Py_ssize_t ix
    cdef Py_ssize_t k
    cdef Py_ssize_t m = 0
    #cdef int n, ix
    #cdef int m = 0
    cdef float[:] alpha_stable_values = draw_random_stable(num_draws, alpha_diff_view[0])

    for k in range(n_trials):
        delta_t_alpha = s * pow(delta_t, 1.0 / alpha_diff_view[k])
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

        # Precompute boundary evaluations
        if boundary_multiplicative:
            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)

        # Loop over samples
        for n in range(n_samples):
            y = (-1) * boundary_view[0] + (z_view[k] * 2 * (boundary_view[0]))  # reset starting position 
            t_particle = 0.0 # reset time
            ix = 0 # reset boundary index
            if n == 0:
                if k == 0:
                    traj_view[0, 0] = y

            # Random walker
            while y >= (-1) * boundary_view[ix] and y <= boundary_view[ix] and t_particle <= max_t:
                y += (v_view[k] * delta_t) + (delta_t_alpha * alpha_stable_values[m])
                t_particle += delta_t
                ix += 1
                m += 1
                if n == 0:
                    if k == 0:
                        traj_view[ix, 0] = y
                if m == num_draws:
                    alpha_stable_values = draw_random_stable(num_draws, alpha_diff_view[k])
                    m = 0

            rts_view[n, k, 0] = t_particle + t_view[k] # Store rt
            choices_view[n, k, 0] = sign(y) # Store choice
        
    return (rts, choices,  {'v': v,
                            'a': a,
                            'z': z,
                            't': t,
                            'alpha': alpha_diff,
                            's': s,
                            **boundary_params,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'simulator': 'levy_flexbound',
                            'boundary_fun_type': boundary_fun.__name__,
                            'possible_choices': [-1, 1],
                            'trajectory': traj,
                            'boundary': boundary})
# -------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: Full DDM with flexible bounds --------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def full_ddm(np.ndarray[float, ndim = 1] v, # = 0,
             np.ndarray[float, ndim = 1] a, # = 1,
             np.ndarray[float, ndim = 1] z, # = 0.5,
             np.ndarray[float, ndim = 1] t, # = 0.0,
             np.ndarray[float, ndim = 1] sz, # = 0.05,
             np.ndarray[float, ndim = 1] sv, # = 0.1,
             np.ndarray[float, ndim = 1] st, # = 0.0,
             float s = 1,
             float delta_t = 0.001,
             float max_t = 20,
             int n_samples = 20000,
             int n_trials = 1,
             boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
             boundary_multiplicative = True,
             boundary_params = {}
             ):

    # cdef int cov_length = np.max([v.size, a.size, w.size, t.size]).astype(int)

    # Param views
    cdef float[:] v_view  = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] t_view = t
    cdef float[:] sz_view = sz
    cdef float[:] sv_view = sv
    cdef float[:] st_view = st

    # Data-structs for trajectory storage
    traj = np.zeros((int(max_t / delta_t) + 1, 1), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:, :] traj_view = traj

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    cdef float y, t_particle, t_tmp
    cdef Py_ssize_t n, ix, k
    cdef Py_ssize_t m = 0
    cdef float drift_increment = 0.0
    cdef float[:] gaussian_values = draw_gaussian(num_draws) 

    # Loop over trials
    for k in range(n_trials):
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

        # Precompute boundary evaluations
        if boundary_multiplicative:
            boundary_view[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary_view[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        
        # Loop over samples
        for n in range(n_samples):
            # initialize starting point
            y = ((-1) * boundary_view[0]) + (z_view[k] * 2.0 * (boundary_view[0]))  # reset starting position
            
            # get drift by random displacement of v 
            drift_increment = (v_view[k] + sv_view[k] * gaussian_values[m]) * delta_t
            t_tmp = t_view[k] + (2 * (random_uniform() - 0.5) * st_view[k])
            
            # apply uniform displacement on y
            y += 2 * (random_uniform() - 0.5) * sz_view[k]
            
            # increment m appropriately
            m += 1
            if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0
            
            t_particle = 0.0 # reset time
            ix = 0 # reset boundary index
            
            if n == 0:
                if k == 0:
                    traj_view[0, 0] = y

            # Random walker
            while y >= (-1) * boundary_view[ix] and y <= boundary_view[ix] and t_particle <= max_t:
                y += drift_increment + (sqrt_st * gaussian_values[m])
                t_particle += delta_t
                ix += 1
                m += 1
                
                if n == 0:
                    if k == 0:
                        traj_view[ix, 0] = y
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            rts_view[n, k, 0] = t_particle + t_tmp # Store rt
            choices_view[n, k, 0] = np.sign(y) # Store choice

    return (rts, choices,  {'v': v,
                            'a': a,
                            'z': z,
                            't': t,
                            'sz': sz,
                            'sv': sv,
                            'st': st,
                            's': s,
                            **boundary_params,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'simulator': 'full_ddm',
                            'boundary_fun_type': boundary_fun.__name__,
                            'possible_choices': [-1, 1],
                            'trajectory': traj,
                            'boundary': boundary})

# -------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: Full DDM with flexible bounds --------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_sdv(np.ndarray[float, ndim = 1] v,
            np.ndarray[float, ndim = 1] a,
            np.ndarray[float, ndim = 1] z,
            np.ndarray[float, ndim = 1] t,
            np.ndarray[float, ndim = 1] sv,
            float s = 1,
            float delta_t = 0.001,
            float max_t = 20,
            int n_samples = 20000,
            int n_trials = 1,
            boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
            boundary_multiplicative = True,
            boundary_params = {}
            ):

    # Data-structs for trajectory storage
    traj = np.zeros((int(max_t / delta_t) + 1, 1), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:,:] traj_view = traj

    # Param views
    cdef float[:] v_view  = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] t_view = t
    cdef float[:] sv_view = sv
    
    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    # Precompute boundary evaluations
    if boundary_multiplicative:
        boundary_view[:] = np.multiply(a, boundary_fun(t = t_s, **boundary_params)).astype(DTYPE)
    else:
        boundary_view[:] = np.add(a, boundary_fun(t = t_s, **boundary_params)).astype(DTYPE)
    
    cdef float y, t_particle
    cdef Py_ssize_t n, ix, k
    cdef Py_ssize_t m = 0
    cdef float drift_increment = 0.0
    cdef float[:] gaussian_values = draw_gaussian(num_draws) 

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

        if boundary_multiplicative:
            boundary_view[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary_view[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)

        # Loop over samples
        for n in range(n_samples):
            # initialize starting point
            y = ((-1) * boundary_view[0]) + (z_view[k] * 2.0 * (boundary_view[0]))  # reset starting position
            
            # get drift by random displacement of v 
            drift_increment = (v_view[k] + sv_view[k] * gaussian_values[m]) * delta_t
            
            # increment m appropriately
            m += 1
            if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0
            
            t_particle = 0.0 # reset time
            ix = 0 # reset boundary index
            
            if n == 0:
                if k == 0:
                    traj_view[0, 0] = y

            # Random walker
            while y >= (-1) * boundary_view[ix] and y <= boundary_view[ix] and t_particle <= max_t:
                y += drift_increment + (sqrt_st * gaussian_values[m])
                t_particle += delta_t
                ix += 1
                m += 1
                
                if n == 0:
                    if k == 0:
                        traj_view[ix, 0] = y

                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0


            rts_view[n, k, 0] = t_particle + t_view[k] # Store rt
            choices_view[n, k, 0] = np.sign(y) # Store choice


    return (rts, choices,  {'v': v,
                            'a': a,
                            'z': z,
                            't': t,
                            'sv': sv,
                            's': s,
                            **boundary_params,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'simulator': 'ddm_sdv',
                            'boundary_fun_type': boundary_fun.__name__,
                            'possible_choices': [-1, 1],
                            'trajectory': traj,
                            'boundary': boundary})

# -------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES AND FLEXIBLE SLOPE -----------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_flex(np.ndarray[float, ndim = 1] v,
             np.ndarray[float, ndim = 1] a,
             np.ndarray[float, ndim = 1] z,
             np.ndarray[float, ndim = 1] t,
             float s = 1,
             float delta_t = 0.001,
             float max_t = 20,
             int n_samples = 20000,
             int n_trials = 1,
             boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
             drift_fun = None,
             boundary_multiplicative = True,
             boundary_params = {},
             drift_params = {},
             ):

    # Param views:
    cdef float[:] v_view  = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] t_view = t

    traj = np.zeros((int(max_t / delta_t) + 1, 1), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:,:] traj_view = traj

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    drift = np.zeros(t_s.shape, dtype = DTYPE)

    cdef float y, t_particle
    cdef Py_ssize_t n 
    cdef Py_ssize_t ix
    cdef Py_ssize_t m = 0
    cdef Py_ssize_t k
    cdef float[:] gaussian_values = draw_gaussian(num_draws)
    cdef float[:] boundary_view = boundary
    cdef float[:] drift_view = drift

    # Loop over samples
    for k in range(n_trials):
        # Precompute boundary evaluations and drift evaluations
        
        # Drift
        drift_params_tmp = {key: drift_params[key][k] for key in drift_params.keys()}
        drift[:] = np.add(v_view[k], drift_fun(t = t_s, **drift_params_tmp)).astype(DTYPE)

        # Boundary
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
        if boundary_multiplicative:
            # print('passed')
            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)

        for n in range(n_samples):
            y = (-1) * boundary_view[0] + (z_view[k] * 2 * (boundary_view[0]))  # reset starting position 
            t_particle = 0.0 # reset time
            ix = 0 # reset boundary index
            
            # Can improve with less checks
            if n == 0:
                if k == 0:
                    traj_view[0, 0] = y

            # Random walker
            while (y >= (-1) * boundary_view[ix]) and (y <= boundary_view[ix]) and (t_particle <= max_t):
                y += (drift_view[ix] * delta_t) + (sqrt_st * gaussian_values[m])
                t_particle += delta_t
                ix += 1
                m += 1
                
                # Can improve with less checks
                if n == 0:
                    if k == 0:
                        traj_view[ix, 0] = y
                
                # Can improve with less checks
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            rts_view[n, k, 0] = t_particle + t_view[k] # Store rt
            choices_view[n, k, 0] = sign(y) # Store choice
    
    return (rts, choices,  {'v': v,
                            'a': a,
                            'z': z,
                            't': t,
                            's': s,
                            **boundary_params,
                            **drift_params,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'simulator': 'ddm_flex',
                            'boundary_fun_type': boundary_fun.__name__,
                            'drift_fun_type': boundary_fun.__name__,
                            'possible_choices': [-1, 1],
                            'trajectory': traj,
                            'drift': drift,
                            'boundary': boundary})
# ----------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: Onstein-Uhlenbeck with flexible bounds -----------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ornstein_uhlenbeck(np.ndarray[float, ndim = 1] v, # drift parameter
                       np.ndarray[float, ndim = 1] a, # initial boundary separation
                       np.ndarray[float, ndim = 1] z, # starting point bias
                       np.ndarray[float, ndim = 1] g, # decay parameter
                       np.ndarray[float, ndim = 1] t,
                       float s = 1, # standard deviation
                       float delta_t = 0.001, # size of timestep
                       float max_t = 20, # maximal time in trial
                       int n_samples = 20000, # number of samples from process
                       int n_trials = 1,
                       boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                       boundary_multiplicative = True,
                       boundary_params = {}
                      ):

    # Data-structs for trajectory storage
    traj = np.zeros((int(max_t / delta_t) + 1, 1), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:,:] traj_view = traj

    # Param views
    cdef float[:] v_view  = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] g_view = g
    cdef float[:] t_view = t

    # Initializations
    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE) # rt storage
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc) # choice storage

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = np.sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = s * delta_t_sqrt

    # Boundary Storage
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    cdef float y, t_particle
    cdef Py_ssize_t n, ix, k
    cdef Py_ssize_t m = 0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

        # Precompute boundary evaluations
        if boundary_multiplicative:
            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
    
        # Loop over samples
        for n in range(n_samples):
            y = (-1) * boundary_view[0] + (z_view[k] * 2 * boundary_view[0])
            t_particle = 0.0
            ix = 0

            if n == 0:
                if k == 0:
                    traj_view[0, 0] = y

            # Random walker
            while y >= (-1) * boundary_view[ix] and y <= boundary_view[ix] and t_particle <= max_t:
                y += ((v_view[k] - (g_view[k] * y)) * delta_t) + sqrt_st * gaussian_values[m]
                t_particle += delta_t
                ix += 1
                m += 1

                if n == 0:
                    if k == 0:
                        traj_view[ix, 0] = y

                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            rts_view[n, k, 0] = t_particle + t_view[k] 
            choices_view[n, k, 0] = sign(y)

    return (rts, choices, {'v': v,
                           'a': a,
                           'z': z,
                           'g': g,
                           't': t,
                           's': s,
                           **boundary_params,
                           'delta_t': delta_t,
                           'max_t': max_t,
                           'n_samples': n_samples,
                           'simulator': 'ornstein_uhlenbeck',
                           'boundary_fun_type': boundary_fun.__name__,
                           'possible_choices': [-1, 1],
                           'trajectory': traj,
                           'boundary': boundary})
# --------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: RACE MODEL WITH N SAMPLES ----------------------------------

# Check if any of the particles in the race model have crossed
# @cythonboundscheck(False)
# @cythonwraparound(False)

# Function that checks boundary crossing of particles
cdef bint check_finished(float[:] particles, float boundary, int n):
    cdef int i # ,n
    #n = particles.shape[0]
    for i in range(n):
        if particles[i] > boundary:
            return True
    return False

#def test_check():
#    # Quick sanity check for the check_finished function
#    temp = np.random.normal(0,1, 10).astype(DTYPE)
#    cdef float[:] temp_view = temp
#    start = time()
#    [check_finished(temp_view, 3) for _ in range(1000000)]
#    print(check_finished(temp_view, 3))
#    end = time()
#    print("cython check: {}".format(start - end))
#    start = time()
#    [(temp > 3).any() for _ in range(1000000)]
#    end = time()
#    print("numpy check: {}".format(start - end))

# @cythonboundscheck(False)
# @cythonwraparound(False)
def race_model(np.ndarray[float, ndim = 2] v,  # np.array expected, one column of floats
               np.ndarray[float, ndim = 2] a, # initial boundary separation
               np.ndarray[float, ndim = 2] z, # np.array expected, one column of floats
               np.ndarray[float, ndim = 2] t, # for now we we don't allow t by choice
               np.ndarray[float, ndim = 2] s, # np.array expected, one column of floats
               float delta_t = 0.001, # time increment step
               float max_t = 20, # maximum rt allowed
               int n_samples = 2000, 
               int n_trials = 1,
               boundary_fun = None,
               boundary_multiplicative = True,
               boundary_params = {}):

    # Param views
    cdef float[:, :] v_view = v
    cdef float[:, :] z_view = z
    cdef float[:, :] a_view = a
    cdef float[:, :] t_view = t
    cdef float[:, :] s_view = s

    cdef float delta_t_sqrt = sqrt(delta_t)
    sqrt_st = delta_t_sqrt * s
    cdef float[:, :] sqrt_st_view = sqrt_st

    cdef int n_particles = v.shape[1]
    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    cdef float[:, :, :] rts_view = rts
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
    cdef int[:, :, :] choices_view = choices
    
    particles = np.zeros((n_particles), dtype = DTYPE)
    cdef float [:] particles_view = particles


    # TD: Add Trajectory
    traj = np.zeros((int(max_t / delta_t) + 1, n_particles), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:, :] traj_view = traj    

    # Boundary storage
    cdef int num_steps = int((max_t / delta_t) + 1)

    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    # Initialize variables needed for for loop 
    cdef float t_particle
    cdef Py_ssize_t n, ix, j, k
    cdef Py_ssize_t m = 0

    cdef int num_draws = num_steps * n_particles
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

        # Precompute boundary evaluations
        if boundary_multiplicative:
            boundary[:] = np.multiply(a_view[k, 0], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary[:] = np.add(a_view[k, 0], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
    
        # Loop over samples
        for n in range(n_samples):
            for j in range(n_particles):
                particles_view[j] = z_view[k, j] * boundary_view[0] # Reset particle starting points
            
            t_particle = 0.0 # reset time
            ix = 0

            if n == 0:
                if k == 0:
                    for j in range(n_particles):
                        traj_view[0, j] = particles[j]

            # Random walker
            while not check_finished(particles_view, boundary_view[ix], n_particles) and t_particle <= max_t:
                for j in range(n_particles):
                    particles_view[j] += (v_view[k, j] * delta_t) + sqrt_st_view[k, j] * gaussian_values[m]
                    m += 1
                    if m == num_draws:
                        m = 0
                        gaussian_values = draw_gaussian(num_draws)
                t_particle += delta_t
                ix += 1
                if n == 0:
                    if k == 0:
                        for j in range(n_particles):
                            traj_view[ix, j] = particles[j]

            choices_view[n, k, 0] = np.argmax(particles)
            #rts_view[n, 0] = t + t[choices_view[n, 0]]
            rts_view[n , k, 0] = t_particle + t[k, 0] # for now no t per choice option

        # Create some dics
        v_dict = {}
        z_dict = {}
        #t_dict = {}
        for i in range(n_particles):
            v_dict['v' + str(i)] = v[:, i]
            z_dict['z' + str(i)] = z[:, i]
            #t_dict['t_' + str(i)] = t[i] # for now no t by choice


    return (rts, choices, {**v_dict,
                        'a': a[:, 0], 
                        **z_dict, # if z's are different
                        'z': z[:, 0], # single z if z's all the same
                        't': t[:, 0],
                        # **t_dict, # for now no t by choice
                        's': s,
                        **boundary_params,
                        'delta_t': delta_t,
                        'max_t': max_t,
                        'n_samples': n_samples,
                        'simulator': 'race_model',
                        'boundary_fun_type': boundary_fun.__name__,
                        'possible_choices': list(np.arange(0, n_particles, 1)),
                        'trajectory': traj,
                        'boundary': boundary})
    # -------------------------------------------------------------------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)

# Simulate (rt, choice) tuples from: Leaky Competing Accumulator Model -----------------------------
def lca(np.ndarray[float, ndim = 2] v, # drift parameters (np.array expect: one column of floats)
        np.ndarray[float, ndim = 2] a, # criterion height
        np.ndarray[float, ndim = 2] z, # initial bias parameters (np.array expect: one column of floats)
        np.ndarray[float, ndim = 2] g, # decay parameter
        np.ndarray[float, ndim = 2] b, # inhibition parameter
        np.ndarray[float, ndim = 2] t,
        np.ndarray[float, ndim = 2] s, # variance (can be one value or np.array of size as v and w)
        float delta_t = 0.001, # time-step size in simulator
        float max_t = 20, # maximal time
        int n_samples = 2000, # number of samples to produce
        int n_trials = 1,
        boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
        boundary_multiplicative = True,
        boundary_params = {}):


    # Param views
    cdef float[:, :] v_view = v
    cdef float[:, :] a_view = a
    cdef float[:, :] z_view = z
    cdef float[:, :] g_view = g
    cdef float[:, :] b_view = b
    cdef float[:, :] t_view = t
    cdef float[:, :] s_view = s

    # Trajectory
    cdef int n_particles = v.shape[1]
    traj = np.zeros((int(max_t / delta_t) + 1, n_particles), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:, :] traj_view = traj

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    cdef float[:, :, :] rts_view = rts
    
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
    cdef int[:, :, :] choices_view = choices

    particles = np.zeros(n_particles, dtype = DTYPE)
    cdef float[:] particles_view = particles
    
    particles_reduced_sum = np.zeros(n_particles, dtype = DTYPE)
    cdef float[:] particles_reduced_sum_view = particles_reduced_sum
    
    cdef float delta_t_sqrt = sqrt(delta_t)
    sqrt_st = s * delta_t_sqrt
    cdef float[:, :] sqrt_st_view = sqrt_st
    
    cdef Py_ssize_t n, i, ix, k
    cdef Py_ssize_t m = 0
    cdef float t_par, particles_sum
    
    # Boundary storage                                                             
    cdef int num_steps = int((max_t / delta_t) + 2)

    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    cdef int num_draws = num_steps * n_particles
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

        # Precompute boundary evaluations
        if boundary_multiplicative:
            # print(a)
            boundary[:] = np.multiply(a_view[k, 0], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            # print(a)
            boundary[:] = np.add(a_view[k, 0], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)

        for n in range(n_samples):
            # Reset particle starting points
            for i in range(n_particles):
                particles_view[i] = z_view[k, i] * boundary_view[0]
            
            t_particle = 0.0 # reset time
            ix = 0 # reset boundary index

            if n == 0:
                if k == 0:
                    for i in range(n_particles):
                        traj_view[0, i] = particles[i]

            while not check_finished(particles_view, boundary_view[ix], n_particles) and t_particle <= max_t:
                # calculate current sum over particle positions
                particles_sum = csum(particles_view)
                
                # update particle positions 
                for i in range(n_particles):
                    particles_reduced_sum_view[i] = (- 1) * particles_view[i] + particles_sum
                    particles_view[i] += ((v_view[k, i] - (g_view[k, 0] * particles_view[i]) - \
                            (b_view[k, 0] * particles_reduced_sum_view[i])) * delta_t) + (sqrt_st_view[k, i] * gaussian_values[m])
                    particles_view[i] = fmax(0.0, particles_view[i])
                    m += 1

                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0
                
                t_particle += delta_t # increment time
                ix += 1 # increment boundary index

                if n == 0:
                    if k == 0:
                        for i in range(n_particles):
                            traj_view[ix, i] = particles[i]
        
            choices_view[n, k, 0] = np.argmax(particles) # store choices for sample n
            rts_view[n, k, 0] = t_particle + t_view[k, 0] # t[choices_view[n, 0]] # store reaction time for sample n
        
    # Create some dics
    v_dict = {}
    z_dict = {}
    #t_dict = {}
    
    for i in range(n_particles):
        v_dict['v' + str(i)] = v[:, i]
        z_dict['z_' + str(i)] = z[:, i]

    return (rts, choices, {**v_dict,
                           'a': a[:, 0],
                           **z_dict, # --> if different z's
                           'z': z[:, 0], # z --> if all z_s the same , 
                           'g': g[:, 0],
                           'b': b[:, 0],
                           't': t[:, 0],
                           's': s,
                           **boundary_params,
                           'delta_t': delta_t,
                           'max_t': max_t,
                           'n_samples': n_samples,
                           'simulator' : 'lca',
                           'boundary_fun_type': boundary_fun.__name__,
                           'possible_choices': list(np.arange(0, n_particles, 1)),
                           'trajectory': traj,
                           'boundary': boundary})

# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES ------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_flexbound_seq2(np.ndarray[float, ndim = 1] v_h,
                       np.ndarray[float, ndim = 1] v_l_1,
                       np.ndarray[float, ndim = 1] v_l_2,
                       np.ndarray[float, ndim = 1] a,
                       np.ndarray[float, ndim = 1] z_h,
                       np.ndarray[float, ndim = 1] z_l_1,
                       np.ndarray[float, ndim = 1] z_l_2,
                       np.ndarray[float, ndim = 1] t,
                       float s = 1,
                       float delta_t = 0.001,
                       float max_t = 20,
                       int n_samples = 20000,
                       int n_trials = 1,
                       print_info = True,
                       boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                       boundary_multiplicative = True,
                       boundary_params = {}
                       ):

    # Param views
    cdef float[:] v_h_view = v_h
    cdef float[:] v_l_1_view = v_l_1
    cdef float[:] v_l_2_view = v_l_2
    cdef float[:] a_view = a
    cdef float[:] z_h_view = z_h
    cdef float[:] z_l_1_view = z_l_1
    cdef float[:] z_l_2_view = z_l_2
    cdef float[:] t_view = t

    # Trajectory
    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    cdef float y_h, t_particle, y_l
    cdef Py_ssize_t n, ix, k
    cdef Py_ssize_t m = 0
    #cdef Py_ssize_t traj_id
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

        # Precompute boundary evaluations
        if boundary_multiplicative:
            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
    
        # Loop over samples
        for n in range(n_samples):
            t_particle = 0.0 # reset time
            ix = 0 # reset boundary index

            # Random walker 1
            y_h = (-1) * boundary_view[0] + (z_h_view[k] * 2 * (boundary_view[0]))  # reset starting position 

            while y_h >= (-1) * boundary_view[ix] and y_h <= boundary_view[ix] and t_particle <= max_t:
                y_h += (v_h_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
                t_particle += delta_t
                ix += 1
                m += 1
                
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            # If we are already at maximum t, to generate a choice we just sample from a bernoulli
            if t_particle >= max_t:
                if random_uniform() > 0.5:
                    choices_view[n, k, 0] = choices_view[n, k, 0] + 1
            else:
                if sign(y_h) < 0: # Store intermediate choice
                    choices_view[n, k, 0] = 0 
                    
                    # In case boundary is negative already, we flip a coin with bias determined by w_l_ parameter
                    if boundary_view[ix] <= 0:
                        if random_uniform() < z_l_1_view[k]:
                            choices_view[n, k, 0] += 1
                    else:
                        y_l = (-1) * boundary_view[ix] + (z_l_1_view[k] * 2 * (boundary_view[ix])) 
                        v_l = v_l_1_view[k]
                        #traj_id = 1
                else:
                    choices_view[n, k, 0] = 2
                    
                    # In case boundary is negative already, we flip a coin with bias determined by w_l_ parameter
                    if boundary_view[ix] <= 0:
                        if random_uniform() < z_l_2_view[k]:
                            choices_view[n, k, 0] += 1
                    else:
                        y_l = (-1) * boundary_view[ix] + (z_l_2_view[k] * 2 * (boundary_view[ix])) 
                        v_l = v_l_2_view[k]
                        #traj_id = 2

            # Random walker 2
            while y_l >= (-1) * boundary_view[ix] and y_l <= boundary_view[ix] and t_particle <= max_t:
                y_l += (v_l * delta_t) + (sqrt_st * gaussian_values[m])
                t_particle += delta_t
                ix += 1
                m += 1
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            rts_view[n, k, 0] = t_particle + t_view[k]
            if sign(y_l) >= 0: # store choice update
                choices_view[n, k, 0] += 1

    return (rts, choices,  {'vh': v_h,
                            'vl1': v_l_1,
                            'vl2': v_l_2,
                            'a': a,
                            'zh': z_h,
                            'zl1': z_l_1,
                            'zl2': z_l_2,
                            't': t,
                            's': s,
                            **boundary_params,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'simulator': 'ddm_flexbound',
                            'boundary_fun_type': boundary_fun.__name__,
                            'possible_choices': [0, 1, 2, 3],
                            'trajectory': 'This simulator does not yet allow for trajectory simulation',
                            'boundary': boundary})
# -----------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES ------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_flexbound_par2(np.ndarray[float, ndim = 1] v_h,
                       np.ndarray[float, ndim = 1] v_l_1,
                       np.ndarray[float, ndim = 1] v_l_2,
                       np.ndarray[float, ndim = 1] a,
                       np.ndarray[float, ndim = 1] z_h,
                       np.ndarray[float, ndim = 1] z_l_1,
                       np.ndarray[float, ndim = 1] z_l_2,
                       np.ndarray[float, ndim = 1] t,
                       float s = 1,
                       float delta_t = 0.001,
                       float max_t = 20,
                       int n_samples = 20000,
                       int n_trials = 1,
                       print_info = True,
                       boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                       boundary_multiplicative = True,
                       boundary_params = {}
                       ):


    # Param views
    cdef float[:] v_h_view = v_h
    cdef float[:] v_l_1_view = v_l_1
    cdef float[:] v_l_2_view = v_l_2
    cdef float[:] a_view = a
    cdef float[:] z_h_view = z_h
    cdef float[:] z_l_1_view = z_l_1
    cdef float[:] z_l_2_view = z_l_2
    cdef float[:] t_view = t

    # TD: Add trajectory --> Tricky here because the simulator is optimized to include only two instead of three particles (high dimension choice determines which low dimension choice will matter for ultimate choice)

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    cdef float y_h, y_l, v_l, t_h, t_l
    cdef Py_ssize_t n, ix, k
    cdef Py_ssize_t m = 0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

        # Precompute boundary evaluations
        if boundary_multiplicative:
            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        
        # Loop over samples
        for n in range(n_samples):
            t_h = 0.0 # reset time high dimension
            t_l = 0.0 # reset time low dimension
            ix = 0 # reset boundary index

            # Initialize walkers
            y_h = (-1) * boundary_view[0] + (z_h_view[k] * 2 * (boundary_view[0])) 

            # Random walks until y_h hits bound
            while y_h >= (-1) * boundary_view[ix] and y_h <= boundary_view[ix] and t_h <= max_t:
                y_h += (v_h_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
                t_h += delta_t
                ix += 1
                m += 1
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            if sign(y_h) < 0: # Store intermediate choice
                choices_view[n, k, 0] = 0 
                y_l = (-1) * boundary_view[0] + (z_l_1_view[k] * 2 * (boundary_view[0])) 
                v_l = v_l_1_view[k]
            
            else:
                choices_view[n, k, 0] = 2
                y_l = (-1) * boundary_view[0] + (z_l_2_view[k] * 2 * (boundary_view[0])) 
                v_l = v_l_2_view[k]

            # Random walks until the y_l corresponding to y_h hits bound
            ix = 0
            while y_l >= (-1) * boundary_view[ix] and y_l <= boundary_view[ix] and t_l <= max_t:
                y_l += (v_l * delta_t) + (sqrt_st * gaussian_values[m])
                t_l += delta_t
                ix += 1
                m += 1
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            rts_view[n, k, 0] = fmax(t_h, t_l) + t_view[k]

            if sign(y_l) >= 0: # store choice update
                choices_view[n, k, 0] = choices_view[n, k, 0] + 1

    return (rts, choices,  {'vh': v_h,
                            'vl1': v_l_1,
                            'vl2': v_l_2,
                            'a': a,
                            'zh': z_h,
                            'zl1': z_l_1,
                            'zl2': z_l_2,
                            't': t,
                            's': s,
                            **boundary_params,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'simulator': 'ddm_flexbound',
                            'boundary_fun_type': boundary_fun.__name__,
                            'possible_choices': [0, 1, 2, 3],
                            'trajectory': 'This simulator does not yet allow for trajectory simulation',
                            'boundary': boundary})
# -----------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES ------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_flexbound_mic2_adj(np.ndarray[float, ndim = 1] v_h, 
                           np.ndarray[float, ndim = 1] v_l_1,
                           np.ndarray[float, ndim = 1] v_l_2,
                           np.ndarray[float, ndim = 1] a,
                           np.ndarray[float, ndim = 1] z_h,
                           np.ndarray[float, ndim = 1] z_l_1,
                           np.ndarray[float, ndim = 1] z_l_2,
                           np.ndarray[float, ndim = 1] d, # d for 'dampen' effect on drift parameter
                           np.ndarray[float, ndim = 1] t,
                           float s = 1,
                           float delta_t = 0.001,
                           float max_t = 20,
                           int n_samples = 20000,
                           int n_trials = 1,
                           print_info = True,
                           boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                           boundary_multiplicative = True,
                           boundary_params = {}
                           ):
    # Param views
    cdef float[:] v_h_view = v_h # slope corresponding to high dimension slope
    cdef float[:] v_l_1_view = v_l_1 # slope corresponding to irrelevant low dimension
    cdef float[:] v_l_2_view = v_l_2 # slope corresponding to correct low dimension
    cdef float[:] a_view = a
    cdef float[:] z_h_view = z_h # bias corresponding to high dimension 
    cdef float[:] z_l_1_view = z_l_1 # bias corresponding to irrelevant low dimension
    cdef float[:] z_l_2_view = z_l_2 # bias corresponding to correct low dimension
    cdef float[:] d_view = d
    cdef float[:] t_view = t

    # TD: Add trajectory --> same issue as with par2 model above... might need to make a separate simulator for trajectories
    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    # Y particle trace
    bias_trace = np.zeros(num_draws, dtype = DTYPE) # tracks the y_h position in the accumulator (normalized to be between [0, 1])
    cdef float[:] bias_trace_view = bias_trace

    cdef float y_h, y_l, v_l, t_h, t_l
    cdef Py_ssize_t n, ix, ix_tmp, k
    cdef Py_ssize_t m = 0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

        # Precompute boundary evaluations
        if boundary_multiplicative:
            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
    
        # Loop over samples
        for n in range(n_samples):
            choices_view[n, k, 0] = 0 # reset choice
            t_h = 0 # reset time high dimension
            t_l = 0 # reset time low dimension
            ix = 0 # reset boundary index

            # Initialize walkers
            y_h = (-1) * boundary_view[0] + (z_h_view[k] * 2 * (boundary_view[0])) 
            bias_trace_view[0] = ((y_h + boundary_view[0]) / (2 * boundary_view[0]))

            # Random walks until y_h hits bound
            while (y_h >= ((-1) * boundary_view[ix])) and ((y_h <= boundary_view[ix])) and (t_h <= max_t):
                y_h += (v_h_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
                bias_trace_view[ix] = ((y_h + boundary_view[ix]) / (2 * boundary_view[ix]))
                t_h += delta_t
                ix += 1
                m += 1
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> y_h + boundary_view[ix] / 2 * boundary_view[ix] = 1 --> choices_view[n, k, 0] adds two     
            # y at lower bound --> y_h + boundary_view[ix] / 2 * boundary_view[ix] = 0 --> choices_view[n, k, 0] stays the same 
            if random_uniform() <= ((y_h + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 2
           
            if choices_view[n, k, 0] == 2:
                y_l = (- 1) * boundary_view[0] + (z_l_2_view[k] * 2 * (boundary_view[0])) 
                v_l = v_l_2_view[k]

                # Fill bias trace until max_rt reached
                ix_tmp = ix + 1
                while ix_tmp < num_draws:
                    bias_trace_view[ix_tmp] = 1.0
                    ix_tmp += 1

            else: # Store intermediate choice
                y_l = (- 1) * boundary_view[0] + (z_l_1_view[k] * 2 * (boundary_view[0])) 
                v_l = v_l_1_view[k]

                # Fill bias trace until max_rt reached
                ix_tmp = ix + 1
                while ix_tmp < num_draws:
                    bias_trace_view[ix_tmp] = 0.0
                    ix_tmp += 1

                # We need to reverse the bias_trace if we took the lower choice
                ix_tmp = 0
                while ix_tmp < num_draws:
                    bias_trace_view[ix_tmp] = 1.0 - bias_trace_view[ix_tmp]
                    ix_tmp += 1
            
            # Random walks until the y_l corresponding to high dimensional choice hits bound
            ix = 0
            while (y_l >= ((-1) * boundary_view[ix])) and (y_l <= boundary_view[ix]) and (t_l <= max_t):
                # If high-dim choice has not been taken --> apply the slope discount according to parameter d and the position of y_h (normalized and sitting in bias_trace_view)
                if (bias_trace_view[ix] < 1) and (bias_trace_view[ix] > 0): # Note I think > 0 part is unnecessary 
                    y_l += ((v_l * bias_trace_view[ix] * d_view[k]) * delta_t)
                else: # If high-dim choice already taken, apply low dim slope undiscounted
                    y_l += (v_l * delta_t)
                
                y_l += (sqrt_st * gaussian_values[m]) # add noise
                
                t_l += delta_t # add time
                ix += 1
                m += 1
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            rts_view[n, k, 0] = fmax(t_h, t_l) + t_view[k]

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> y_l + boundary_view[ix] / 2 * boundary_view[ix] = 1 --> choices_view[n, k, 0] adds one     
            # y at lower bound --> y_l + boundary_view[ix] / 2 * boundary_view[ix] = 0 --> choices_view[n, k, 0] stays the same 
            if random_uniform() <= ((y_l + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 1

    return (rts, choices, {'vh': v_h,
                           'vl1': v_l_1,
                           'vl2': v_l_2,
                           'a': a,
                           'zh': z_h,
                           'zl1': z_l_1,
                           'zl2': z_l_2,
                           'd': d,
                           't': t,
                           's': s,
                           **boundary_params,
                           'delta_t': delta_t,
                           'max_t': max_t,
                           'n_samples': n_samples,
                           'simulator': 'ddm_flexbound',
                           'boundary_fun_type': boundary_fun.__name__,
                           'possible_choices': [0, 1, 2, 3],
                           'trajectory': 'This simulator does not yet allow for trajectory simulation',
                           'boundary': boundary})
# -----------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES ------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_flexbound_tradeoff(np.ndarray[float, ndim = 1] v_h, 
                           np.ndarray[float, ndim = 1] v_l_1,
                           np.ndarray[float, ndim = 1] v_l_2,
                           np.ndarray[float, ndim = 1] a,
                           np.ndarray[float, ndim = 1] z_h,
                           np.ndarray[float, ndim = 1] z_l_1,
                           np.ndarray[float, ndim = 1] z_l_2,
                           np.ndarray[float, ndim = 1] d, # d for 'dampen' effect on drift parameter
                           np.ndarray[float, ndim = 1] t,
                           float s = 1,
                           float delta_t = 0.001,
                           float max_t = 20,
                           int n_samples = 20000,
                           int n_trials = 1,
                           print_info = True,
                           boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                           boundary_multiplicative = True,
                           boundary_params = {}
                           ):
    # Param views
    cdef float[:] v_h_view = v_h
    cdef float[:] v_l_1_view = v_l_1
    cdef float[:] v_l_2_view = v_l_2
    cdef float[:] a_view = a
    cdef float[:] z_h_view = z_h
    cdef float[:] z_l_1_view = z_l_1
    cdef float[:] z_l_2_view = z_l_2
    cdef float[:] d_view = d
    cdef float[:] t_view = t

    # TD: Add trajectory --> same issue as with par2 model above... might need to make a separate simulator for trajectories

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    # Y particle trace
    bias_trace = np.zeros(num_draws, dtype = DTYPE)
    cdef float[:] bias_trace_view = bias_trace

    cdef float y_h, y_l, v_l, t_h, t_l, tmp_pos_dep
    cdef Py_ssize_t n, ix, ix_tmp, k
    cdef Py_ssize_t m = 0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

        # Precompute boundary evaluations
        if boundary_multiplicative:
            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
    
        # Loop over samples
        for n in range(n_samples):
            choices_view[n, k, 0] = 0 # reset choice
            t_h = 0 # reset time high dimension
            t_l = 0 # reset time low dimension
            ix = 0 # reset boundary index

            # Initialize walkers
            y_h = (-1) * boundary_view[0] + (z_h_view[k] * 2 * (boundary_view[0])) 
            bias_trace_view[0] = ((y_h + boundary_view[0]) / (2 * boundary_view[0]))

            # Random walks until y_h hits bound
            while (y_h >= ((-1) * boundary_view[ix])) and ((y_h <= boundary_view[ix])) and (t_h <= max_t):
                y_h += (v_h_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
                bias_trace_view[ix] = ((y_h + boundary_view[ix]) / (2 * boundary_view[ix]))
                t_h += delta_t
                ix += 1
                m += 1
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> choices_view[n, k, 0] add 2 deterministically
            # y at lower bound --> choice_view[n, k, 0] stay the same deterministically
            if random_uniform() <= ((y_h + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 2
           
            if choices_view[n, k, 0] == 2:
                y_l = (- 1) * boundary_view[0] + (z_l_2_view[k] * 2 * (boundary_view[0])) 
                v_l = v_l_2_view[k]

                # Fill bias trace until max_rt reached
                ix_tmp = ix + 1
                while ix_tmp < num_draws:
                    bias_trace_view[ix_tmp] = 1.0
                    ix_tmp += 1

            else: # Store intermediate choice
                y_l = (- 1) * boundary_view[0] + (z_l_1_view[k] * 2 * (boundary_view[0])) 
                v_l = v_l_1_view[k]

                # Fill bias trace until max_rt reached
                ix_tmp = ix + 1
                while ix_tmp < num_draws:
                    bias_trace_view[ix_tmp] = 0.0
                    ix_tmp += 1

                #We need to reverse the bias_trace if we took the lower choice
                ix_tmp = 0 
                while ix_tmp < num_draws:
                    bias_trace_view[ix_tmp] = 1.0 - bias_trace_view[ix_tmp]
                    ix_tmp += 1

                #print('new bias_trace: ', bias_trace)
            
            # Random walks until the y_l corresponding to y_h hits bound
            ix = 0
            while (y_l >= ((-1) * boundary_view[ix])) and (y_l <= boundary_view[ix]) and (t_l <= max_t):
                # Compute local position dependence
                tmp_pos_dep = (1 + (d_view[k] * (bias_trace_view[ix] - 1.0))) / (2 - d_view[k])

                if (bias_trace_view[ix] < 1) and (bias_trace_view[ix] > 0):
                    # Before high-dim choice is taken
                    y_l += tmp_pos_dep * (v_l * delta_t) # Add drift
                    y_l += tmp_pos_dep * sqrt_st * gaussian_values[m] # Add noise
                else:
                    # After high-dim choice is taken
                    y_l += (v_l * delta_t) # Add drift
                    y_l += sqrt_st * gaussian_values[m] # Add noise
    
                t_l += delta_t # update time for low_dim choice
                ix += 1 # update time index
                m += 1 # update rv couter

                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            rts_view[n, k, 0] = fmax(t_h, t_l) + t_view[k]

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> choices_view[n, k, 0] add one deterministically
            # y at lower bound --> choice_view[n, k, 0] stays the same deterministically
            if random_uniform() <= ((y_l + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 1

    return {'rts': rts, 'choices': choices, 'metadata': {'vh': v_h,
                                                         'vl1': v_l_1,
                                                         'vl2': v_l_2,
                                                         'a': a,
                                                         'zh': z_h,
                                                         'zl1': z_l_1,
                                                         'zl2': z_l_2,
                                                         'd': d,
                                                         't': t,
                                                         's': s,
                                                         **boundary_params,
                                                         'delta_t': delta_t,
                                                         'max_t': max_t,
                                                         'n_samples': n_samples,
                                                         'simulator': 'ddm_flexbound_mic2_adj',
                                                         'boundary_fun_type': boundary_fun.__name__,
                                                         'possible_choices': [0, 1, 2, 3],
                                                         'trajectory': 'This simulator does not yet allow for trajectory simulation',
                                                         'boundary': boundary}}
# -----------------------------------------------------------------------------------------------
