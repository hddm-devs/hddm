from __future__ import division
from copy import copy
import platform
import pymc as pm
import numpy as np
np.seterr(divide='ignore')

import hddm

def wiener_like_simple(value, v, z, ter, a):
    """Log-likelihood for the simple DDM"""
    if z is None:
        z = a/2.
    return np.sum(hddm.wfpt.pdf_array(value, v=v, a=a, z=z, ter=ter, err=.0001, logp=1))
    #return hddm.wfpt.wiener_like_simple(value, v=v, z=z, ter=ter, a=a, err=.0001)

@pm.randomwrap
def wiener_simple(v, z, ter, a, size=1):
    if z is None:
        z = a/2.
    return gen_ddm_rts(v=v, z=z, ter=ter, a=a, sz=0, sv=0, ster=0, size=size)

WienerSimple = pm.stochastic_from_dist(name="Wiener Simple Diffusion Process",
                                       logp=wiener_like_simple,
                                       random=wiener_simple,
                                       dtype=np.float,
                                       mv=True)

def pdf_array_multi_py(x, v, a, z, ter, multi=None, err=0.0001, logp=1):
    size = x.shape[0]
    y = np.empty(size, dtype=np.float)
    if multi is None:
        return hddm.wfpt.pdf_array(x, v=v, a=a, z=z, ter=ter, err=err, logp=logp)
    else:
        params = {'v':v, 'z':z, 'ter':ter, 'a':a}
        params_iter = copy(params)
        for i in range(size):
            for param in multi:
                params_iter[param] = params[param][i]
            y[i] = hddm.wfpt.pdf_sign(t=x[i], v=params_iter['v'], a=params_iter['a'], z=params_iter['z'], ter=params_iter['ter'], err=err, logp=logp)
        return y

def wiener_like_simple_multi(value, v, z, ter, a, multi=None):
    """Log-likelihood for the simple DDM"""
    if z is None:
        z = a/2.
    return np.sum(hddm.wfpt.pdf_array_multi(value, v=v, a=a, z=z, ter=ter, err=.001, logp=1, multi=multi))
    #return np.sum(pdf_array_multi_py(value, v=v, a=a, z=z, ter=ter, err=.001, logp=1, multi=multi))
            
WienerSimpleMulti = pm.stochastic_from_dist(name="Wiener Simple Diffusion Process",
                                            logp=wiener_like_simple_multi,
                                            dtype=np.float,
                                            mv=True)

@pm.randomwrap
def wiener_full(v, z, ter, a, sv, sz, ster, size=1):
    if z is None:
        z = a/2.
    return gen_ddm_rts(v=v, z=z, ter=ter, a=a, sz=sz, sv=sv, ster=ster, size=size)

def wiener_like_full_avg_multi(value, v, z, ter, a, sz, ster, sv, multi=None):
    """Log-likelihood for the simple DDM"""
    if z is None:
        z = a/2.
    # DEBUG: overwrite s parameters
    sz = 0.01
    sv = 0.01
    ster = 0.01
    return np.sum(hddm.wfpt.wiener_like_full_avg(value, v, sv, z, sz, ter, ster, a, err=.001, logp=1, reps=30, a_is_multi=True))
    #return np.sum(pdf_array_multi_py(value, v=v, a=a, z=z, ter=ter, err=.001, logp=1, multi=multi))
            
WienerFullAvgMulti = pm.stochastic_from_dist(name="Wiener Simple Diffusion Process",
                                             logp=wiener_like_full_avg_multi,
                                             dtype=np.float,
                                             mv=True)


def wiener_like_full_avg(value, v, sv, z, sz, ter, ster, a):
    """Log-likelihood for the full DDM using the sampling method"""
    return np.sum(hddm.wfpt.wiener_like_full_avg(t=value, v=v, sv=sv, z=z, sz=sz, ter=ter, ster=ster, a=a, err=.0001, reps=10))
 
WienerAvg = pm.stochastic_from_dist(name="Wiener Diffusion Process",
                                    logp=wiener_like_full_avg,
                                    random=wiener_full,
                                    dtype=np.float,
                                    mv=True)

# def wiener_like(value, v, a, z, ter):
#     """Log-likelihood of the DDM for one RT point."""
#     if a<z or z<=0:
#         return -np.Inf
#     prob = hddm.wfpt.pdf(t=value-ter, v=v, a=a, z=z, err=.001, logp=1)
#     if prob == -np.Inf:
#         print value, ter, v, a, z
#     return prob

# Wiener = pm.stochastic_from_dist(name="Wiener Diffusion Process",
#                                  logp=wiener_like,
#                                  dtype=np.float,
#                                  mv=False)

def wiener_like2(value, v, a, z, ter):
    """Log-likelihood of the DDM for one RT point."""
    prob = hddm.wfpt.pdf_sign(t=value, v=v, a=a, z=z, ter=ter, err=0.0001, logp=1)
    
    #if prob == -np.Inf:
    #    print value, ter, v, a, z
    return prob

Wiener2 = pm.stochastic_from_dist(name="Wiener Diffusion Process",
                                  logp=wiener_like2,
                                  random=wiener_simple,
                                  dtype=np.float,
                                  mv=False)

def centeruniform_like(value, center, width):
    R"""Likelihood of centered uniform"""
    return pm.uniform_like(value, lower=center-(width/2.), upper=center+(width/2.))

@pm.randomwrap
def centeruniform(center, width, size=1):
    R"""Sample from centered uniform"""
    return np.random.uniform(size=size, low=center-(width/2.), high=center+(width/2.))

CenterUniform = pm.stochastic_from_dist(name="Centered Uniform",
                                        logp=centeruniform_like,
                                        random=centeruniform,
                                        dtype=np.float,
                                        mv=False)


def get_avg_likelihood(x, params):
    pdf_upper = np.mean(hddm.wfpt.wiener_like_mult(t=x,
                                              v=-params['v'],
                                              sv=params['sv'],
                                              z=params['a']-params['z'],
                                              sz=params['sz'], ter=params['ter'],
                                              ster=params['ster'],
                                              a=params['a'],
                                              err=.0001, reps=100), axis=0)
    pdf_lower = np.mean(hddm.wfpt.wiener_like_mult(t=x,
                                              v=params['v'],
                                              sv=params['sv'],
                                              z=params['z'],
                                              sz=params['sz'],
                                              ter=params['ter'],
                                              ster=params['ster'],
                                              a=params['a'],
                                              err=.0001, reps=100), axis=0)
    
    pdf = np.concatenate((pdf_lower[::-1], pdf_upper))

    return pdf


################################
# Linear Ballistic Accumulator

def LBA_like(value, a, z, ter, sv, v0, v1=0., logp=True, normalize_v=False):
    """Linear Ballistic Accumulator PDF
    """
    if z is None:
        z = a/2.

    #print a, z, ter, v, sv
    prob = hddm.lba.lba_like(np.asarray(value, dtype=np.double), z, a, ter, sv, v0, v1, int(logp), int(normalize_v))
    return prob
    
def LBA_like_multi(value, a, z, ter, sv, v0, v1=0., multi=None, logp=True, normalize_v=False):
    """Linear Ballistic Accumulator PDF
    """
    size = value.shape[0]
    y = np.empty(size, dtype=np.float)
    if multi is None:
        return hddm.lba.lba_like(np.asarray(value, dtype=np.double),
                            z, a, ter, sv, v0, v1, int(logp), int(normalize_v))
    else:
        params = {'v0':v0, 'v1':v1, 'z':z, 'ter':ter, 'a':a, 'sv':sv}
        params_iter = copy(params) # Here we set the individual values

        for i in range(size):
            for param in multi:
                params_iter[param] = params[param][i]
            y[i] = hddm.lba.lba_like(np.asarray([value[i]], dtype=np.double),
                                np.double(params_iter['z']),
                                np.double(params_iter['a']),
                                np.double(params_iter['ter']),
                                np.double(params_iter['sv']),
                                np.double(params_iter['v0']),
                                np.double(params_iter['v1']),
                                logp=int(logp), normalize_v=int(normalize_v))
        prob = np.sum(y)
        return prob
    
LBA = pm.stochastic_from_dist(name='LBA likelihood',
                              logp=LBA_like,
                              dtype=np.float,
                              mv=True)


LBA_multi = pm.stochastic_from_dist(name='LBA likelihood',
                              logp=LBA_like_multi,
                              dtype=np.float,
                              mv=True)

################################################
# GPU Wiener likelihood functions EXPERIMENTAL #
################################################
try:
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    import hddm.wfpt_gpu
    from pycuda.tools import DeviceMemoryPool, PageLockedMemoryPool
    input_gpu = None
    output_gpu = None
    a_gpu = None
    z_gpu = None
    v_gpu = None
    ter_gpu = None
    out = None
    dev_pool_input = DeviceMemoryPool()
    dev_pool_output = DeviceMemoryPool()
    dev_pool_a = DeviceMemoryPool()
    dev_pool_z = DeviceMemoryPool()
    dev_pool_ter = DeviceMemoryPool()
    dev_pool_v = DeviceMemoryPool()

    gpu = True
except:
    gpu = False

def wiener_like_gpu_single(value, a, z, v, ter, debug=False):
    """Log-likelihood for the simple DDM"""
    if not gpu:
        raise NotImplementedError, "GPU likelihood function could not be loaded, is CUDA installed?"

    if z is None:
        z = np.float32(a/2.)
    if (ter<0) or np.any(np.abs(value)-ter < 0) or (a<z):
        if not debug:
            return -np.Inf

    input = gpuarray.to_gpu(value-np.float32(ter), allocator=dev_pool_input.allocate)
    out_gpu = gpuarray.empty_like(input)
    #ut_gpu = gpuarray.empty(input.shape, allocator=dev_pool_output.allocate)
    
    hddm.wfpt_gpu.pdf_func(input,
                      np.float32(a),
                      np.float32(z),
                      np.float32(v),
                      np.float32(.0001), # err
                      np.int16(1), # log
                      out_gpu,
                      block=(128, 1, 1),
                      grid=(int(np.ceil(value.shape[0]/128.)), 1))

    if debug:
        return out_gpu.get()
    else:
        return np.sum(out_gpu.get())



def wiener_like_cpu(value, a, z, v, ter, debug=False):
    """Test log likelihood for gpu (uses cpu).
    """
    if z[0] is None:
        z = a/2.

    out = np.empty_like(value)
    for i, val in enumerate(value):
        out[i] = hddm.wfpt.pdf_sign(val, v[i], a[i], z[i], ter[i], 0.001, 1)

    if not debug:
        return np.sum(out)
    else:
        return out
    
    
def wiener_like_gpu(value, a, z, v, ter, debug=False):
    """Log-likelihood for the simple DDM."""
    if not gpu:
        raise NotImplementedError, "GPU likelihood function could not be loaded, is CUDA installed?"

    a_array = np.asarray(a).astype(np.float32)
    if z[0] is None:
        z_array = a_array/2.
    else:
        z_array = np.asarray(z).astype(np.float32)
    v_array = np.asarray(v).astype(np.float32)
    ter_array = np.asarray(ter).astype(np.float32)

    tt = copy(value)
    tt[value>0] -= ter_array[value>0]
    tt[value<0] += ter_array[value<0]

    # Sanity checks
    if np.any(ter_array<0) or np.any(np.abs(value) < ter_array) or np.any(a_array<z_array):
        if not debug:
            return -np.Inf

    out = np.empty_like(tt)
    out_buf = cuda.Out(out)

    hddm.wfpt_gpu.pdf_func_complete(cuda.In(tt),
                               cuda.In(a_array),
                               cuda.In(z_array),                                   
                               cuda.In(v_array),
                               np.float32(.0001), np.int16(1),
                               out_buf,
                               block=(256, 1, 1),
                               grid=(int(np.ceil(tt.shape[0]/256.)), 1))

    if debug:
        out[ter_array<0] = -np.Inf
        out[np.abs(value) < ter_array] = -np.Inf
        out[a_array<z_array] = -np.Inf
        
    if not debug:
        return np.sum(out)
    else:
        return out


def free_gpu():
    global input_gpu, output_gpu, out, dev_pool, a_gpu, z_gpu, v_gpu, ter_gpu
    input_gpu.free()
    output_gpu.free()
    # a_gpu.free()
    # z_gpu.free()
    # v_gpu.free()
    # ter_gpu.free()
    input_gpu = None
    output_gpu = None
    
    #dev_pool.stop_holding()
    
    
def wiener_like_gpu_global(value, a, z, v, ter, debug=False):
    """Log-likelihood for the simple DDM."""
    if not gpu:
        raise NotImplementedError, "GPU likelihood function could not be loaded, is CUDA installed?"

    global input_gpu, output_gpu, out, dev_pool, a_gpu, z_gpu, v_gpu, ter_gpu
    
    a_array = np.asarray(a).astype(np.float32)
    if z[0] is None:
        z_array = a_array/2.
    else:
        z_array = np.asarray(z).astype(np.float32)
    v_array = np.asarray(v).astype(np.float32)
    ter_array = np.asarray(ter).astype(np.float32)

    # Sanity checks
    if np.any(ter_array<0) or np.any(np.abs(value) < ter_array) or np.any(a_array<z_array):
        if not debug:
            return -np.Inf

    if input_gpu is None:
        input_gpu = cuda.mem_alloc(value.nbytes)
        cuda.memcpy_htod(input_gpu, value)
    if output_gpu is None:
        output_gpu = cuda.mem_alloc(value.nbytes)
        out = np.empty_like(value)
        # a_gpu = cuda.mem_alloc(value.nbytes)
        # z_gpu = cuda.mem_alloc(value.nbytes)
        # v_gpu = cuda.mem_alloc(value.nbytes)
        # ter_gpu = cuda.mem_alloc(value.nbytes)

    #cuda.memcpy_htod(a_gpu, a_array)
    #cuda.memcpy_htod(z_gpu, z_array)
    #cuda.memcpy_htod(v_gpu, v_array)
    #cuda.memcpy_htod(ter_gpu, ter_array)
    a_gpu = gpuarray.to_gpu(a_array, allocator=dev_pool_a.allocate)
    z_gpu = gpuarray.to_gpu(z_array, allocator=dev_pool_z.allocate)
    v_gpu = gpuarray.to_gpu(v_array, allocator=dev_pool_v.allocate)
    ter_gpu = gpuarray.to_gpu(ter_array, allocator=dev_pool_ter.allocate)
    
    hddm.wfpt_gpu.pdf_func_ter(input_gpu,
                          a_gpu,
                          z_gpu, 
                          v_gpu,
                          ter_gpu,
                          np.float32(.0001), np.int16(1),
                          output_gpu,
                          block=(256, 1, 1),
                          grid=(int(np.ceil(value.shape[0]/256.)), 1))
        

    cuda.memcpy_dtoh(out, output_gpu)

    if debug:
        out[ter_array<0] = -np.Inf
        out[np.abs(value) < ter_array] = -np.Inf
        out[a_array<z_array] = -np.Inf
        
    if not debug:
        return np.sum(out)
    else:
        return out

def wiener_like_gpu_opt(value, a, z, v, ter, debug=False):
    """Log-likelihood for the simple DDM."""
    if not gpu:
        raise NotImplementedError, "GPU likelihood function could not be loaded, is CUDA installed?"
    a_array = np.asarray(a).astype(np.float32)
    if z[0] is None:
        no_bias = True
        z_array = a_array/2.
        mem = 512
    else:
        no_bias = False
        z_array = np.asarray(z).astype(np.float32)
        mem = 256

    v_array = np.asarray(v).astype(np.float32)
    ter_array = np.asarray(ter).astype(np.float32)

    tt = copy(value)
    tt[value>0] -= ter_array[value>0]
    tt[value<0] += ter_array[value<0]

    z_array[value>0] = a_array[value>0]-z_array[value>0]
    v_array[value>0] = -v_array[value>0]
    
    if np.any(ter_array<0) or np.any(ter_array[value[value>0] < ter_array[value>0]]) or np.any(tt[np.abs(value[value<0]) < ter_array[value<0]]) or np.any(a_array<z_array):
        if not debug:
            return -np.Inf

    out = np.empty_like(tt)
    out_buf = cuda.Out(out)

    if no_bias:
        hddm.wfpt_gpu.pdf_func_opt(cuda.In(np.abs(tt)),
                              cuda.In(a_array),
                              cuda.In(np.array([],dtype=np.float32)),
                              cuda.In(v_array),
                              np.float32(.0001), np.int16(1), np.int16(1),
                              out_buf,
                              block=(mem, 1, 1),
                              grid=(int(np.ceil(tt.shape[0]/mem)), 1))
    else:
        hddm.wfpt_gpu.pdf_func_opt(cuda.In(np.abs(tt)),
                              cuda.In(a_array),
                              cuda.In(z_array),
                              cuda.In(v_array),
                              np.float32(.0001), np.int16(1), np.int16(0),
                              out_buf,
                              block=(mem, 1, 1),
                              grid=(int(np.ceil(tt.shape[0]/mem)), 1))
        
    if np.any(out == -9999) or np.any(out == np.nan):
        return -np.Inf

    if not debug:
        return np.sum(out)
    else:
        return out
    
WienerGPUSingle = pm.stochastic_from_dist(name="Wiener Simple Diffusion Process",
                                    logp=wiener_like_gpu_single,
                                    dtype=np.float32,
                                    mv=True)

WienerGPU = pm.stochastic_from_dist(name="Wiener Simple Diffusion Process",
                                    logp=wiener_like_gpu,
                                    dtype=np.float32,
                                    mv=True)

WienerGPUGlobal = pm.stochastic_from_dist(name="Wiener Simple Diffusion Process",
                                          logp=wiener_like_gpu_global,
                                          dtype=np.float32,
                                          mv=True)

WienerGPUOpt = pm.stochastic_from_dist(name="Wiener Simple Diffusion Process",
                                    logp=wiener_like_gpu_opt,
                                    dtype=np.float32,
                                    mv=True)
