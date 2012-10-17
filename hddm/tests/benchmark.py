import numpy as np
from numpy.random import rand

import time
import pymc as pm
import matplotlib.pyplot as plt

import hddm
from hddm.likelihoods import *
from hddm.generate import *
from scipy.integrate import *
from scipy.stats import kstest
from scipy.optimize import fmin_powell

np.random.seed(3123)

def compare_cdf_from_pdf_to_cdf_from_fastdm(repeats=500, bins=25, include=('sv', 'st', 'sz', 'z')):
    """Comparing the numerical integration of wfpt PDF to fastdm CDF."""
    N = 500
    np.random.seed(10)
    diff = np.zeros(repeats)
    for i in range(repeats):
        params = hddm.generate.gen_rand_params(include=include)
        tic1 = time.time()
        x, cum_pdf = hddm.wfpt.gen_cdf_from_pdf(err=1e-4, N=N, **params)
        tic2 = time.time()
        x_cdf, cdf = hddm.wfpt.gen_cdf(N=N, **params)
        tic3 = time.time()
        
        pdf_time = tic2 - tic1
        fastdm_time = tic3 - tic2
        diff[i] = pdf_time - fastdm_time
        
    plt.figure()
    plt.hist(diff, bins)
    plt.title('compare gen_cdf_from_pdf to gen_cdf (using fastdm)')
    plt.xlabel('Difference in time. (positive values mean the fastdm is faster)')