import numpy as np
from numpy.random import rand

import time
import pymc as pm
import matplotlib.pyplot as plt
import pandas as pd

import hddm
import hddm.diag
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


def check_outlier_model(seed=None, p_outlier=0.05):
    """Estimate data which contains outliers"""

    if seed is not None:
        np.random.seed(seed)

    #generate params and data
    params_true = hddm.generate.gen_rand_params(include=())
    data, temp = hddm.generate.gen_rand_data(size=500, params=params_true)
    data = pd.DataFrame(data)

    #generating outliers
    n_outliers = int(len(data) * p_outlier)
    outliers = data[:n_outliers].copy()
    #fast outliers
    outliers.rt[:n_outliers//2] = np.random.rand(n_outliers//2) * (min(abs(data['rt'])) - 0.11)  + 0.11
    #slow outliers
    outliers.rt[n_outliers//2:] = np.random.rand(n_outliers - n_outliers//2) * 2 + max(abs(data['rt']))
    outliers.response = np.random.randint(0,2,n_outliers)
    print("generating %d outliers. %f of the dataset" % (n_outliers, float(n_outliers)/(n_outliers + len(data))))
    print("%d outliers are fast" % sum(outliers.rt < min(data.rt)))
    print("%d outliers are slow" % sum(outliers.rt > max(data.rt)))

    #Estimating the data without outliers. this is the best estimation we could get
    #from this data
    hm = hddm.HDDMTruncated(data)
    hm.map()
    index = ['true', 'estimated']
    best_estimate = hm.values
    df = pd.DataFrame([params_true, hm.values], index=index, dtype=np.float).dropna(1)
    print("benchmark: MAP of clean data. This is as good as we can get")
    print(df)

    #combine data with outliers
    data = pd.concat((data, outliers), ignore_index=True)

    #estimate the data with outlier, to confirm that it is worse
    hm = hddm.HDDMTruncated(data)
    hm.map()
    index = ['best_estimate', 'this_estimate']
    df = pd.DataFrame([best_estimate, hm.values], index=index, dtype=np.float).dropna(1)
    print("MAP with outliers: This is as bas as we can get")
    print(df)


    #MAP with p_outlier as random variable
    hm = hddm.HDDMTruncated(data,include='p_outlier')
    hm.map()
    df = pd.DataFrame([best_estimate, hm.values], index=index, dtype=np.float)
    df.loc['best_estimate', 'p_outlier'] = 0
    print("MAP with random p_outlier (Estimated from the data)")
    print(df.dropna(1))

    #MAP with fixed p_outlier
    fixed_p_outlier = 0.1
    hm = hddm.HDDMTruncated(data, p_outlier=fixed_p_outlier)
    hm.map()
    df = pd.DataFrame([best_estimate, hm.values], index=index, dtype=np.float)
    print("MAP with fixed p_outlier (%.3f) " % fixed_p_outlier)
    print(df.dropna(1))


    #Chi-square
    hm = hddm.HDDMTruncated(data)
    hm.optimize(method='chisquare')
    df = pd.DataFrame([best_estimate, hm.values], index=index, dtype=np.float).dropna(1)
    print("Chisquare method")
    print(df)

    return data

data = check_outlier_model(seed=1, p_outlier=0)