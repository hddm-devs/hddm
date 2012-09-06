.. index:: Tutorial
.. _chap_tutorial_config_subjects:


***********************************
Creating a hierarchical group model
***********************************

Up until now, we have been looking at data that was generated from the
same set of parameters. However, in most experiments, we test multiple
subjects and may only gather relatively few trials per
subject. Traditionally, we would either fit a separate model to each
individual subject or fit one model to all subjects. Neither of these
approaches are ideal as we will see below. We can expect that subjects
will be similar in many ways yet have non-negligable individual
differences. If we fit separate models we ignore their similarities
and need much more data per subject to make useful inference. If we
fit one model to all subjects we ignore their differences and may get
a model fit that does not well characterize any individual. It would
be best if the model fitting could determine to what extent all
subjects are similar to the others in some respects, and use this
information as a prior to draw inferences about the paramters of any
given subject. The hierarchical approach optimally allocates the
information from the group vs the individual depending on the
statistics of the data.

To illustrate this point, consider the following example: we tested 40
subjects on the above task with the easy and hard condition. For
practical reasons, however, we only collected 20 trials per
condition. As an example of what happens when trying to fit separate
models to each subject, we will run HDDM on the first subject. The
file simple_difficulty_subjs_single.csv only contains data from the
first subject. Lets run our model and see what happens:

::

    hddm_fit.py simple_difficulty.conf simple_subjs_difficulty_single.csv

    Creating model...
    Sampling: 100% [0000000000000000000000000000000000] Iterations: 10000
       name       mean   std    2.5q   25q    50q    75q    97.5  mc_err
    a         :  1.936  0.152  1.665  1.830  1.926  2.030  2.259  0.007
    t         :  0.314  0.044  0.214  0.288  0.318  0.347  0.386  0.002
    v('easy',):  0.468  0.248 -0.017  0.298  0.473  0.642  0.929  0.009
    v('hard',):  0.426  0.234 -0.028  0.268  0.428  0.571  0.869  0.008

    logp: -60.214194
    DIC: 113.099890

As you can see, the estimates are far worse (especially in the hard
condition) and the posterior distributions are much wider indicating a
lack of confidence in the estimates. Looking at the posterior
predictive and ill-shaped RT distribution makes obvious why fitting a
DDM to 10 trials is a fruitless attempt.

However, what about the data from the 39 other subjects? We certainly
wouldn't expect everyone to have the exact same parameters, but they
should be fairly similar. Couldn't we combine the data? This is where
the hierarchical approach becomes useful -- we can estimate individual
parameters, but at the same time have the parameters feed into a group
distribution so that the tiny bit we learn from each subject can be
pooled together and constrain the subject fits. Unfortunately, such a
model is a little bit more difficult to create in
general. Fortunately, HDDM does all of this automatically. Simply
running hddm_fit.py on a data file that has a column named 'subj_idx'
will make HDDM create a hierarchical model where each subject gets
assigned its own subject distribution which is assumed to be
distributed around a normal group distribution. Running this example
produces the following output:

::

    >>hddm_fit.py simple_difficulty.conf simple_subjs_difficulty.csv
    Creating model...
    Sampling: 100% [00000000000000000000000000000000] Iterations: 20000

         name        mean   std    2.5q   25q    50q    75q    97.5  mc_err
    a            :  2.000  0.026  1.950  1.985  1.996  2.018  2.059  0.002
    a0           :  2.002  0.053  1.889  1.977  1.996  2.032  2.120  0.003
    a1           :  1.987  0.051  1.870  1.966  1.990  2.011  2.084  0.002
    a10          :  2.007  0.049  1.918  1.981  1.997  2.032  2.123  0.003
    a11          :  2.009  0.050  1.916  1.981  1.998  2.033  2.122  0.003
    a12          :  2.001  0.052  1.898  1.977  1.994  2.028  2.111  0.002
    a13          :  2.000  0.052  1.890  1.977  1.994  2.027  2.112  0.002
    a14          :  1.995  0.051  1.885  1.973  1.993  2.020  2.104  0.002
    a15          :  1.978  0.058  1.829  1.956  1.986  2.009  2.082  0.003
    a16          :  2.000  0.050  1.900  1.976  1.995  2.023  2.121  0.002
    a17          :  2.011  0.053  1.915  1.980  1.999  2.040  2.142  0.003
    a18          :  2.013  0.057  1.906  1.983  2.000  2.045  2.149  0.003
    a19          :  2.009  0.054  1.910  1.981  1.999  2.038  2.127  0.003
    a2           :  2.003  0.051  1.900  1.980  1.996  2.029  2.119  0.003
    a20          :  2.041  0.069  1.951  1.991  2.022  2.076  2.213  0.005
    a21          :  1.987  0.054  1.856  1.966  1.990  2.015  2.094  0.003
    a22          :  2.003  0.053  1.894  1.976  1.995  2.030  2.112  0.002
    a23          :  2.002  0.054  1.886  1.978  1.995  2.029  2.125  0.002
    a24          :  1.999  0.049  1.903  1.976  1.994  2.026  2.107  0.002
    a25          :  2.012  0.055  1.914  1.983  2.000  2.041  2.139  0.003
    a26          :  2.006  0.052  1.906  1.981  1.997  2.033  2.122  0.002
    a27          :  1.988  0.053  1.865  1.968  1.991  2.012  2.087  0.003
    a28          :  2.015  0.055  1.923  1.983  2.001  2.045  2.147  0.003
    a29          :  1.982  0.054  1.853  1.961  1.988  2.010  2.082  0.003
    a3           :  1.981  0.056  1.842  1.960  1.988  2.008  2.081  0.003
    a30          :  2.013  0.056  1.910  1.983  1.999  2.042  2.147  0.003
    a31          :  1.999  0.048  1.901  1.976  1.993  2.024  2.101  0.002
    a32          :  1.999  0.050  1.899  1.977  1.994  2.024  2.111  0.002
    a33          :  2.017  0.053  1.929  1.985  2.004  2.044  2.153  0.003
    a34          :  2.001  0.052  1.898  1.978  1.994  2.023  2.120  0.002
    a35          :  2.002  0.052  1.895  1.978  1.995  2.027  2.117  0.002
    a36          :  1.983  0.052  1.849  1.964  1.988  2.010  2.077  0.003
    a37          :  2.002  0.054  1.890  1.977  1.994  2.029  2.127  0.003
    a38          :  1.984  0.055  1.845  1.965  1.991  2.011  2.083  0.003
    a39          :  1.975  0.058  1.826  1.954  1.985  2.007  2.071  0.003
    a4           :  1.998  0.052  1.886  1.975  1.994  2.025  2.105  0.002
    a5           :  2.024  0.056  1.930  1.987  2.008  2.053  2.164  0.003
    a6           :  2.003  0.051  1.900  1.980  1.996  2.029  2.120  0.002
    a7           :  2.004  0.051  1.901  1.978  1.996  2.030  2.119  0.002
    a8           :  1.989  0.051  1.879  1.967  1.990  2.015  2.089  0.002
    a9           :  1.984  0.055  1.847  1.964  1.990  2.014  2.083  0.003
    avar         :  0.042  0.030  0.005  0.016  0.038  0.065  0.103  0.003
    t            :  0.305  0.006  0.293  0.302  0.305  0.309  0.316  0.000
    t0           :  0.308  0.012  0.286  0.301  0.307  0.314  0.337  0.001
    t1           :  0.304  0.012  0.279  0.298  0.305  0.311  0.328  0.000
    t10          :  0.308  0.012  0.284  0.301  0.307  0.314  0.334  0.001
    t11          :  0.308  0.013  0.284  0.301  0.307  0.314  0.341  0.001
    t12          :  0.306  0.012  0.281  0.300  0.306  0.312  0.331  0.000
    t13          :  0.309  0.013  0.286  0.301  0.307  0.314  0.340  0.001
    t14          :  0.304  0.011  0.277  0.299  0.305  0.311  0.327  0.000
    t15          :  0.305  0.011  0.279  0.299  0.305  0.311  0.326  0.001
    t16          :  0.305  0.012  0.281  0.299  0.305  0.311  0.328  0.000
    t17          :  0.307  0.012  0.284  0.300  0.306  0.312  0.333  0.001
    t18          :  0.306  0.011  0.283  0.300  0.306  0.312  0.331  0.001
    t19          :  0.307  0.012  0.285  0.301  0.307  0.313  0.335  0.001
    t2           :  0.306  0.012  0.281  0.299  0.306  0.312  0.329  0.001
    t20          :  0.309  0.014  0.287  0.301  0.308  0.315  0.343  0.001
    t21          :  0.303  0.011  0.275  0.298  0.304  0.310  0.323  0.001
    t22          :  0.306  0.011  0.282  0.300  0.306  0.312  0.331  0.001
    t23          :  0.307  0.012  0.284  0.301  0.306  0.313  0.335  0.001
    t24          :  0.303  0.012  0.275  0.298  0.304  0.310  0.325  0.001
    t25          :  0.305  0.012  0.281  0.299  0.306  0.312  0.330  0.001
    t26          :  0.308  0.012  0.286  0.301  0.307  0.313  0.338  0.001
    t27          :  0.302  0.012  0.273  0.296  0.303  0.310  0.323  0.001
    t28          :  0.307  0.011  0.285  0.300  0.306  0.313  0.332  0.001
    t29          :  0.300  0.011  0.272  0.295  0.302  0.308  0.319  0.001
    t3           :  0.303  0.011  0.278  0.298  0.304  0.310  0.323  0.001
    t30          :  0.308  0.012  0.286  0.301  0.307  0.314  0.338  0.001
    t31          :  0.304  0.011  0.280  0.298  0.305  0.311  0.325  0.001
    t32          :  0.305  0.011  0.278  0.299  0.305  0.311  0.328  0.001
    t33          :  0.304  0.011  0.278  0.298  0.304  0.310  0.325  0.001
    t34          :  0.306  0.012  0.281  0.299  0.306  0.312  0.330  0.000
    t35          :  0.304  0.012  0.275  0.298  0.304  0.311  0.327  0.001
    t36          :  0.299  0.011  0.270  0.293  0.301  0.307  0.316  0.001
    t37          :  0.307  0.012  0.283  0.300  0.307  0.313  0.335  0.001
    t38          :  0.302  0.011  0.276  0.296  0.303  0.309  0.321  0.001
    t39          :  0.306  0.012  0.282  0.299  0.305  0.312  0.332  0.001
    t4           :  0.306  0.011  0.283  0.300  0.305  0.312  0.329  0.000
    t5           :  0.305  0.011  0.280  0.299  0.305  0.311  0.327  0.001
    t6           :  0.306  0.012  0.285  0.300  0.306  0.312  0.334  0.001
    t7           :  0.308  0.012  0.285  0.301  0.307  0.313  0.337  0.001
    t8           :  0.305  0.012  0.281  0.299  0.306  0.312  0.329  0.001
    t9           :  0.299  0.012  0.269  0.294  0.301  0.307  0.318  0.001
    tvar         :  0.009  0.006  0.002  0.004  0.008  0.013  0.025  0.001
    v('easy',)   :  0.966  0.050  0.863  0.934  0.969  0.998  1.057  0.004
    v('easy',)0  :  0.970  0.092  0.786  0.920  0.971  1.017  1.190  0.004
    v('easy',)1  :  0.983  0.091  0.810  0.929  0.983  1.031  1.180  0.005
    v('easy',)10 :  0.964  0.093  0.762  0.915  0.970  1.013  1.159  0.004
    v('easy',)11 :  0.989  0.093  0.806  0.936  0.988  1.038  1.195  0.005
    v('easy',)12 :  0.962  0.085  0.786  0.914  0.967  1.012  1.125  0.004
    v('easy',)13 :  0.964  0.090  0.764  0.915  0.971  1.017  1.153  0.005
    v('easy',)14 :  0.962  0.089  0.770  0.918  0.967  1.010  1.136  0.004
    v('easy',)15 :  0.963  0.090  0.774  0.916  0.967  1.013  1.151  0.004
    v('easy',)16 :  0.949  0.091  0.739  0.902  0.960  1.005  1.119  0.004
    v('easy',)17 :  0.969  0.089  0.784  0.919  0.968  1.016  1.160  0.005
    v('easy',)18 :  0.978  0.089  0.806  0.926  0.979  1.026  1.176  0.004
    v('easy',)19 :  0.933  0.101  0.687  0.887  0.950  0.997  1.098  0.006
    v('easy',)2  :  0.989  0.093  0.821  0.933  0.983  1.035  1.204  0.005
    v('easy',)20 :  0.981  0.094  0.798  0.927  0.978  1.024  1.197  0.005
    v('easy',)21 :  0.995  0.096  0.816  0.937  0.990  1.042  1.213  0.005
    v('easy',)22 :  0.936  0.097  0.695  0.888  0.952  1.001  1.095  0.005
    v('easy',)23 :  0.958  0.088  0.755  0.912  0.968  1.010  1.129  0.005
    v('easy',)24 :  0.971  0.085  0.788  0.925  0.974  1.019  1.147  0.004
    v('easy',)25 :  0.961  0.089  0.773  0.909  0.968  1.011  1.134  0.004
    v('easy',)26 :  0.977  0.087  0.794  0.930  0.977  1.025  1.168  0.004
    v('easy',)27 :  0.995  0.096  0.830  0.939  0.987  1.037  1.234  0.005
    v('easy',)28 :  0.957  0.090  0.756  0.909  0.966  1.009  1.133  0.004
    v('easy',)29 :  0.963  0.093  0.765  0.917  0.970  1.014  1.135  0.005
    v('easy',)3  :  0.970  0.093  0.784  0.919  0.972  1.018  1.163  0.004
    v('easy',)30 :  0.950  0.093  0.748  0.896  0.961  1.007  1.117  0.005
    v('easy',)31 :  0.959  0.087  0.780  0.910  0.965  1.011  1.134  0.004
    v('easy',)32 :  0.958  0.087  0.770  0.907  0.965  1.011  1.128  0.004
    v('easy',)33 :  0.983  0.093  0.816  0.930  0.982  1.029  1.188  0.005
    v('easy',)34 :  0.969  0.088  0.785  0.921  0.973  1.016  1.144  0.004
    v('easy',)35 :  0.965  0.086  0.789  0.913  0.971  1.015  1.142  0.004
    v('easy',)36 :  0.949  0.090  0.751  0.901  0.961  1.001  1.103  0.005
    v('easy',)37 :  0.908  0.111  0.639  0.856  0.933  0.982  1.067  0.007
    v('easy',)38 :  0.938  0.091  0.727  0.894  0.954  0.997  1.089  0.005
    v('easy',)39 :  0.969  0.091  0.786  0.922  0.975  1.016  1.150  0.005
    v('easy',)4  :  0.983  0.090  0.807  0.934  0.979  1.029  1.180  0.004
    v('easy',)5  :  0.965  0.087  0.788  0.915  0.969  1.013  1.152  0.004
    v('easy',)6  :  0.961  0.087  0.782  0.909  0.966  1.013  1.138  0.004
    v('easy',)7  :  0.973  0.088  0.794  0.924  0.975  1.018  1.173  0.004
    v('easy',)8  :  0.989  0.095  0.817  0.935  0.982  1.028  1.230  0.005
    v('easy',)9  :  0.969  0.089  0.780  0.918  0.972  1.021  1.148  0.004
    v('hard',)   :  0.401  0.045  0.317  0.370  0.400  0.430  0.500  0.003
    v('hard',)0  :  0.488  0.138  0.278  0.387  0.464  0.565  0.803  0.007
    v('hard',)1  :  0.424  0.119  0.203  0.354  0.411  0.489  0.695  0.004
    v('hard',)10 :  0.333  0.115  0.075  0.273  0.344  0.406  0.551  0.005
    v('hard',)11 :  0.393  0.111  0.181  0.322  0.387  0.461  0.626  0.004
    v('hard',)12 :  0.400  0.109  0.183  0.336  0.393  0.466  0.632  0.003
    v('hard',)13 :  0.444  0.120  0.232  0.365  0.430  0.506  0.727  0.005
    v('hard',)14 :  0.385  0.109  0.162  0.321  0.380  0.447  0.618  0.003
    v('hard',)15 :  0.387  0.116  0.157  0.320  0.383  0.454  0.629  0.004
    v('hard',)16 :  0.379  0.111  0.150  0.313  0.375  0.448  0.614  0.004
    v('hard',)17 :  0.435  0.119  0.238  0.350  0.421  0.502  0.717  0.005
    v('hard',)18 :  0.374  0.109  0.140  0.313  0.373  0.438  0.597  0.003
    v('hard',)19 :  0.423  0.112  0.212  0.350  0.417  0.486  0.654  0.005
    v('hard',)2  :  0.396  0.117  0.148  0.327  0.391  0.463  0.653  0.004
    v('hard',)20 :  0.422  0.109  0.230  0.351  0.412  0.485  0.675  0.004
    v('hard',)21 :  0.437  0.120  0.229  0.357  0.423  0.510  0.705  0.005
    v('hard',)22 :  0.414  0.116  0.196  0.342  0.406  0.479  0.671  0.004
    v('hard',)23 :  0.375  0.110  0.140  0.312  0.376  0.438  0.590  0.004
    v('hard',)24 :  0.303  0.130 -0.010  0.240  0.323  0.388  0.522  0.006
    v('hard',)25 :  0.465  0.129  0.248  0.377  0.447  0.546  0.767  0.007
    v('hard',)26 :  0.398  0.116  0.167  0.330  0.395  0.460  0.638  0.004
    v('hard',)27 :  0.437  0.117  0.223  0.360  0.423  0.501  0.697  0.005
    v('hard',)28 :  0.347  0.117  0.081  0.284  0.357  0.420  0.557  0.004
    v('hard',)29 :  0.405  0.116  0.202  0.332  0.396  0.471  0.653  0.004
    v('hard',)3  :  0.420  0.117  0.193  0.344  0.412  0.489  0.682  0.004
    v('hard',)30 :  0.357  0.107  0.111  0.296  0.359  0.421  0.564  0.004
    v('hard',)31 :  0.395  0.108  0.182  0.328  0.392  0.454  0.626  0.004
    v('hard',)32 :  0.338  0.115  0.079  0.274  0.347  0.410  0.561  0.004
    v('hard',)33 :  0.389  0.112  0.159  0.324  0.389  0.456  0.629  0.004
    v('hard',)34 :  0.450  0.124  0.254  0.366  0.437  0.519  0.727  0.005
    v('hard',)35 :  0.394  0.112  0.172  0.327  0.387  0.461  0.630  0.004
    v('hard',)36 :  0.326  0.120  0.045  0.266  0.342  0.397  0.537  0.004
    v('hard',)37 :  0.393  0.111  0.164  0.327  0.388  0.459  0.636  0.004
    v('hard',)38 :  0.433  0.118  0.223  0.356  0.418  0.497  0.704  0.005
    v('hard',)39 :  0.424  0.119  0.202  0.349  0.418  0.490  0.680  0.005
    v('hard',)4  :  0.430  0.126  0.201  0.355  0.420  0.499  0.718  0.005
    v('hard',)5  :  0.380  0.110  0.144  0.319  0.377  0.446  0.603  0.004
    v('hard',)6  :  0.357  0.115  0.114  0.295  0.362  0.423  0.586  0.004
    v('hard',)7  :  0.449  0.124  0.238  0.366  0.434  0.527  0.720  0.006
    v('hard',)8  :  0.425  0.122  0.214  0.349  0.411  0.494  0.706  0.005
    v('hard',)9  :  0.476  0.133  0.264  0.383  0.456  0.553  0.796  0.007
    vvar('easy',):  0.111  0.053  0.030  0.068  0.107  0.145  0.221  0.004
    vvar('hard',):  0.070  0.047  0.017  0.033  0.058  0.097  0.185  0.004

    logp: -1161.693344
    DIC: 2882.445038
    Plotting t
    Plotting a
    Plotting v('hard',)
    Plotting v('easy',)
    Plotting posterior predictive...


The first you can see when examining the recovered parameter values is
that the mean of the group distributions (i.e. a, t, v('hard',) and
v('easy',)) is that they match very well the parameters we used to
generate the data from. So by pooling the very little data we had on
each subject we can make useful inference about the group parameters.

The second thing you can see is that individual subject parameters
(ending with the index of the subject) are very close to the group
mean (this is also indicated by the fact that the var posteriors, the
variance of the group distribution -- representing the spread of the
individual subject parameters). This property is called
*shrinkage*. Intuitively, if we can not make meaningful inference
about individual subjects we will assume that they are distributed as
the rest of the group.
