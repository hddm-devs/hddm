from __future__ import division

import unittest

import numpy as np
import numpy.testing
import numpy.lib.recfunctions as rec
import matplotlib.pyplot as plt

import platform
from copy import copy
import subprocess

import pymc as pm

import hddm
from hddm.likelihoods import *
from hddm.generate import *

def scale(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

class TestLikelihoodFuncsGPU(unittest.TestCase):
    def runTest(self):
        pass
        #self.setUp()
    
    def setUp(self, size=20):
        import wfpt_gpu
        import pycuda.driver as cuda
        import pycuda

        pycuda.tools.mark_cuda_test(wfpt_gpu.pdf_func_complete)
        x = np.random.rand(size)+.5
        x32 = x.astype(np.float32)
        
        self.params_single = {'x':1., 'a': 2., 'z': 1., 'ter':.3, 'v':.5}
        self.params_multi32 = {'x':x32, 'a': 2., 'z': 1., 'ter':.3, 'v':.5}
        self.params_multi = {'x':x, 'a': 2., 'z': 1., 'ter':.3, 'v':.5}
        self.params_multi_multi = self.create_params(x32)

    def create_params(self, x):
        params = {'x':x,
                  'a': np.ones_like(x)*2,
                  'z': np.ones_like(x)*1,
                  'ter':np.ones_like(x)*.3,
                  'v':np.ones_like(x)*.5}

        return params

    def test_GPU(self):
        logp = hddm.likelihoods.wiener_like_gpu(value=self.params_multi_multi['x'],
                               a=self.params_multi_multi['a'],
                               z=self.params_multi_multi['z'],
                               v=self.params_multi_multi['v'],
                               ter=self.params_multi_multi['ter'], debug=True)
        logp_single = hddm.likelihoods.wiener_like_gpu_single(value=self.params_multi32['x'],
                                       a=self.params_multi32['a'],
                                       z=self.params_multi32['z'],
                                       v=self.params_multi32['v'],
                                       ter=self.params_multi32['ter'], debug=True)

        np.testing.assert_array_almost_equal(logp, logp_single, 4)

    def test_GPU_direct(self):
        out = np.empty_like(self.params_multi_multi['x'])
        wfpt_gpu.pdf_func_complete(cuda.In(-(self.params_multi_multi['x']-self.params_multi_multi['ter'])),
                                   cuda.In(self.params_multi_multi['a']),
                                   cuda.In(self.params_multi_multi['z']),
                                   cuda.In(self.params_multi_multi['v']),
                                   np.float32(0.0001), np.int16(1), cuda.Out(out),
                                   block = (self.params_multi_multi['x'].shape[0], 1, 1))

        probs = hddm.wfpt.pdf_array(-self.params_multi['x'],
                               self.params_multi['v'],
                               self.params_multi['a'],
                               self.params_multi['z'],
                               self.params_multi['ter'],
                               0.0001, 1)


        np.testing.assert_array_almost_equal(out,probs,4)

    def test_simple(self):
        logp = hddm.likelihoods.wiener_like_simple(value=self.params_multi['x'],
                                  a=self.params_multi['a'],
                                  z=self.params_multi['z'],
                                  v=self.params_multi['v'],
                                  ter=self.params_multi['ter'])

        #t=timeit.Timer("""wiener_like_simple(value=-self.params_multi['x'], a=self.params_multi['a'], z=self.params_multi['z'], v=self.params_multi['v'], ter=self.params_multi['ter'])""", setup="from ddm_likelihood import *")
        #print t.timeit()

        logp_gpu = hddm.likelihoods.wiener_like_gpu(value=self.params_multi_multi['x'],
                               a=self.params_multi_multi['a'],
                               z=self.params_multi_multi['z'],
                               v=self.params_multi_multi['v'],
                               ter=self.params_multi_multi['ter'])

        self.assertAlmostEqual(np.float32(logp), logp_gpu, 4)

    def test_gpu_global(self):
        logp_gpu_global = hddm.likelihoods.wiener_like_gpu_global(value=self.params_multi_multi['x'],
                                                 a=self.params_multi_multi['a'],
                                                 z=self.params_multi_multi['z'],
                                                 v=self.params_multi_multi['v'],
                                                 ter=self.params_multi_multi['ter'], debug=True)

        logp_cpu = hddm.likelihoods.wiener_like_cpu(value=self.params_multi_multi['x'],
                                   a=self.params_multi_multi['a'],
                                   z=self.params_multi_multi['z'],
                                   v=self.params_multi_multi['v'],
                                   ter=self.params_multi_multi['ter'], debug=True)

        np.testing.assert_array_almost_equal(logp_cpu, logp_gpu_global, 4)

        free_gpu()
        
    def benchmark(self):
        logp_gpu = hddm.likelihoods.wiener_like_gpu(value=-self.params_multi_multi['x'],
                                   a=self.params_multi_multi['a'],
                                   z=self.params_multi_multi['z'],
                                   v=self.params_multi_multi['v'],
                                   ter=self.params_multi_multi['ter'], debug=True)

        logp_gpu_opt = hddm.likelihoods.wiener_like_gpu_opt(value=-self.params_multi_multi['x'],
                                   a=self.params_multi_multi['a'],
                                   z=self.params_multi_multi['z'],
                                   v=self.params_multi_multi['v'],
                                   ter=self.params_multi_multi['ter'], debug=True)

        logp_cpu = hddm.likelihoods.wiener_like_cpu(value=-self.params_multi_multi['x'],
                                   a=self.params_multi_multi['a'],
                                   z=self.params_multi_multi['z'],
                                   v=self.params_multi_multi['v'],
                                   ter=self.params_multi_multi['ter'], debug=True)

        #np.testing.assert_array_almost_equal(logp_cpu, logp_gpu, 4)

        #print logp_cpu, logp_gpu

    def benchmark_global(self):
        logp_gpu_global = hddm.likelihoods.wiener_like_gpu_global(value=-self.params_multi_multi['x'],
                                                 a=self.params_multi_multi['a'],
                                                 z=self.params_multi_multi['z'],
                                                 v=self.params_multi_multi['v'],
                                                 ter=self.params_multi_multi['ter'], debug=False)

        logp_cpu = hddm.likelihoods.wiener_like_cpu(value=-self.params_multi_multi['x'],
                                   a=self.params_multi_multi['a'],
                                   z=self.params_multi_multi['z'],
                                   v=self.params_multi_multi['v'],
                                   ter=self.params_multi_multi['ter'], debug=False)


class TestLikelihoodFuncsLBA(unittest.TestCase):
    #def __init__(self):
    #    super(TestLikelihoodFuncsLBA, self).__init__()

    def runTest(self):
        pass
        #self.setUp()
    
    def setUp(self, size=20):
        import rpy2.robjects as robjects
        import rpy2.robjects.numpy2ri
        robjects.r.source('lba-math.r')

        self.x = np.random.rand(size)*3000
        self.out = np.empty_like(self.x)
        self.a = np.random.rand(1)*1000
        self.z = self.a / np.random.rand(1)
        self.v = np.random.rand(2)
        self.v_multi = np.random.rand(5)
        self.sv = np.random.rand(1)

    def test_lba_single(self):
        like_cython = hddm.lba.lba_like(self.x, self.z, self.a, self.sv, self.v[0], self.v[1])
        like_r = np.array(robjects.r.n1PDF(t=self.x, x0max=np.float(self.z), chi=np.float(self.a), drift=self.v, sdI=np.float(self.sv)))

        np.testing.assert_array_almost_equal(like_cython, like_r)

    def test_lba_single2(self):
        return
        like_cython = hddm.lba.lba_single(self.x, self.z, self.a, self.sv, self.v[0], self.v[1])
        like_r = np.array(robjects.r.n1PDF(t=self.x, x0max=np.float(self.z), chi=np.float(self.a), drift=self.v, sdI=np.float(self.sv)))

        np.testing.assert_array_almost_equal(like_cython, like_r)

    def call_cython(self):
        return hddm.lba.lba(self.x, self.z, self.a, self.sv, self.v[0], self.v[1])

    def call_r(self):
        return np.array(robjects.r.n1PDF(t=self.x, x0max=np.float(self.z), chi=np.float(self.a), drift=self.v, sdI=np.float(self.sv)))
        
    def test_lba_multi(self):
        return
        like_cython = hdmd.lba.lba(self.x, self.z, self.a, self.sv, selv.v_multi[0])
        like_r = np.array(robjects.r.n1PDF(t=self.x, x0max=np.float(self.z), chi=np.float(self.a), drift=self.v_multi, sdI=np.float(self.sv)))

        np.testing.assert_array_almost_equal(like_cython, like_r)        
        # logp_lba = LBA_like(self.params['x'],
        #                     self.params['z'],
        #                     self.params['a'],
        #                     self.params['ter'],
        #                     self.params['v'],
        #                     self.params['z'])
        


def benchmark(size=100, reps=2000):
    import cProfile
    import pstats
#    cProfile.run('import hddm_test; bench = hddm_test.TestLikelihoodFuncs(); bench.setUp(size=%i); [bench.benchmark() for i in range(%i)]'%(size, reps), 'benchmark')
#    p = pstats.Stats('benchmark')
#    p.print_stats('wiener_like')

    cProfile.run('import hddm_test; bench = hddm_test.TestLikelihoodFuncs(); bench.setUp(size=%i); [bench.benchmark_global() for i in range(%i)]'%(size, reps), 'benchmark')
    p = pstats.Stats('benchmark')
    p.print_stats('wiener_like')
    free_gpu()

    return p
    
def analyze_regress(posterior_trace, x_range=(-.5, .5), bins=100, plot=True):
    import scipy.interpolate
    
    x = np.linspace(x_range[0], x_range[1], bins)
    #prior_dist = pm.Normal('prior', mu=0, tau=100)
    prior_dist = pm.Uniform('prior', lower=-.5, upper=.5)

    prior_trace = np.array([prior_dist.rand() for i in range(10000)])

    sav_dick, prior, posterior, prior0, posterior0 = savage_dickey(prior_trace, posterior_trace, range=x_range, bins=bins)

    print "Savage Dickey: %f." % (sav_dick)

    prior_inter = scipy.interpolate.spline(xk=x, yk=prior, xnew=x)
    posterior_inter = scipy.interpolate.spline(xk=x, yk=posterior, xnew=x)

    plt.plot(x, prior_inter, label='prior')
    plt.plot(x, posterior_inter, label='posterior')

    return sav_dick

class TestHDDM(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestHDDM, self).__init__(*args, **kwargs)
        gen_data = False
        self.data, self.params_true = gen_rand_data(gen_data=gen_data, tag='global_test')
        self.data_subj, self.params_true_subj = gen_rand_subj_data(gen_data=gen_data, num_samples=300, add_noise=False, tag='subj_test')
        self.data_basic, self.params_true_basic = gen_rand_data(gen_data=gen_data, num_samples=2000, no_var=True, tag='global_basic_test')
        
        self.assert_ = True
        
        self.params_true_no_s = copy(self.params_true)
        del self.params_true_no_s['sv']
        del self.params_true_no_s['sz']
        del self.params_true_no_s['ster']

        self.params_true_subj_no_s = copy(self.params_true_subj)
        del self.params_true_subj_no_s['sv']
        del self.params_true_subj_no_s['sz']
        del self.params_true_subj_no_s['ster']

        self.params_true_lba = copy(self.params_true)
        del self.params_true_lba['sz']
        del self.params_true_lba['ster']

        self.params_true_basic_no_s = copy(self.params_true_basic)
        del self.params_true_basic_no_s['sv']
        del self.params_true_basic_no_s['sz']
        del self.params_true_basic_no_s['ster']

        self.samples = 10000
        self.burn = 5000

    def runTest(self):
        pass

    def test_basic(self):
        model = hddm.models.Multi(self.data_basic, no_bias=True)
        model.mcmc(samples=self.samples, burn=self.burn)
        self.check_model(model, self.params_true_basic_no_s)
        
    def test_full_avg(self):
        model = hddm.models.Multi(self.data, model_type='full_avg', no_bias=False)
        model.mcmc(samples=self.samples, burn=self.burn)
        self.check_model(model, self.params_true)

    def test_lba(self):
        model = hddm.models.Multi(self.data, is_subj_model=False, normalize_v=True, model_type='lba')
        model.mcmc(samples=self.samples, burn=self.burn)
        #self.check_model(model, self.params_true_lba)
        print model.params_est
        return model

    def test_lba_subj(self):
        model = hddm.models.Multi(self.data_subj, is_subj_model=True, normalize_v=True, model_type='lba')
        model.mcmc(samples=self.samples, burn=self.burn)
        #self.check_model(model, self.params_true_lba)
        print model.params_est
        return model

    def test_full(self):
        return
        model = hddm.models.Multi(self.data, model_type='full', no_bias=False)
        model.mcmc(samples=self.samples, burn=self.burn)
        self.check_model(model, self.params_true)

    def test_full_subj(self):
        return
        model = hddm.models.Multi(self.data_subj, model_type='full', is_subj_model=True, no_bias=False)
        model.mcmc(samples=self.samples, burn=self.burn)
        self.check_model(model, self.params_true_subj)

    def test_subjs_fixed_z(self):
        model = hddm.models.Multi(self.data_subj, model_type='simple', is_subj_model=True, no_bias=False)
        model.mcmc(samples=self.samples, burn=self.burn)
        self.check_model(model, self.params_true)

    def test_simple(self):
        model = hddm.models.Multi(self.data, model_type='simple', no_bias=True)
        model.mcmc(samples=self.samples, burn=self.burn)
        self.check_model(model, self.params_true_no_s)
        return model
    
    def test_simple_subjs(self):
        model = hddm.models.Multi(self.data_subj, model_type='simple', is_subj_model=True, no_bias=True)
        model.mcmc(samples=self.samples, burn=self.burn)
        self.check_model(model, self.params_true_subj_no_s)
        return model
    
    def test_simple_diff(self):
        return
        # Fit first model
        model = hddm.models.HDDM_simple(data, no_bias=True)
        model.mcmc()

        # Fit second model
        params = {'v': .1, 'sv': 0.3, 'z': 0.9, 'sz': 0.25, 'ter': .3, 'ster': 0.1, 'a': 1.8}
        data2, params_true2 = gen_rand_data(gen_data=True, params=params, num_samples=200)
        del params_true2['sv']
        del params_true2['sz']
        del params_true2['ster']
        model2 = hddm.models.HDDM_simple(data2, no_bias=True)
        model2.mcmc()

        self.check_model(model, params_true, assert_=True)
        self.check_model(model2, params_true2, assert_=True)
        model.plot_global(params_true=params_true)
        model2.plot_global(params_true=params_true2)

        return model, model2

    def test_chains(self):
        return
        models = run_parallel_chains(hddm.models.Multi, [data, data], ['test1', 'test2'])
        return models

    def check_model(self, model, params_true):
        """calculate the posterior estimate error if hidden parameters are known (e.g. when simulating data)."""
        err=[]

        # Check for proper chain convergence
        check_geweke(model, assert_=True)

        # Test for correct parameter estimation
        for param in params_true.iterkeys():
            # no-bias models do not contain z
            if model.no_bias and param == 'z':
                continue
            
            est = model.params_est[param]
            est_std = model.params_est_std[param]
            truth = params_true[param]
            err.append((est - truth)**2)
            if self.assert_:
                self.assertAlmostEqual(est, truth, 1)
            
            print "%s: Truth: %f, Estimated: %f, Error: %f, Std: %f" % (param, truth, est, err[-1], est_std)
                
            print "Summed error: %f" % np.sum(err)

        return np.sum(err)

def posterior_predictive_check(model, data):
    params = copy(model.params_est)
    if model.model_type.startswith('simple'):
        params['sv'] = 0.1
        params['sz'] = 0
        params['ster'] = 0
    if model.no_bias:
        params['z'] = params['a']/2.
        
    data_sampled = _gen_rts_params(params)

    # Check
    return pm.discrepancy(data_sampled, data, .5)
    
    
def check_geweke(model, assert_=True):
    # Test for convergence using geweke method
    for param in model.group_params.itervalues():
        geweke = np.array(pm.geweke(param))
        if assert_:
            assert (np.any(np.abs(geweke[:,1]) < 2)), 'Chain of %s not properly converged'%param
            return False
        else:
            if np.any(np.abs(geweke[:,1]) > 2):
                print "Chain of %s not properly converged" % param
                return False

    return True
    
if __name__=='__main__':
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestLikelihoodFuncsGPU)
    #unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestLikelihoodFuncsLBA)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestHDDM)
    unittest.TextTestRunner(verbosity=2).run(suite2)
