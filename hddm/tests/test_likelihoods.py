import numpy as np
from numpy.random import rand

import unittest

import pymc as pm

import hddm
from hddm.likelihoods import *
from hddm.generate import *

class TestGPU(unittest.TestCase):
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
        
        self.params_single = {'x':1., 'a': 2., 'z': .5, 't':.3, 'v':.5}
        self.params_multi32 = {'x':x32, 'a': 2., 'z': .5, 't':.3, 'v':.5}
        self.params_multi = {'x':x, 'a': 2., 'z': .5, 't':.3, 'v':.5}
        self.params_multi_multi = self.create_params(x32)

    def create_params(self, x):
        params = {'x':x,
                  'a': np.ones_like(x)*2,
                  'z': np.ones_like(x)*1,
                  't':np.ones_like(x)*.3,
                  'v':np.ones_like(x)*.5}

        return params

    def test_GPU(self):
        logp = hddm.likelihoods.wiener_like_gpu(value=self.params_multi_multi['x'],
                               a=self.params_multi_multi['a'],
                               z=self.params_multi_multi['z'],
                               v=self.params_multi_multi['v'],
                               t=self.params_multi_multi['t'], debug=True)
        logp_single = hddm.likelihoods.wiener_like_gpu_single(value=self.params_multi32['x'],
                                       a=self.params_multi32['a'],
                                       z=self.params_multi32['z'],
                                       v=self.params_multi32['v'],
                                       t=self.params_multi32['t'], debug=True)

        np.testing.assert_array_almost_equal(logp, logp_single, 4)

    def test_GPU_direct(self):
        out = np.empty_like(self.params_multi_multi['x'])
        wfpt_gpu.pdf_func_complete(cuda.In(-(self.params_multi_multi['x']-self.params_multi_multi['t'])),
                                   cuda.In(self.params_multi_multi['a']),
                                   cuda.In(self.params_multi_multi['z']),
                                   cuda.In(self.params_multi_multi['v']),
                                   np.float32(0.0001), np.int16(1), cuda.Out(out),
                                   block = (self.params_multi_multi['x'].shape[0], 1, 1))

        probs = hddm.wfpt.pdf_array(-self.params_multi['x'],
                               self.params_multi['v'],
                               self.params_multi['a'],
                               self.params_multi['z'],
                               self.params_multi['t'],
                               0.0001, 1)


        np.testing.assert_array_almost_equal(out,probs,4)

    def test_simple(self):
        logp = hddm.likelihoods.wiener_like_simple(value=self.params_multi['x'],
                                  a=self.params_multi['a'],
                                  z=self.params_multi['z'],
                                  v=self.params_multi['v'],
                                  t=self.params_multi['t'])

        #t=timeit.Timer("""wiener_like_simple(value=-self.params_multi['x'], a=self.params_multi['a'], z=self.params_multi['z'], v=self.params_multi['v'], ter=self.params_multi['ter'])""", setup="from ddm_likelihood import *")
        #print t.timeit()

        logp_gpu = hddm.likelihoods.wiener_like_gpu(value=self.params_multi_multi['x'],
                               a=self.params_multi_multi['a'],
                               z=self.params_multi_multi['z'],
                               v=self.params_multi_multi['v'],
                               t=self.params_multi_multi['t'])

        self.assertAlmostEqual(np.float32(logp), logp_gpu, 4)

    def test_gpu_global(self):
        logp_gpu_global = hddm.likelihoods.wiener_like_gpu_global(value=self.params_multi_multi['x'],
                                                 a=self.params_multi_multi['a'],
                                                 z=self.params_multi_multi['z'],
                                                 v=self.params_multi_multi['v'],
                                                 t=self.params_multi_multi['t'], debug=True)

        logp_cpu = hddm.likelihoods.wiener_like_cpu(value=self.params_multi_multi['x'],
                                   a=self.params_multi_multi['a'],
                                   z=self.params_multi_multi['z'],
                                   v=self.params_multi_multi['v'],
                                   t=self.params_multi_multi['t'], debug=True)

        np.testing.assert_array_almost_equal(logp_cpu, logp_gpu_global, 4)

        free_gpu()
        
    def benchmark(self):
        logp_gpu = hddm.likelihoods.wiener_like_gpu(value=-self.params_multi_multi['x'],
                                   a=self.params_multi_multi['a'],
                                   z=self.params_multi_multi['z'],
                                   v=self.params_multi_multi['v'],
                                   t=self.params_multi_multi['t'], debug=True)

        logp_gpu_opt = hddm.likelihoods.wiener_like_gpu_opt(value=-self.params_multi_multi['x'],
                                   a=self.params_multi_multi['a'],
                                   z=self.params_multi_multi['z'],
                                   v=self.params_multi_multi['v'],
                                   t=self.params_multi_multi['t'], debug=True)

        logp_cpu = hddm.likelihoods.wiener_like_cpu(value=-self.params_multi_multi['x'],
                                   a=self.params_multi_multi['a'],
                                   z=self.params_multi_multi['z'],
                                   v=self.params_multi_multi['v'],
                                   t=self.params_multi_multi['t'], debug=True)

        #np.testing.assert_array_almost_equal(logp_cpu, logp_gpu, 4)

        #print logp_cpu, logp_gpu

    def benchmark_global(self):
        logp_gpu_global = hddm.likelihoods.wiener_like_gpu_global(value=-self.params_multi_multi['x'],
                                                 a=self.params_multi_multi['a'],
                                                 z=self.params_multi_multi['z'],
                                                 v=self.params_multi_multi['v'],
                                                 t=self.params_multi_multi['t'], debug=False)

        logp_cpu = hddm.likelihoods.wiener_like_cpu(value=-self.params_multi_multi['x'],
                                   a=self.params_multi_multi['a'],
                                   z=self.params_multi_multi['z'],
                                   v=self.params_multi_multi['v'],
                                   t=self.params_multi_multi['t'], debug=False)


    def benchmark_cpu(self):
        logp_cpu = hddm.likelihoods.wiener_like_cpu(value=-self.params_multi_multi['x'],
                                                    a=self.params_multi_multi['a'],
                                                    z=self.params_multi_multi['z'],
                                                    v=self.params_multi_multi['v'],
                                                    t=self.params_multi_multi['t'], debug=True)

def benchmark(size=100, reps=2000):
    import cProfile
    import pstats
#    cProfile.run('import hddm_test; bench = hddm_test.TestLikelihoodFuncs(); bench.setUp(size=%i); [bench.benchmark() for i in range(%i)]'%(size, reps), 'benchmark')
#    p = pstats.Stats('benchmark')
#    p.print_stats('wiener_like')

    cProfile.run('import test_likelihoods; bench = hddm_test.TestLikelihoodFuncs(); bench.setUp(size=%i); [bench.benchmark_global() for i in range(%i)]'%(size, reps), 'benchmark')
    p = pstats.Stats('benchmark')
    p.print_stats('wiener_like')
    free_gpu()

    return p

class TestWfpt(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestWfpt, self).__init__(*args, **kwargs)
        bins=100
        range_=(-4,4)
        self.x = np.linspace(range_[0], range_[1], bins)
        
        self.params_novar = {}
        self.params_novar['v'] = (rand()-.5)*1.5
        self.params_novar['t'] = rand()*.5
        self.params_novar['a'] = 1.5+rand()
        self.params_novar['z'] = rand()
        self.params_novar['V'] = 0
        self.params_novar['T'] = 0
        self.params_novar['Z'] = 0
        self.samples_novar = hddm.generate.gen_rts(self.params_novar, samples=30000)
        self.histo_novar = np.histogram(self.samples_novar, bins=bins, range=range_)[0]
        
        self.params = {}
        self.params['v'] = (rand()-.5)*1.5
        self.params['t'] = rand()*.5
        self.params['a'] = 1.5+rand()
        self.params['z'] = rand()
        self.params['V'] = rand()
        self.params['T'] = rand()*(self.params['t']/2.)
        self.params['Z'] = rand()*(self.params['z']/2.)
        self.samples = hddm.generate.gen_rts(self.params, samples=30000)
        self.histo = np.histogram(self.samples, bins=bins, range=range_)[0]
       
    def runTest(self):
        pass
    
    def test_simple(self):
        analytical_simple = hddm.wfpt.pdf_array(self.x,
                                                self.params_novar['v'],
                                                self.params_novar['a'],
                                                self.params_novar['z'],
                                                self.params_novar['t'],
                                                err=0.0001, logp=0)

        scaled_sim = hddm.utils.scale(self.histo_novar)
        scaled_analytic = hddm.utils.scale(analytical_simple)
        print np.mean(scaled_sim - scaled_analytic)
        np.testing.assert_array_almost_equal(scaled_sim, scaled_analytic, 1)
        
    def test_full_avg(self):
        analytical_full_avg = hddm.wfpt.wiener_like_full_mc(self.x,
                                                            self.params['v'],
                                                            self.params['V'],
                                                            self.params['z'],
                                                            self.params['Z'],
                                                            self.params['t'],
                                                            self.params['T'],
                                                            self.params['a'],
                                                            reps=1000,
                                                            err=0.0001, logp=0)

        scaled_sim = hddm.utils.scale(self.histo)
        scaled_analytic = hddm.utils.scale(analytical_full_avg)
        print np.mean(scaled_sim - scaled_analytic)
        np.testing.assert_array_almost_equal(scaled_sim, scaled_analytic, 1)


class TestLBA(unittest.TestCase):
    def runTest(self):
        pass
        #self.setUp()
    
    def setUp(self, size=200):
        self.x = np.random.rand(size)
        self.a = np.random.rand(1)+1
        self.z = np.random.rand(1)*self.a
        self.v = np.random.rand(2)+.5
        #self.v_multi = np.random.rand(5)
        self.V = np.random.rand(1)+.5

    def test_lba_single(self):
        import rpy2.robjects as robjects
        import rpy2.robjects.numpy2ri
        robjects.r.source('lba-math.r')

        like_cython = hddm.likelihoods.LBA_like(self.x, self.a, self.z, 0., self.V, self.v[0], self.v[1], logp=False)
        like_r = np.array(robjects.r.n1PDF(t=self.x, x0max=np.float(self.z), chi=np.float(self.a), drift=self.v, sdI=np.float(self.V)))
        np.testing.assert_array_almost_equal(like_cython, like_r,5)

    def call_cython(self):
        return hddm.lba.lba(self.x, self.z, self.a, self.V, self.v[0], self.v[1])

    def call_r(self):
        return np.array(robjects.r.n1PDF(t=self.x, x0max=np.float(self.z), chi=np.float(self.a), drift=self.v, sdI=np.float(self.V)))
        


if __name__=='__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestWfpt)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestLBA)
    unittest.TextTestRunner(verbosity=2).run(suite)
