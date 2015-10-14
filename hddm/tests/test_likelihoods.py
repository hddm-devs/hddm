import numpy as np
from numpy.random import rand
import scipy as sp

import unittest

import pymc as pm

import hddm
from hddm.likelihoods import *
from hddm.generate import *
from scipy.integrate import *
from scipy.stats import kstest
from scipy.optimize import fmin_powell
from nose import SkipTest

np.random.seed(3123)

def gen_like_from_matlab(samples=20, seed=3123):
    # Generate samples from the reference implementation by Navarro & Fuss 2009
    # This was used to generate matlab_values.py for comparison with our implementation.
    import mlabwrap

    np.random.seed(seed)

    vals = []

    for i in range(samples):
        v = (rand()-.5)*1.5
        t = rand()*.5
        a = 1.5+rand()
        z = .5*rand()
        z_nonorm = a*z
        rt = rand()*4 + t
        err = 10**(round(rand()*-10))
        # Test if equal up to the 9th decimal.
        matlab_wfpt = mlabwrap.mlab.wfpt(rt-t, v, a, z_nonorm, err)[0][0]

        vals.append((v, t, a, z, z_nonorm, rt, err, matlab_wfpt))

    return vals

class TestWfpt(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestWfpt, self).__init__(*args, **kwargs)
        self.samples=5000
        self.range_ = (-10,10)

    def test_pdf_no_matlab(self):
        # Test if our wfpt pdf implementation yields the same results as the reference implementation by Navarro & Fuss 2009
        from .matlab_values import vals

        for v, t, a, z, z_nonorm, rt, err, matlab_wfpt in vals:
            python_wfpt = hddm.wfpt.full_pdf(-rt, v, 0, a, z, 0, t, 0, err, 0)
            print(v,t,a,z,z_nonorm,rt,err, matlab_wfpt, python_wfpt)
            np.testing.assert_array_almost_equal(matlab_wfpt, python_wfpt, 9)

    def test_pdf(self):
        # Test if our wfpt pdf implementation yields the same results as the reference implementation by Navarro & Fuss 2009
        try:
            import mlabwrap
        except ImportError:
            raise SkipTest("Could not import mlabwrap, not performing pdf comparison test.")

        for i in range(500):
            v = (rand()-.5)*1.5 # TODO!
            t = rand()*.5
            a = 1.5+rand()
            z = .5*rand()
            z_nonorm = a*z
            rt = rand()*4 + t
            err = 1e-4
            # Test if equal up to the 9th decimal.
            matlab_wfpt = mlabwrap.mlab.wfpt(rt-t, v, a, z_nonorm, err)[0][0]
            python_wfpt = hddm.wfpt.pdf_array(np.asarray([-rt]), v, 0, a, z, 0, t, 0, err, 0)[0]
            print(v,t,a,z,z_nonorm,rt,err, matlab_wfpt, python_wfpt)
            np.testing.assert_array_almost_equal(matlab_wfpt, python_wfpt, 9)

    def test_summed_logp(self):
        np.random.seed(123)
        params = hddm.generate.gen_rand_params(include=('sv','sz','st'))

        # Generate random valid RTs
        rts = params['t'] + params['st'] + rand(50)*2
        p = hddm.wfpt.pdf_array(rts, params['v'], params['sv'],
                                params['a'], params['z'], params['sz'], params['t'],
                                params['st'], 1e-4, logp=True)

        summed_logp = np.sum(p)

        summed_logp_like = hddm.wfpt.wiener_like(rts, params['v'],
                                                 params['sv'], params['a'], params['z'], params['sz'],
                                                 params['t'], params['st'], 1e-4)

        np.testing.assert_almost_equal(summed_logp, summed_logp_like, 2)

        self.assertTrue(-np.Inf == hddm.wfpt.wiener_like(np.array([1.,2.,3.,0.]), 1, 0, 2, .5, 0, 0, 0, 1e-4)), "wiener_like_simple should have returned -np.Inf"

    def test_pdf_sv(self, samples=50):
        """Test if our wfpt pdf_sv implementation produces the same value as numerical integration over v"""
        func = lambda v_i,value,err,v,sv,z,a: hddm.wfpt.full_pdf(value, v_i, 0, a, z, 0, 0, 0, err) * norm.pdf(v_i,v,sv)

        for i in range(50):
            sv = rand()*0.4+0.1
            v = (rand()-.5)*4
            t = rand()*.5
            a = 1.5+rand()
            z = .5*rand()
            rt = rand()*4 + t
            err = 10**(-3- np.ceil(rand()*12))
            # Test if equal up to the 9th decimal.
            sp_result = quad(func, -np.inf, np.inf, args=(rt,err,v,sv,z,a), epsrel=1e-10, epsabs=1e-10)[0]
            hddm_result = hddm.wfpt.full_pdf(rt, v, sv, a, z, 0, 0, 0, err)
            np.testing.assert_array_almost_equal(hddm_result, sp_result)

class TestWfptFull(unittest.TestCase):
    def test_adaptive(self):
        for i in range(20):
            sv = rand()*0.4+0.1
            v = (rand()-.5)*4
            st = rand()*0.3
            t = rand()*.5+(st/2)
            a = 1.5+rand()
            rt = (rand()*4 + t) * np.sign(rand())
            err = 10**-9
            sz = rand()*0.3
            z = .5*rand()+sz/2
            logp = 0#np.floor(rand()*2)
            n_sz = 60
            n_st = 60

            my_res = hddm.wfpt.full_pdf(rt,v=v,sv=0,a=a,z=z,sz=0,t=t, st=st,err=err, n_st=5, n_sz=5, use_adaptive=1)
            res = hddm.wfpt.full_pdf(rt,v=v,sv=0,a=a,z=z,sz=0,t=t, st=st,err=err, n_st=n_st, n_sz=n_sz, use_adaptive=0)

            print("(%d) rt %f, v: %f, sv: %f, z: %f, sz: %f, t: %f, st: %f a: %f" % (i,rt,v,sv,z,sz,t,st,a))
            print(my_res)
            print(res)
            if np.isinf(my_res):
                my_res = 100
            if np.isinf(res):
                res = 100
            self.assertTrue(not np.isnan(my_res)), "Found NaN in the results"
            self.assertTrue(not np.isnan(res)), "Found NaN in the simulated results"
            np.testing.assert_array_almost_equal(my_res, res, 3)


    def test_pdf_integrate_to_one(self):
        np.random.seed(123)
        for tests in range(2):
            sv = rand()*0.4+0.1
            v = (rand()-.5)*4
            st = rand()*0.3
            t = rand()*.5+(st/2)
            a = 1.5+rand()
            err = 10**-8
            sz = rand()*0.3
            z = .5*rand()+sz/2
            func = lambda x: np.exp(hddm.wfpt.wiener_like(np.array([x]), v, sv, a, z, sz, t, st, err))
            integ, error = sp.integrate.quad(func, a=-5, b=5)

            np.testing.assert_almost_equal(integ, 1, 2)

    def test_wiener_like_full_single(self):
        for i in range(20):
            sv = rand()*0.4+0.1
            v = (rand()-.5)*4
            st = rand()*0.3
            t = rand()*.5+(st/2)
            a = 1.5+rand()
            rt = (rand()*4 + t) * np.sign(rand())
            err = 10**-8
            sz = rand()*0.3
            z = .5*rand()+sz/2
            n_sz = 60
            n_st = 60

            my_res = np.zeros(8)
            res = np.zeros(8)
            y_z = np.zeros(n_sz+1);
            y_t = np.zeros(n_st+1)

            for vvv in range(2):
                #test pdf
                my_res[0+vvv*4] = hddm.wfpt.full_pdf(rt,v=v,sv=sv*vvv,a=a,z=z,sz=0,t=t, st=0,err=err, n_st=n_st, n_sz=n_sz)
                res[0+vvv*4]    = hddm.wfpt.full_pdf(rt,v=v,sv=sv*vvv,a=a,z=z,sz=0,t=t, st=0,err=err, n_st=0, n_sz=0)

                #test pdf + sz
                my_res[1+vvv*4] = hddm.wfpt.full_pdf(rt,v=v,sv=sv*vvv,a=a,z=z,sz=sz,t=t, st=0,err=err, n_st=n_st, n_sz=n_sz)
                hsz = sz/n_sz
                for j in range(n_sz+1):
                    z_tag = z-sz/2. + hsz*j
                    y_z[j] = hddm.wfpt.full_pdf(rt,v=v,sv=sv*vvv,a=a,z=z_tag,sz=0,t=t, st=0,err=err, n_st=0, n_sz=0)/sz
                    res[1+vvv*4] = simps(y_z, x=None, dx=hsz)

                #test pdf + st
                my_res[2+vvv*4] = hddm.wfpt.full_pdf(rt,v=v,sv=sv*vvv,a=a,z=z,sz=0,t=t, st=st,err=err, n_st=n_st, n_sz=n_sz)
                hst = st/n_st
                for j in range(n_st+1):
                    t_tag = t-st/2. + hst*j
                    y_t[j] = hddm.wfpt.full_pdf(rt,v=v,sv=sv*vvv,a=a,z=z,sz=0,t=t_tag, st=0,err=err, n_st=0, n_sz=0)/st
                    res[2+vvv*4] = simps(y_t, x=None, dx=hst)

                #test pdf + sz + st
                my_res[3+vvv*4] = hddm.wfpt.full_pdf(rt,v=v,sv=sv*vvv,a=a,z=z,sz=sz,t=t, st=st,err=err, n_st=n_st, n_sz=n_sz)
                hst = st/n_st
                hsz = sz/n_sz
                for j_t in range(n_st+1):
                    t_tag = t-st/2. + hst*j_t
                    for j_z in range(n_sz+1):
                        z_tag = z-sz/2. + hsz*j_z
                        y_z[j_z] = hddm.wfpt.full_pdf(rt,v=v,sv=sv*vvv,a=a,z=z_tag,sz=0,t=t_tag, st=0,err=err, n_st=0, n_sz=0)/sz/st
                    y_t[j_t] = simps(y_z, x=None, dx=hsz)
                    res[3+vvv*4] = simps(y_t, x=None, dx=hst)

            print("(%d) rt %f, v: %f, sv: %f, z: %f, sz: %f, t: %f, st: %f a: %f" % (i,rt,v,sv,z,sz,t,st,a))
            print(my_res)
            print(res)
            my_res[np.isinf(my_res)] = 100
            res[np.isinf(res)] = 100
            self.assertTrue(not any(np.isnan(my_res))), "Found NaN in the results"
            self.assertTrue(not any(np.isnan(res))), "Found NaN in the simulated results"
            np.testing.assert_array_almost_equal(my_res, res, 2)


    def test_failure_mode(self):
        rt = 0.6
        for i in range(2):
            rt = rt * -1
            v = 1
            sv = 1
            a = 1.5
            z = 0.5
            sz = 0.2
            t = 0.2
            st = 0.1
            n_st = 10; n_sz =10


            z = 1.1
            self.assertTrue(hddm.wfpt.full_pdf(rt,v=v,sv=sv,a=a,z=z,sz=sz,t=t, st=st,err=1e-10, n_st=10, n_sz=10)==0)
            z = -0.1
            self.assertTrue(hddm.wfpt.full_pdf(rt,v=v,sv=sv,a=a,z=z,sz=sz,t=t, st=st,err=1e-10, n_st=10, n_sz=10)==0)
            z = 0.5

            z = 0.1
            sz = 0.25
            self.assertTrue(hddm.wfpt.full_pdf(rt,v=v,sv=sv,a=a,z=z,sz=sz,t=t, st=st,err=1e-10, n_st=10, n_sz=10)==0)
            z = 0.5

            a = -0.1
            self.assertTrue(hddm.wfpt.full_pdf(rt,v=v,sv=sv,a=a,z=z,sz=sz,t=t, st=st,err=1e-10, n_st=10, n_sz=10)==0)
            a = 1.5

            t = 0.7
            st = 0
            self.assertTrue(hddm.wfpt.full_pdf(rt,v=v,sv=sv,a=a,z=z,sz=sz,t=t, st=st,err=1e-10, n_st=10, n_sz=10)==0)
            t = -0.3
            self.assertTrue(hddm.wfpt.full_pdf(rt,v=v,sv=sv,a=a,z=z,sz=sz,t=t, st=st,err=1e-10, n_st=10, n_sz=10)==0)
            t = 0.1
            st = 0.3
            self.assertTrue(hddm.wfpt.full_pdf(rt,v=v,sv=sv,a=a,z=z,sz=sz,t=t, st=st,err=1e-10, n_st=10, n_sz=10)==0)
            t = 0.2
            st = 0.1

if __name__=='__main__':
    print("Run nosetest.")
