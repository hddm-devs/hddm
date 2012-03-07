import numpy as np
from numpy.random import rand

import unittest

import pymc as pm

import hddm
from hddm.likelihoods import *
from hddm.generate import *
from scipy.integrate import *
from scipy.stats import kstest

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
        from matlab_values import vals

        for v, t, a, z, z_nonorm, rt, err, matlab_wfpt in vals:
            python_wfpt = hddm.wfpt.full_pdf(-rt, v, 0, a, z, 0, t, 0, err, 0)
            print v,t,a,z,z_nonorm,rt,err, matlab_wfpt, python_wfpt
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
            print v,t,a,z,z_nonorm,rt,err, matlab_wfpt, python_wfpt
            np.testing.assert_array_almost_equal(matlab_wfpt, python_wfpt, 9)

    def test_summed_logp(self):
        params = hddm.generate.gen_rand_params(include=('V','Z','T'))

        # Generate random valid RTs
        rts = params['t'] + params['T'] + rand(50)*2
        p = hddm.wfpt.pdf_array(rts, params['v'], params['V'],
                                params['a'], params['z'], params['Z'], params['t'],
                                params['T'], 1e-4, logp=True)

        summed_logp = np.sum(p)

        summed_logp_like = hddm.wfpt.wiener_like(rts, params['v'],
                                                 params['V'], params['a'], params['z'], params['Z'],
                                                 params['t'], params['T'], 1e-4)

        np.testing.assert_almost_equal(summed_logp, summed_logp_like, 2)

        self.assertTrue(-np.Inf == hddm.wfpt.wiener_like(np.array([1.,2.,3.,0.]), 1, 0, 2, .5, 0, 0, 0, 1e-4)), "wiener_like_simple should have returned -np.Inf"

    def test_pdf_V(self, samples=50):
        """Test if our wfpt pdf_V implementation produces the same value as numerical integration over v"""
        func = lambda v_i,value,err,v,V,z,a: hddm.wfpt.full_pdf(value, v_i, 0, a, z, 0, 0, 0, err) * norm.pdf(v_i,v,V)

        for i in range(50):
            V = rand()*0.4+0.1
            v = (rand()-.5)*4
            t = rand()*.5
            a = 1.5+rand()
            z = .5*rand()
            rt = rand()*4 + t
            err = 10**(-3- np.ceil(rand()*12))
            # Test if equal up to the 9th decimal.
            sp_result = quad(func, -np.inf, np.inf, args=(rt,err,v,V,z,a), epsrel=1e-10, epsabs=1e-10)[0]
            hddm_result = hddm.wfpt.full_pdf(rt, v, V, a, z, 0, 0, 0, err)
            np.testing.assert_array_almost_equal(hddm_result, sp_result)

class TestWfptFull(unittest.TestCase):
    def test_adaptive(self):
        for i in range(20):
            V = rand()*0.4+0.1
            v = (rand()-.5)*4
            T = rand()*0.3
            t = rand()*.5+(T/2)
            a = 1.5+rand()
            rt = (rand()*4 + t) * np.sign(rand())
            err = 10**-9
            Z = rand()*0.3
            z = .5*rand()+Z/2
            logp = 0#np.floor(rand()*2)
            nZ = 60
            nT = 60

            my_res = hddm.wfpt.full_pdf(rt,v=v,V=0,a=a,z=z,Z=0,t=t, T=T,err=err, nT=5, nZ=5, use_adaptive=1)
            res = hddm.wfpt.full_pdf(rt,v=v,V=0,a=a,z=z,Z=0,t=t, T=T,err=err, nT=nT, nZ=nZ, use_adaptive=0)

            print "(%d) rt %f, v: %f, V: %f, z: %f, Z: %f, t: %f, T: %f a: %f" % (i,rt,v,V,z,Z,t,T,a)
            print my_res
            print res
            if np.isinf(my_res):
                my_res = 100
            if np.isinf(res):
                res = 100
            self.assertTrue(not np.isnan(my_res)), "Found NaN in the results"
            self.assertTrue(not np.isnan(res)), "Found NaN in the simulated results"
            np.testing.assert_array_almost_equal(my_res, res, 3)


    def test_pdf_integrate_to_one(self):
        for tests in range(2):
            V = rand()*0.4+0.1
            v = (rand()-.5)*4
            T = rand()*0.3
            t = rand()*.5+(T/2)
            a = 1.5+rand()
            err = 10**-8
            Z = rand()*0.3
            z = .5*rand()+Z/2
            func = lambda x: np.exp(hddm.wfpt.wiener_like(np.array([x]), v, V, a, z, Z, t, T, err))
            integ, error = sp.integrate.quad(func, a=-5, b=5)

            np.testing.assert_almost_equal(integ, 1, 2)

    def test_wiener_like_full_single(self):
        for i in range(20):
            V = rand()*0.4+0.1
            v = (rand()-.5)*4
            T = rand()*0.3
            t = rand()*.5+(T/2)
            a = 1.5+rand()
            rt = (rand()*4 + t) * np.sign(rand())
            err = 10**-8
            Z = rand()*0.3
            z = .5*rand()+Z/2
            nZ = 60
            nT = 60

            my_res = np.zeros(8)
            res = np.zeros(8)
            y_z = np.zeros(nZ+1);
            y_t = np.zeros(nT+1)

            for vvv in range(2):
                #test pdf
                my_res[0+vvv*4] = hddm.wfpt.full_pdf(rt,v=v,V=V*vvv,a=a,z=z,Z=0,t=t, T=0,err=err, nT=nT, nZ=nZ)
                res[0+vvv*4]    = hddm.wfpt.full_pdf(rt,v=v,V=V*vvv,a=a,z=z,Z=0,t=t, T=0,err=err, nT=0, nZ=0)

                #test pdf + Z
                my_res[1+vvv*4] = hddm.wfpt.full_pdf(rt,v=v,V=V*vvv,a=a,z=z,Z=Z,t=t, T=0,err=err, nT=nT, nZ=nZ)
                hZ = Z/nZ
                for j in range(nZ+1):
                    z_tag = z-Z/2. + hZ*j
                    y_z[j] = hddm.wfpt.full_pdf(rt,v=v,V=V*vvv,a=a,z=z_tag,Z=0,t=t, T=0,err=err, nT=0, nZ=0)/Z
                    res[1+vvv*4] = simps(y_z, x=None, dx=hZ)

                #test pdf + T
                my_res[2+vvv*4] = hddm.wfpt.full_pdf(rt,v=v,V=V*vvv,a=a,z=z,Z=0,t=t, T=T,err=err, nT=nT, nZ=nZ)
                hT = T/nT
                for j in range(nT+1):
                    t_tag = t-T/2. + hT*j
                    y_t[j] = hddm.wfpt.full_pdf(rt,v=v,V=V*vvv,a=a,z=z,Z=0,t=t_tag, T=0,err=err, nT=0, nZ=0)/T
                    res[2+vvv*4] = simps(y_t, x=None, dx=hT)

                #test pdf + Z + T
                my_res[3+vvv*4] = hddm.wfpt.full_pdf(rt,v=v,V=V*vvv,a=a,z=z,Z=Z,t=t, T=T,err=err, nT=nT, nZ=nZ)
                hT = T/nT
                hZ = Z/nZ
                for j_t in range(nT+1):
                    t_tag = t-T/2. + hT*j_t
                    for j_z in range(nZ+1):
                        z_tag = z-Z/2. + hZ*j_z
                        y_z[j_z] = hddm.wfpt.full_pdf(rt,v=v,V=V*vvv,a=a,z=z_tag,Z=0,t=t_tag, T=0,err=err, nT=0, nZ=0)/Z/T
                    y_t[j_t] = simps(y_z, x=None, dx=hZ)
                    res[3+vvv*4] = simps(y_t, x=None, dx=hT)

            print "(%d) rt %f, v: %f, V: %f, z: %f, Z: %f, t: %f, T: %f a: %f" % (i,rt,v,V,z,Z,t,T,a)
            print my_res
            print res
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
            V = 1
            a = 1.5
            z = 0.5
            Z = 0.2
            t = 0.2
            T = 0.1
            nT = 10; nZ =10


            z = 1.1
            self.assertTrue(hddm.wfpt.full_pdf(rt,v=v,V=V,a=a,z=z,Z=Z,t=t, T=T,err=1e-10, nT=10, nZ=10)==0)
            z = -0.1
            self.assertTrue(hddm.wfpt.full_pdf(rt,v=v,V=V,a=a,z=z,Z=Z,t=t, T=T,err=1e-10, nT=10, nZ=10)==0)
            z = 0.5

            z = 0.1
            Z = 0.25
            self.assertTrue(hddm.wfpt.full_pdf(rt,v=v,V=V,a=a,z=z,Z=Z,t=t, T=T,err=1e-10, nT=10, nZ=10)==0)
            z = 0.5

            a = -0.1
            self.assertTrue(hddm.wfpt.full_pdf(rt,v=v,V=V,a=a,z=z,Z=Z,t=t, T=T,err=1e-10, nT=10, nZ=10)==0)
            a = 1.5

            t = 0.7
            T = 0
            self.assertTrue(hddm.wfpt.full_pdf(rt,v=v,V=V,a=a,z=z,Z=Z,t=t, T=T,err=1e-10, nT=10, nZ=10)==0)
            t = -0.3
            self.assertTrue(hddm.wfpt.full_pdf(rt,v=v,V=V,a=a,z=z,Z=Z,t=t, T=T,err=1e-10, nT=10, nZ=10)==0)
            t = 0.1
            T = 0.3
            self.assertTrue(hddm.wfpt.full_pdf(rt,v=v,V=V,a=a,z=z,Z=Z,t=t, T=T,err=1e-10, nT=10, nZ=10)==0)
            t = 0.2
            T = 0.1



class TestWfptSwitch(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestWfptSwitch, self).__init__(*args, **kwargs)
        self.tests = 2
        self.skip = False
        try:
            import wfpt_switch
        except ImportError:
            self.skip = True

    def gen_rand_params(self):
        vpp = (rand()-.5)*4
        vcc = (rand()-.5)*4
        tcc = rand()*0.3
        t = rand()*.5
        a = 1.+rand()

        return vpp, vcc, tcc, t, a

    @unittest.expectedFailure
    def testDriftDensIntegrateToOne(self):
        if self.skip:
            raise SkipTest("Could not import wfpt_switch.")

        # Test for small tcc where no drifts should have crossed the threshold
        integ, error = sp.integrate.quad(hddm.wfpt_switch.calc_drift_dens, args=(.05, 0, 2, 1, False), a=0, b=2)
        # Not sure why this returns 2, but the resulting likelihood seems to work
        np.testing.assert_almost_equal(integ, 1, 2)

    def test_pdf_integrate_to_one_precomp(self):
        if self.skip:
            raise SkipTest("Could not import wfpt_switch.")

        for tests in range(self.tests):
            vpp, vcc, tcc, t, a = self.gen_rand_params()

            func = lambda x: np.exp(hddm.wfpt_switch.wiener_like_antisaccade_precomp(np.array([x]), np.array([1]), vpp, vcc, 0, a, .5, t, tcc, 0, 1e-4))
            integ, error = sp.integrate.quad(func, a=-5, b=5)

            np.testing.assert_almost_equal(integ, 1, 2)

    def test_pdf_integrate_to_one(self):
        if self.skip:
            raise SkipTest("Could not import wfpt_switch.")

        for tests in range(self.tests):
            vpp, vcc, tcc, t, a = self.gen_rand_params()

            func = lambda x: np.exp(hddm.wfpt_switch.wiener_like_antisaccade(np.array([x]), np.array([1]), vpp, vcc, 0, a, .5, t, tcc, 0, 1e-6))
            integ, error = sp.integrate.quad(func, a=-5, b=5)

            np.testing.assert_almost_equal(integ, 1, 2)


    def test_pdf_precomp_integrate_to_one(self):
        if self.skip:
            raise SkipTest("Could not import wfpt_switch.")

        for tests in range(self.tests):
            vpp, vcc, tcc, t, a = self.gen_rand_params()
            func = lambda x: np.exp(hddm.wfpt_switch.wiener_like_antisaccade_precomp(np.array([x]), np.array([1]), vpp, vcc, 0, a, .5, t, tcc, 0, 1e-6))
            integ, error = sp.integrate.quad(func, a=-5, b=5)

            np.testing.assert_almost_equal(integ, 1, 2)

    def test_ks(self):
        if self.skip:
            raise SkipTest("Could not import wfpt_switch.")

        for tests in range(self.tests):
            vpp, vcc, tcc, t, a = self.gen_rand_params()
            tcc += .1 # Test for bigger tcc
            sampler = hddm.likelihoods.wfpt_switch

            [D, p_value] = kstest(sampler.rvs, sampler.cdf,
                                  args=(vpp, vcc, 0, a, .5, t, tcc, 0), N=1000)

            print 'p_value: %f' % p_value
            self.assertTrue(p_value > 0.05)

    def test_ks_small_tcc(self):
        if self.skip:
            raise SkipTest("Could not import wfpt_switch.")

        for tests in range(self.tests):
            vpp, vcc, tcc, t, a = self.gen_rand_params()
            tcc = 0.02

            sampler = hddm.likelihoods.wfpt_switch

            [D, p_value] = kstest(sampler.rvs, sampler.cdf,
                                  args=(vpp, vcc, 0, a, .5, t, tcc, 0), N=1000)

            print 'p_value: %f' % p_value
            self.assertTrue(p_value > 0.05)


    def test_ks_precomp_small_tcc(self):
        if self.skip:
            raise SkipTest("Could not import wfpt_switch.")

        for tests in range(self.tests):
            vpp, vcc, tcc, t, a = self.gen_rand_params()
            tcc = 0.02

            sampler = wfpt_switch_precomp

            [D, p_value] = kstest(sampler.rvs, sampler.cdf,
                                  args=(vpp, vcc, 0, a, .5, t, tcc, 0), N=1000)

            print 'p_value: %f' % p_value
            self.assertTrue(p_value > 0.05)

    def test_ks_precomp(self):
        if self.skip:
            raise SkipTest("Could not import wfpt_switch.")

        for tests in range(self.tests):
            vpp, vcc, tcc, t, a = self.gen_rand_params()
            tcc += 0.1

            sampler = wfpt_switch_precomp

            [D, p_value] = kstest(sampler.rvs, sampler.cdf,
                                  args=(vpp, vcc, 0, a, .5, t, tcc, 0), N=1000)

            print 'p_value: %f' % p_value
            self.assertTrue(p_value > 0.05)


#class wfpt_switch_precomp_gen(hddm.likelihoods.wfpt_switch_gen):
#    """Helper function for testing wiener_like_antisaccade_precomp."""
#    def _pdf(self, x, v, v_switch, V_switch, a, z, t, t_switch, T):
#        if np.isscalar(x):
#            out = np.exp(hddm.wfpt_switch.wiener_like_antisaccade_precomp(np.array([x]), np.array([1]), v, v_switch, V_sw#itch, a, z, t, t_switch, T, 1e-4, evals=100))
#        else:
#            out = np.empty_like(x)
#            for i in xrange(len(x)):
#                out[i] = np.exp(hddm.sandbox.model.wiener_like_antisaccade_precomp(np.array([x[i]]), np.array([1]), v[i], v_switch[i], V_switch[i], a[i], z[i], t[i], t_switch[i], T[i], 1e-4))

#        return out

wfpt_switch_precomp = hddm.sandbox.model.wfpt_switch_like

if __name__=='__main__':
    print "Run nosetest."
