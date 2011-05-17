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

def pdf_with_params(rt, params):
    v = params['v']; V= params['V']; z = params['z']; Z = params['Z']; t = params['t'];
    T = params['T']; a = params['a']
    return hddm.wfpt_full.full_pdf(rt,v=v,V=V,a=a,z=z,Z=Z,t=t, 
                        T=T,err=1e-10, nT=2, nZ=2, use_adaptive=1, simps_err=1e-3)         

def create_wfpt_cdf(params, range_ = (-10,10), bins=5000):
    x = np.linspace(range_[0], range_[1], bins)
    pdf = [pdf_with_params(rt, params) for rt in x]
    l_cdf = np.cumsum(pdf)
    l_cdf = l_cdf/l_cdf[-1]
    def cdf(rt, x=x, l_cdf=l_cdf):
        idx = np.where(x>rt)[0][0]
        l_cdf[idx]
        m = (l_cdf[idx] - l_cdf[idx-1])/(x[idx]-x[idx-1])
        b = l_cdf[idx] - m*x[idx]
        return m*rt+b
    return cdf


class TestWfpt(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestWfpt, self).__init__(*args, **kwargs)
        self.samples=5000
        self.range_ = (-10,10)

    def test_pdf(self):
        # Test if our wfpt pdf implementation yields the same results as the reference implementation by Navarro & Fuss 2009
        try:
            import mlabwrap
        except ImportError:
            print "Could not import mlabwrap, not performing pdf comparison test."
            return

        for i in range(500):
            v = (rand()-.5)*1.5
            t = rand()*.5
            a = 1.5+rand()
            z = .5*rand()
            z_nonorm = a*z
            rt = rand()*4 + t
            err = 10*(-3- ceil(rand()*20))
            # Test if equal up to the 9th decimal.
            matlab_wfpt = mlabwrap.mlab.wfpt(rt, v, a, z_nonorm, err)[0][0]
            python_wfpt = hddm.wfpt.pdf(rt, v, a, z, err)
            print v,t,a,z,z_nonorm,rt,err, matlab_wfpt, python_wfpt
            np.testing.assert_array_almost_equal(matlab_wfpt, python_wfpt, 9)
            
   
    def test_compare_sampled_data_to_analytic(self):
        excludes = [['Z','T','V'],['Z','T'],['V'],['T'],['Z']]
        for i_exclude in excludes:
            ok = False
            while not ok:
                params = hddm.diag.rand_params(model_type='full_intrp', exclude=i_exclude)            
                samples = hddm.generate.gen_rts(params, samples=self.samples, method='cdf', dt=1e-3)
                if max(abs(samples))<=self.range_[1]:
                    ok = True                
            cdf_single = create_wfpt_cdf(params, self.range_)
            cdf_vec =lambda rts:[cdf_single(x) for x in rts]        
            [D, p_value] = kstest(samples, cdf_vec)
            print 'p_value: %f' % p_value
            self.assertTrue(p_value > 0.1)

    def test_compare_drift_simulated_data_to_analytic(self):
        excludes = [['Z','T','V'],['Z','T'],['V'],['T'],['Z']]
        for i_exclude in excludes:
            ok = False
            while not ok:
                params = hddm.diag.rand_params(model_type='full_intrp', exclude=i_exclude)            
                samples = hddm.generate.gen_rts(params, samples=self.samples, method='drift', dt=1e-5)
                if max(abs(samples))<=self.range_[1]:
                    ok = True                
            cdf_single = create_wfpt_cdf(params, self.range_)
            cdf_vec =lambda rts:[cdf_single(x) for x in rts]        
            [D, p_value] = kstest(samples, cdf_vec)
            print 'p_value: %f' % p_value
            self.assertTrue(p_value > 0.1)


    def test_simple_summed_logp(self):
        v = (rand()-.5)*1.5
        t = rand()*.5
        a = 1.5+rand()
        z = .5

        # Test for if sum is the same

        # Generate random valid RTs
        rts = t + rand(5000)*2
        p = [hddm.wfpt.pdf_sign(rt, v, a, z, t, 1e-4) for rt in rts]
        summed_logp = np.sum(np.log(p))

        self.assertTrue(summed_logp == hddm.wfpt.wiener_like_simple(np.array(rts), v, a, z, t, 1e-4)), "Summed logp does not match"

        self.assertTrue(-np.Inf == hddm.wfpt.wiener_like_simple(np.array([1.,2.,3.,0.]), v, a, z, t+.1, 1e-4)), "wiener_like_simple should have returned -np.Inf"
        
            
        
            
    def test_pdf_V(self):
        """Test if our wfpt pdf_V implementation yields the right results"""       
        func = lambda v_i,value,err,v,V,z,a: hddm.wfpt.pdf(value, v_i, a, z, err) *norm.pdf(v_i,v,V)

        for i in range(50):
            V = rand()*0.4+0.1
            v = (rand()-.5)*4
            t = rand()*.5
            a = 1.5+rand()
            z = .5*rand()
            rt = rand()*4 + t
            err = 10**(-3- np.ceil(rand()*12))
            # Test if equal up to the 9th decimal.
            res =  quad(func, -np.inf,np.inf,args = (rt,err,v,V,z,a), epsrel=1e-10, epsabs=1e-10)[0]
            np.testing.assert_array_almost_equal(hddm.wfpt_full.pdf_V(rt, v=v, V=V, a=a, z=z, err=err), res)

class TestWfptFull(unittest.TestCase):


    def test_adaptive(self):
          
        for i in range(200):
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

            my_res = hddm.wfpt_full.full_pdf(rt,v=v,V=0,a=a,z=z,Z=0,t=t, T=T,err=err, nT=5, nZ=5, use_adaptive=1)
            res = hddm.wfpt_full.full_pdf(rt,v=v,V=0,a=a,z=z,Z=0,t=t, T=T,err=err, nT=nT, nZ=nZ, use_adaptive=0)
            
            print "(%d) rt %f, v: %f, V: %f, z: %f, Z: %f, t: %f, T: %f a: %f" % (i,rt,v,V,z,Z,t,T,a)
            print my_res
            print res
            if np.isinf(my_res):
                my_res = 100
            if np.isinf(res):
                res = 100            
            self.assertTrue(not np.isnan(my_res)), "Found NaN in the results"
            self.assertTrue(not np.isnan(res)), "Found NaN in the simulated results"                                                                                    
            np.testing.assert_array_almost_equal(my_res, res,6)

    
    def test_full_pdf(self):
        for i in range(200):
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
                my_res[0+vvv*4] = hddm.wfpt_full.full_pdf(rt,v=v,V=V*vvv,a=a,z=z,Z=0,t=t, T=0,err=err, nT=nT, nZ=nZ)
                res[0+vvv*4]    = hddm.wfpt_full.full_pdf(rt,v=v,V=V*vvv,a=a,z=z,Z=0,t=t, T=0,err=err, nT=0, nZ=0)
                
                #test pdf + Z
                my_res[1+vvv*4] = hddm.wfpt_full.full_pdf(rt,v=v,V=V*vvv,a=a,z=z,Z=Z,t=t, T=0,err=err, nT=nT, nZ=nZ)
                hZ = Z/nZ
                for j in range(nZ+1):
                    z_tag = z-Z/2. + hZ*j
                    y_z[j] = hddm.wfpt_full.full_pdf(rt,v=v,V=V*vvv,a=a,z=z_tag,Z=0,t=t, T=0,err=err, nT=0, nZ=0)/Z                             
                    res[1+vvv*4] = simps(y_z, x=None, dx=hZ)
                    
                #test pdf + T
                my_res[2+vvv*4] = hddm.wfpt_full.full_pdf(rt,v=v,V=V*vvv,a=a,z=z,Z=0,t=t, T=T,err=err, nT=nT, nZ=nZ)
                hT = T/nT
                for j in range(nT+1):
                    t_tag = t-T/2. + hT*j
                    y_t[j] = hddm.wfpt_full.full_pdf(rt,v=v,V=V*vvv,a=a,z=z,Z=0,t=t_tag, T=0,err=err, nT=0, nZ=0)/T      
                    res[2+vvv*4] = simps(y_t, x=None, dx=hT)
             
                #test pdf + Z + T
                my_res[3+vvv*4] = hddm.wfpt_full.full_pdf(rt,v=v,V=V*vvv,a=a,z=z,Z=Z,t=t, T=T,err=err, nT=nT, nZ=nZ)
                hT = T/nT
                hZ = Z/nZ
                for j_t in range(nT+1):
                    t_tag = t-T/2. + hT*j_t
                    for j_z in range(nZ+1):
                        z_tag = z-Z/2. + hZ*j_z
                        y_z[j_z] = hddm.wfpt_full.full_pdf(rt,v=v,V=V*vvv,a=a,z=z_tag,Z=0,t=t_tag, T=0,err=err, nT=0, nZ=0)/Z/T    
                    y_t[j_t] = simps(y_z, x=None, dx=hZ)             
                    res[3+vvv*4] = simps(y_t, x=None, dx=hT)
                
            print "(%d) rt %f, v: %f, V: %f, z: %f, Z: %f, t: %f, T: %f a: %f" % (i,rt,v,V,z,Z,t,T,a)
            print my_res
            print res
            my_res[np.isinf(my_res)] = 100
            res[np.isinf(res)] = 100            
            self.assertTrue(not any(np.isnan(my_res))), "Found NaN in the results"
            self.assertTrue(not any(np.isnan(res))), "Found NaN in the simulated results"                                                                                    
            np.testing.assert_array_almost_equal(my_res, res,5)
            
        
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
            self.assertTrue(hddm.wfpt_full.full_pdf(rt,v=v,V=V,a=a,z=z,Z=Z,t=t, T=T,err=1e-10, nT=10, nZ=10)==0)
            z = -0.1
            self.assertTrue(hddm.wfpt_full.full_pdf(rt,v=v,V=V,a=a,z=z,Z=Z,t=t, T=T,err=1e-10, nT=10, nZ=10)==0)
            z = 0.5
            
            z = 0.1
            Z = 0.25
            self.assertTrue(hddm.wfpt_full.full_pdf(rt,v=v,V=V,a=a,z=z,Z=Z,t=t, T=T,err=1e-10, nT=10, nZ=10)==0)
            z = 0.5
        
        
            a = -0.1
            self.assertTrue(hddm.wfpt_full.full_pdf(rt,v=v,V=V,a=a,z=z,Z=Z,t=t, T=T,err=1e-10, nT=10, nZ=10)==0)
            a = 1.5
            
            t = 0.7
            T = 0
            self.assertTrue(hddm.wfpt_full.full_pdf(rt,v=v,V=V,a=a,z=z,Z=Z,t=t, T=T,err=1e-10, nT=10, nZ=10)==0)            
            t = -0.3
            self.assertTrue(hddm.wfpt_full.full_pdf(rt,v=v,V=V,a=a,z=z,Z=Z,t=t, T=T,err=1e-10, nT=10, nZ=10)==0)          
            t = 0.1
            T = 0.3
            self.assertTrue(hddm.wfpt_full.full_pdf(rt,v=v,V=V,a=a,z=z,Z=Z,t=t, T=T,err=1e-10, nT=10, nZ=10)==0)
            t = 0.2
            T = 0.1

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
        try:
            import rpy2.robjects as robjects
            import rpy2.robjects.numpy2ri
            robjects.r.source('lba-math.r')
        except ImportError:
            raise SkipTest("rpy2 not installed, not testing against reference implementation.")

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
