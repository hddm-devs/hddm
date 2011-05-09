from __future__ import division
from copy import copy
import platform
import pymc as pm
import numpy as np
np.seterr(divide='ignore')

import hddm

def wiener_like_simple(value, v, z, t, a):
    """Log-likelihood for the simple DDM"""
    return hddm.wfpt.wiener_like_simple(value, v, a, z, t, err=.0001)

@pm.randomwrap
def wiener_simple(v, z, t, a, size=None):
    rts = hddm.generate.gen_rts(params={'v':v, 'z':z, 't':t, 'a':a, 'Z':0, 'V':0, 'T':0}, samples=size)
    print rts
    return rts

WienerSimple = pm.stochastic_from_dist(name="Wiener Simple Diffusion Process",
                                       logp=wiener_like_simple,
                                       random=wiener_simple,
                                       dtype=np.float,
                                       mv=True)

def wiener_like_simple_contaminant(value, cont_x, v, a, z, t, err=.0001):
    """Log-likelihood for the simple DDM including contaminants"""
    return hddm.wfpt.wiener_like_simple_contaminant(value, cont_x, v, a, z, t, 0, 7, err)

WienerSimpleContaminant = pm.stochastic_from_dist(name="Wiener Simple Diffusion Process",
                                       logp=wiener_like_simple_contaminant,
                                       dtype=np.float,
                                       mv=True)

def wiener_like_simple_collCont(value, cont_x, cont_y, v, a, z, t, err=.0001):
    """Log-likelihood for the simple DDM including contaminants"""
    return hddm.wfpt.wiener_like_simple_collCont(value, cont_x, cont_y, v, a, z, t, 0, 7, err)

WienerSimpleCollCont = pm.stochastic_from_dist(name="Wiener Simple Diffusion Process",
                                       logp=wiener_like_simple_contaminant,
                                       dtype=np.float,
                                       mv=True)


def wiener_like_full_collCont(value, cont_x, gamma, v, V, a, z, Z, t, T, err=.0001):
    """Log-likelihood for the DDM with collapsed contaminants"""
    return hddm.wfpt.wiener_like_full_collCont(value, cont_x, gamma, v, V, a, z, Z, t, T, 0, 7, err)

WienerFullCollCont = pm.stochastic_from_dist(name="Wiener CollCont Diffusion Process",
                                       logp=wiener_like_full_collCont,
                                       dtype=np.float,
                                       mv=True)


def wiener_like_simple_multi(value, v, a, z, t, multi=None):
    """Log-likelihood for the simple DDM"""
    return np.sum(hddm.wfpt.pdf_array_multi(value, v, a, z, t, .001, logp=1, multi=multi))
            
WienerSimpleMulti = pm.stochastic_from_dist(name="Wiener Simple Diffusion Process",
                                            logp=wiener_like_simple_multi,
                                            dtype=np.float,
                                            mv=True)
@pm.randomwrap
def wiener_full(v, z, t, a, V, Z, T, size=None):
    return hddm.generate.gen_rts(params={'v':v, 'z':z, 't':t, 'a':a, 'Z':Z, 'V':V, 'T':T}, samples=size)

def wiener_like_full_mc(value, v, V, z, Z, t, T, a):
    """Log-likelihood for the full DDM using the sampling method"""
    return np.sum(hddm.wfpt_full.wiener_like_full_mc(value, v, V, z, Z, t, T, a, err=.0001, reps=10, logp=1))
 
WienerFullMc = pm.stochastic_from_dist(name="Wiener Diffusion Process",
                                       logp=wiener_like_full_mc,
                                       random=wiener_full,
                                       dtype=np.float,
                                       mv=True)

def wiener_like_full_intrp(value, v, V, z, Z, t, T, a, err=1e-5, nT=5, nZ=5, use_adaptive=1, simps_err=1e-8):
    """Log-likelihood for the full DDM using the interpolation method"""
    return hddm.wfpt_full.wiener_like_full_intrp(value, v, V, a, z, Z, t, T, err, nT, nZ, use_adaptive,  simps_err)


def general_WienerFullIntrp_variable(err=1e-5, nT=5, nZ=5, use_adaptive=1, simps_err=1e-8):
    _like = lambda  value, v, V, z, Z, t, T, a, err=err, nT=nT, nZ=nZ, \
    use_adaptive=use_adaptive, simps_err=simps_err: \
    wiener_like_full_intrp(value, v, V, z, Z, t, T, a,\
                            err=err, nT=nT, nZ=nZ, use_adaptive=use_adaptive, simps_err=simps_err)
    _like.__doc__ = wiener_like_full_intrp.__doc__
    return pm.stochastic_from_dist(name="Wiener Diffusion Process",
                                       logp=_like,
                                       random=wiener_full,
                                       dtype=np.float,
                                       mv=True)
 
WienerFullIntrp = pm.stochastic_from_dist(name="Wiener Diffusion Process",
                                       logp=wiener_like_full_intrp,
                                       random=wiener_full,
                                       dtype=np.float,
                                       mv=True)



def wiener_like_full_mc_multi_thresh(value, v, V, z, Z, t, T, a):
    """Log-likelihood for the full DDM using the sampling method"""
    return np.sum(hddm.wfpt_full.wiener_like_full_mc_multi_thresh(value, v, V, z, Z, t, T, a, reps=10, logp=1))

WienerFullMcMultiThresh = pm.stochastic_from_dist(name="Wiener Diffusion Process",
                                       logp=wiener_like_full_mc_multi_thresh,
                                       random=wiener_full,
                                       dtype=np.float,
                                       mv=True)

        
def wiener_like_single_trial(value, v, a, z, t):
    """Log-likelihood of the DDM for one RT point."""
    prob = hddm.wfpt.wiener_like_full(value, np.asarray(v), np.asarray(a), np.asarray(z), np.asarray(t), err=0.001)
    return prob

WienerSingleTrial = pm.stochastic_from_dist(name="Wiener Diffusion Process",
                                            logp=wiener_like_single_trial,
                                            random=wiener_simple,
                                            dtype=np.float,
                                            mv=True)


def centeruniform_like(value, center, width):
    R"""Likelihood of centered uniform"""
    return pm.uniform_like(value,
                           lower=np.asarray(center)-(np.asarray(width)/2.), 
                           upper=np.asarray(center)+(np.asarray(width)/2.))

@pm.randomwrap
def centeruniform(center, width, size=1):
    R"""Sample from centered uniform"""
    return np.random.uniform(low=np.asarray(center)-(np.asarray(width)/2.), 
                             high=np.asarray(center)+(np.asarray(width)/2.))

CenterUniform = pm.stochastic_from_dist(name="Centered Uniform",
                                        logp=centeruniform_like,
                                        random=centeruniform,
                                        dtype=np.float,
                                        mv=True)


def wiener_like_antisaccade(value, instruct, v, v_switch, a, z, t, t_switch, err=1e-4):
    """Log-likelihood for the simple DDM including contaminants"""
    return hddm.wfpt_switch.wiener_like_antisaccade(value, np.asarray(instruct), v, v_switch, a, z, t, t_switch, err)

WienerAntisaccade = pm.stochastic_from_dist(name="Wiener Simple Diffusion Process",
                                            logp=wiener_like_antisaccade,
                                            dtype=np.float,
                                            mv=True)

################################
# Linear Ballistic Accumulator

def LBA_like(value, a, z, t, V, v0, v1=0., logp=True, normalize_v=False):
    """Linear Ballistic Accumulator PDF
    """
    if z is None:
        z = a/2.

    #print a, z, t, v, V
    prob = hddm.lba.lba_like(np.asarray(value, dtype=np.double), z, a, t, V, v0, v1, int(logp), int(normalize_v))
    return prob
    
LBA = pm.stochastic_from_dist(name='LBA likelihood',
                              logp=LBA_like,
                              dtype=np.float,
                              mv=True)

# Scipy Distributions
from scipy import stats, integrate

def expectedfunc(self, fn=None, args=(), lb=None, ub=None, conditional=False):
    '''calculate expected value of a function with respect to the distribution

    only for standard version of distribution,
    location and scale not tested

    Parameters
    ----------
        all parameters are keyword parameters
        fn : function (default: identity mapping)
           Function for which integral is calculated. Takes only one argument.
        args : tuple
           argument (parameters) of the distribution
        lb, ub : numbers
           lower and upper bound for integration, default is set to the support
           of the distribution
        conditional : boolean (False)
           If true then the integral is corrected by the conditional probability
           of the integration interval. The return value is the expectation
           of the function, conditional on being in the given interval.

    Returns
    -------
        expected value : float
    '''
    if fn is None:
        def fun(x, *args):
            return x*self.pdf(x, *args)
    else:
        def fun(x, *args):
            return fn(x)*self.pdf(x, *args)
    if lb is None:
        lb = self.a
    if ub is None:
        ub = self.b
    if conditional:
        invfac = self.sf(lb,*args) - self.sf(ub,*args)
    else:
        invfac = 1.0
    return integrate.quad(fun, lb, ub,
                                args=args)[0]/invfac

import types
stats.distributions.rv_continuous.expectedfunc = types.MethodType(expectedfunc,None,stats.distributions.rv_continuous)

class lba_gen(stats.distributions.rv_continuous):
    def _pdf(self, x, a, z, t, V, v0, v1):
        """Linear Ballistic Accumulator PDF
        """
        return np.asscalar(hddm.lba.lba_like(np.asarray(x, dtype=np.double), z, a, t, V, v0, v1))

lba = lba_gen(a=0, b=5, name='LBA', longname="""Linear Ballistic Accumulator likelihood function.""", extradoc="""Linear Ballistic Accumulator likelihood function. Models two choice decision making as a race between two independet linear accumulators towards one threshold. Once one crosses the threshold, an action with the corresponding RT is performed.

Parameters:
***********
z: width of starting point distribution
a: threshold
t: non-decision time
V: inter-trial variability in drift-rate
v0: drift-rate of first accumulator
v1: drift-rate of second accumulator

References:
***********
The simplest complete model of choice response time: linear ballistic accumulation.
Brown SD, Heathcote A; Cogn Psychol. 2008 Nov ; 57(3): 153-78 

Getting more from accuracy and response time data: methods for fitting the linear ballistic accumulator.
Donkin C, Averell L, Brown S, Heathcote A; Behav Res Methods. 2009 Nov ; 41(4): 1095-110 
""")

class wfpt_gen(stats.distributions.rv_continuous):
    def _pdf(self, x, v, a, z, t):
        return hddm.wfpt.pdf(x, v, a, z, t, err=.0001)
    
    def _rvs(self, v, a, z, t):
        return gen_ddm_rts(v=v, z=z, t=t, a=a, Z=0, V=0, T=0, size=self._size)

wfpt = wfpt_gen(a=0, b=5, name='wfpt', longname="""Wiener likelihood function""", extradoc="""Wfpt likelihood function of the Ratcliff Drift Diffusion Model (DDM). Models two choice decision making tasks as a drift process that accumulates evidence across time until it hits one of two boundaries and executes the corresponding response. Implemented using the Navarro & Fuss (2009) method.

Parameters:
***********
v: drift-rate
a: threshold
z: bias [0,1]
t: non-decision time

References:
***********
Fast and accurate calculations for first-passage times in Wiener diffusion models
Navarro & Fuss - Journal of Mathematical Psychology, 2009 - Elsevier
""")
