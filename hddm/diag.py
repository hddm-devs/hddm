
from copy import copy
import numpy as np
import pymc as pm
np.seterr(divide='ignore')

from sys import stdout
import hddm
from scipy.stats import scoreatpercentile
from time import time
from operator import attrgetter

def check_model(model, params_true, assert_=False, conf_interval = 95):
    """calculate the posterior estimate error if hidden parameters are known (e.g. when simulating data)."""

    print("checking estimation with %d confidence interval" % conf_interval)
    fail = False
    nodes = sorted(model.stochastics, key=attrgetter('__name__'))
    nodes = [x for x in nodes if x.shape == ()]
    for node in nodes:
        if node.__name__ not in params_true:
            continue # Skip non-existent params
        name = node.__name__
        trace = model.db.trace(name)[:]
        est = np.mean(trace)
        node_median = np.median(trace)
        truth = params_true[name]
        lb = (50 - conf_interval/2.)
        lb_score = scoreatpercentile(trace, lb)
        ub = (50 + conf_interval/2.)
        ub_score = scoreatpercentile(trace, ub)
        fell = np.sum(truth > trace)*1./len(trace) * 100
        if lb_score==ub_score:
            fell = 50


        print("%5s: Truth: %6.3f, mean: %6.3f, median: %6.3f, lb: %6.3f, ub: %6.3f,  fell in: %.2f" % \
        (name, truth, est, node_median, lb_score, ub_score, fell))
        if (fell < lb) or (fell > ub):
            fail = True
            print("the true value of %s is outside of the confidence interval !*!*!*!*!*!*!" % name)

    if assert_:
        assert (fail==False)

    ok = not fail
    return ok

def check_rejection(model, assert_=True):
    """check if the rejection ratio is not too high"""

    for node in model.stochastics:
        name = node.__name__
        trace = node.trace()[:]
        rej =  np.sum(np.diff(trace)==0)
        rej_ratio = rej*1.0/len(trace)
        print("rejection ratio for %s: %.3f" %(name, rej_ratio))
        if (rej_ratio < 0.5) or (rej_ratio > 0.8):
            msg = "%s still need to be tuned" % name
            if assert_:
                assert 1==0, msg
            else:
                print(msg)


def test_params_on_data(params, data, include=(), depends_on=None, conf_interval=95):
    thin = 1
    samples = 10000
    burn = 10000
    n_iter = burn + samples*thin
    stdout.flush()
    if depends_on is None:
        depends_on = {}
    if 'pi' in include or 'gamma' in include:
        m_hddm = hddm.HDDMContaminant(data, bias=True, depends_on=depends_on)
    else:
        m_hddm = hddm.HDDM(data, bias=True, include=include, depends_on=depends_on)
    model = m_hddm.mcmc()
    #[model.use_step_method(pm.Metropolis, x,proposal_sd=0.1) for x in model.stochastics]
    i_t = time()
    model.sample(n_iter, burn=burn, thin=thin)
    print("sampling took: %.2f seconds" % (time() - i_t))
    ok = True
    if check_model(model, params, assert_=False, conf_interval = conf_interval)==False:
        print("model checking failed. running again")
        stdout.flush()
        model.sample(n_iter, burn=burn, thin=thin)
        if check_model(model, params, assert_=False, conf_interval = conf_interval)==False:
            print("model checking failed again !!!!!!!!!!!!!!!!!!!!!!!")
            ok  = False

    res = {}
    res['params'] = params
    res['data'] = data
    res['mc'] = model
    check_rejection(model, assert_ = False)
    check_correl(model)
    stdout.flush()
    return ok, res

def run_accuracy_test(nTimes=20, include=(), stop_when_fail = True):
    """ run accuracy test nTime times"""
    n_data = 300
    for i_time in range(nTimes):
        params = hddm.generate.gen_rand_params(include)
        data,temp = hddm.generate.gen_rand_data(params, samples=n_data)
        positive = sum(data['response'])
        print("generated %d data_points (%d positive %d negative)" % (len(data), positive, len(data) - positive))
        print("testing params: a:%.3f, t:%.3f, v:%.3f, z: %.3f, T: %.3f, V: %.3f Z: %.3f" \
        % (params['a'], params['t'], params['v'], params['z'], params['st'], params['sv'], params['sz']))
        ok, res = test_params_on_data(params, data, include=include)

        if stop_when_fail and not ok:
            return res
    return {}


def str_params(params):
    s = ''
    keys = list(params.keys())
    keys.sort(reverse=True)
    for name in keys:
        s = s + "%s: %.3f, " % (name, params[name])
    s = s[:-2] + "\n"
    return s


def gen_cond_data_and_params(n_data,  n_conds = 3, include = ()):
    params = hddm.generate.gen_rand_params(include)
    params_set = {}
    params_true = copy(params)
    all_v = np.linspace(min(0,params['v']/2) , max(params['v']*2, 3), n_conds)
    del params_true['v']
    for i in range(n_conds):
        params_set[i] = copy(params)
        params_set[i]['v'] = all_v[i]
        params_true['v(%d,)'%i] = all_v[i]


    cond_data, temp = hddm.generate.gen_rand_data(params_set, samples=int(n_data/n_conds))
    positive = sum(cond_data['response'])
    print("generated %d data_points (%d positive %d negative)" % \
    (len(cond_data), positive, len(cond_data) - positive))
    print("used params: %s" % str_params(params_true))
    stdout.flush()
    return cond_data, params_true

def run_simple_test(nTimes=20, stop_when_fail = False):
    return run_accuracy_test(nTimes)

def check_correl(model):
    nodes = [x for x in model.stochastics if x.shape == ()]
    threshold = 0.05
    fail = False
    for node in nodes:
        t_lag = np.inf
        for lag in range(1,min(101, len(node.trace()[:])//2)):
            corr = pm.diagnostics.autocorr(node.trace()[:], lag)
            if corr <= threshold:
                t_lag=  lag
                break
        if t_lag < np.inf:
            fail = True
            print("%s: correlation drop under %f after %d steps" %(node.__name__ , threshold, t_lag))
        else:
            print("%s: correlation don't drop under %f!!!!" %(node.__name__ , threshold))
    ok = not fail
    return ok

def test_acc_full_intrp(include = (), n_conds = 6, use_db=False):
    from numpy.random import rand
    burn = 10000
    thin = 1
    n_samples = 10000
    n_iter = n_samples*thin

    all_wp = []
    all_wp = all_wp + [{'err': 1e-5, 'nT':3, 'nZ':3, 'use_adaptive':1, 'simps_err':1e-5}]
    all_wp = all_wp + [{'err': 1e-5, 'nT':2, 'nZ':2, 'use_adaptive':1, 'simps_err':1e-4}]
    all_wp = all_wp + [{'err': 1e-4, 'nT':2, 'nZ':2, 'use_adaptive':1, 'simps_err':1e-3}]

    initial_params = hddm.generate.gen_rand_params(include=include)
    full_params = copy(initial_params)
    params_set = {}
    v_0 = rand()
    all_v = np.linspace(v_0, max(4,v_0*n_conds), n_conds)
    for j in range(n_conds):
        params_set[j] = copy(initial_params)
        params_set[j]['v'] = all_v[j]
        full_params['v(%d,)'%j] = params_set[j]['v']

    data, params = hddm.generate.gen_rand_data(params_set, samples=150)

    print("Using the following params: \n %s" % str_params(full_params))

    i_res={}
    i_res['params'] = copy(full_params)
    i_res['all_wp'] = all_wp
    i_res['data'] = data
    i_res['sample_time'] = [None]*len(all_wp)
    i_res['init_time'] = [None]*len(all_wp)
    i_res['burn_time'] = [None]*len(all_wp)
    i_res['stats'] = [None]*len(all_wp)
    i_res['logp'] = [None]*len(all_wp)
    i_res['dbname'] = [None]*len(all_wp)
    i_res['mc'] = [None]*len(all_wp)

    for i_params in range(len(all_wp)):
        print("working on model %d" % i_params)

        model = hddm.model.HDDM(data, bias=True, wiener_params=all_wp[i_params],
                                include = include, depends_on  = {'v':['condition']})#, init_value=params)
        i_t = time()
        if use_db:
            dbname = 'speed.' + str(time()) + '.db'
            i_res['dbname'][i_params] = dbname[:]
        else:
            dbname = None

        nodes = model.create()
        mc = pm.MCMC(nodes)
        i_res['mc'][i_params] = mc
        [mc.use_step_method(pm.Metropolis, x, proposal_sd=0.5) for x in mc.stochastics]

        i_t = time()
        mc.sample(burn+1, burn)
        d_time = time() - i_t;
        i_res['burn_time'][i_params] = d_time
        print("burn phase took %f secs" % d_time)

        i_t = time()
        mc.sample(n_iter, 0, thin=thin)
        d_time = time() - i_t;
        i_res['sample_time'][i_params] = d_time

        print("sampling took in %f secs" % d_time)
        stdout.flush()

        check_model(mc, full_params, assert_=False, conf_interval = 95)
        check_rejection(mc, assert_=False)
        check_correl(mc)

        if dbname is not None:
            mc.db.commit()
            mc.db.close()

    return i_res

def check_geweke(model, assert_=True):
    # Test for convergence using geweke method
    for param in model.group_params.values():
        geweke = np.array(pm.geweke(param))
        if assert_:
            assert (np.any(np.abs(geweke[:,1]) < 2)), 'Chain of %s not properly converged'%param
            return False
        else:
            if np.any(np.abs(geweke[:,1]) > 2):
                print("Chain of %s not properly converged" % param)
                return False

    return True

