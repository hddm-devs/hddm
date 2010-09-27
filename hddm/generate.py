import numpy as np
import numpy.lib.recfunctions as rec
from copy import copy

try:
    from IPython.Debugger import Tracer; debug_here = Tracer()
except:
    pass


################################################################
# Functions to generate RT distributions with known parameters #
################################################################
def _gen_rts_fastdm(v=0, sv=0, z=0.5, sz=0, a=1, ter=0.3, ster=0, num_samples=500, fname=None, structured=True):
    """Generate simulated RTs with fixed parameters."""
    if fname is None:
        fname = 'example_DDM.txt'
    subprocess.call([sampler_exec, '-v', str(v), '-V', str(sv), '-z', str(z), '-Z', str(sz), '-a', str(a), '-t', str(ter), '-T', str(ster), '-n', str(num_samples), '-o', fname])
    data = np.loadtxt(fname)
    if structured:
        data.dtype = np.dtype([('response', np.float), ('rt', np.float)])

    return data

def gen_ddm_rts(v=.5, sv=0, z=1, sz=0, a=2, ter=0.3, ster=0, size=500, structured=False, subj_idx=None):
    import brownian

    ddm = brownian.DDM()
    ddm.num_samples = size
    ddm.v = v
    ddm.sv = sv
    ddm.z_bias = z
    ddm.sz = sz
    ddm.a = a
    ddm.ter = ter
    ddm.ster = ster

    rts_upper, rts_lower = ddm.rts
    rts_upper = np.array(rts_upper)/ddm.dt
    rts_lower = np.array(rts_lower)/ddm.dt
    num_upper_resps = rts_upper.shape[0]
    num_lower_resps = rts_lower.shape[0]

    if structured:
        if subj_idx is None:
            data = np.empty((num_upper_resps + num_lower_resps), dtype = ([('response', np.float), ('rt', np.float)]))
        else:
            data = np.empty((num_upper_resps + num_lower_resps), dtype = ([('response', np.float), ('rt', np.float), ('subj_idx', np.float)]))
            data['subj_idx'] = subj_idx
        data['response'][:num_upper_resps] = 1.
        data['response'][num_upper_resps:] = 0.
        data['rt'][:num_upper_resps] = rts_upper
        data['rt'][num_upper_resps:] = rts_lower
    else:
        data = np.empty((num_upper_resps + num_lower_resps))
        data[:num_upper_resps] = rts_upper
        data[num_upper_resps:] = -rts_lower
        
    return data

def _gen_rts_params(params, num_samples=500, fname=None, structured=True, subj_idx=None):
    """Generate simulated RTs with fixed parameters."""
    return gen_ddm_rts(v=params['v'],
                       sv=params['sv'],
                       z=params['z'],
                       sz=params['sz'],
                       a=params['a'],
                       ter=params['ter'],
                       ster=params['ster'],
                       size=num_samples,
                       structured=structured,
                       subj_idx=subj_idx)

def gen_rand_data(num_samples=500, params=None, gen_data=True, no_var=False, tag=None):
    """Generate simulated RTs with random parameters."""
    #z = np.random.normal(loc=1, scale=2)
    #ster = np.random.uniform(loc=0, scale=.5)
    #params_true = {'v': np.random.normal(loc=-2, scale=4), 'sv': np.random.normal(loc=0, scale=.5), 'z': z, 'sz': np.random.normal(loc=0, scale=.5), 'ter': np.random.normal(loc=ster/2., scale=ster/2.), 'ster': ster, 'a': z+np.random.normal(loc=.5, scale=3)}
    if params is None:
        if not no_var:
            params = {'v': .5, 'sv': 0.3, 'z': 1., 'sz': 0.25, 'ter': .3, 'ster': 0.1, 'a': 2}
        else:
            params = {'v': .5, 'sv': 0., 'z': 1., 'sz': 0., 'ter': .3, 'ster': 0., 'a': 2}

    if gen_data:
        # Create RT data
        data = _gen_rts_params(params, num_samples=num_samples, fname='test_data.txt', structured=True)

    if tag is None:
        tag = ''
    if gen_data:
        np.save('data_%s'%tag, data)
    else:
        data = np.load('data_%s.npy'%tag)  

    return (data, params)

def gen_rand_correlation_data(v=.5, corr=.1):
    params = {'v': v,
              'sv': .001,
              'ter': .3,
              'ster': 0.,
              'sz':0}

    all_data = []
    a_offset = 2
    for i in np.linspace(-1,1,10):
        params['a'] = a_offset + i*corr
        params['z'] = (a_offset + i*corr)/2.
        data = gen_rand_subj_data(num_subjs=1, params=params, num_samples=20, add_noise=False)[0]
        theta = np.ones(data.shape) * i
        theta.dtype = dtype=np.dtype([('theta', np.float)])
        stim = np.tile('test', data.shape)
        stim.dtype = np.dtype([('stim', 'S4')])
        
        data = rec.append_fields(data, names=['theta', 'stim'],
                                 data=[theta, stim],
                                 usemask=False)
        all_data.append(data)

    return np.concatenate(all_data)
    
def gen_rand_subj_data(num_subjs=10, params=None, num_samples=100, gen_data=True, add_noise=True, tag=None):
    """Generate simulated RTs of multiple subjects with fixed parameters."""
    # Set global parameters
    #z = rnd(loc=1, scale=2)
    #ster = rnd(loc=0, scale=.5)
    #self.params_true = {'v': rnd(loc=-2, scale=4), 'sv': rnd(loc=0, scale=.5), 'z': z, 'sz': rnd(loc=0, scale=.5), 'ter': rnd(loc=ster/2., scale=ster/2.), 'ster': ster, 'a': z+rnd(loc=.5, scale=3)}
    if params is None:
        params = {'v': .5, 'sv': 0.1, 'z': 1., 'sz': 0.1, 'ter': 1., 'ster': 0.1, 'a': 2}

    params_subjs = []
    #data = np.empty((num_samples*num_subjs, 3), dtype=np.float)
    resps = []
    rts = []
    subj_idx = []
    data_gens = []
    # Derive individual parameters
    for i in range(num_subjs):
        params_subj = copy(params)
        # Add noise to all values
        if add_noise:
            for param, value in params_subj.iteritems():
                if param == 'ter' or param == 'ster' or param == 'z':
                    continue
                elif param[0] == 's':
                    params_subj[param] = np.abs(value + np.random.randn()*.01)
                else:
                    params_subj[param] = np.abs(value + np.random.randn()*.05)
        params_subj['z'] = params_subj['a']/2.
        params_subjs.append(params_subj)

        if gen_data:
            # Create RT data
            data_gen = _gen_rts_params(params_subj, num_samples=num_samples, fname='test_data.txt', structured=True, subj_idx=i)
            data_gens.append(data_gen)
    
    if tag is None:
        tag = ''
    if gen_data:
        data = np.concatenate(data_gens)
        np.save('data_%s'%tag, data)
    else:
        data = np.load('data_%s.npy'%tag)

    return (data, params)

def rec2csv(data, fname, sep=None):
    """Save record array to fname as csv.
    """
    if sep is None:
        sep = ','
    with open(fname, 'w') as fd:
        # Write header
        fd.write(sep.join(data.dtype.names))
        fd.write('\n')
        # Write data
        for line in data:
            line_str = [str(i) for i in line]
            fd.write(sep.join(line_str))
            fd.write('\n')
