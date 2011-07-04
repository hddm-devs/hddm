from __future__ import division
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from copy import copy
import matplotlib.pyplot as plt
import numpy.lib.recfunctions as rec
import os.path
import os
from ordereddict import OrderedDict

import hddm
import kabuki

from mpi4py import MPI

def create_tag_names(tag, chains=None):
    import multiprocessing
    if chains is None:
        chains = multiprocessing.cpu_count()
    tag_names = []
    # Create copies of data and the corresponding db names
    for chain in range(chains):
        tag_names.append("db/mcmc%s%i.pickle"% (tag,chain))

    return tag_names

def controller(models, samples=200, burn=15, reps=5):
    process_list = range(1, MPI.COMM_WORLD.Get_size())
    rank = MPI.COMM_WORLD.Get_rank()
    proc_name = MPI.Get_processor_name()
    status = MPI.Status()

    print "Controller %i on %s: ready!" % (rank, proc_name)

    task_iter = iter(models)
    idx = 0
    results = []

    while(True):
        status = MPI.Status()
        recv = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        #print "Controller: received tag %i from %s" % (status.tag, status.source)
        if status.tag == 15:
            results.append(recv)

        if status.tag == 10 or status.tag == 15:
            try:
                model = task_iter.next()
                name = model['name']
                del model['name']
                #print "Controller: Sending task"
                MPI.COMM_WORLD.send(('%s_%i'%(name, idx), model), dest=status.source, tag=10)
                idx += 1
            except StopIteration:
                # Task queue is empty
                print "Controller: Task queue is empty"
                print "Controller: Sending kill signal"
                MPI.COMM_WORLD.send((None, None), dest=status.source, tag=2)

        elif status.tag == 2: # Exit
            process_list.remove(status.source)
            print 'Process %i exited' % status.source
            print 'Processes left: ' + str(process_list)
        else:
            print 'Unkown tag %i with msg %s' % (status.tag, str(data))
            
        if len(process_list) == 0:
            print "No processes left"
            break

    return results

def worker():
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    proc_name = MPI.Get_processor_name()
    status = MPI.Status()

    print "Worker %i on %s: ready!" % (rank, proc_name)
    # Send ready
    MPI.COMM_WORLD.send([{'rank':rank, 'name':proc_name}], dest=0, tag=10)

    # Start main data loop
    while True:
        # Get some data
        #print "Worker %i on %s: waiting for data" % (rank, proc_name)
        name, model = MPI.COMM_WORLD.recv(source=0, tag=MPI.ANY_TAG, status=status)
        print "Worker %i on %s: received data, tag: %i" % (rank, proc_name, status.tag)

        if status.tag == 2:
            print "Worker %i on %s: received kill signal" % (rank, proc_name)
            MPI.COMM_WORLD.send([], dest=0, tag=2)
            return

        if status.tag == 10:
            # Run emergent
            #print "Worker %i on %s: Running %s" % (rank, proc_name, recv)
            retry = 0
            while retry < 5:
                try:
                    print "Running %s: %s\n" % (name, model)
                    result = run_model(name, model)
                    break
                except (pm.ZeroProbability, ValueError):
                    retry +=1
            if retry == 5:
                result = None
                print "Job %s failed" % recv

        #print("Worker %i on %s: finished one job" % (rank, proc_name))
        MPI.COMM_WORLD.send(result, dest=0, tag=15)

    MPI.COMM_WORLD.send([], dest=0, tag=2)
        
def run_model(name, model, load=False):
    import sys
    
    dbname = os.path.join('/','users', 'wiecki', 'scratch', 'theta', name+'.db')
    dbname = name+'.db'

    if model.has_key('effects_on'):
        mc = pm.MCMC(hddm.model.HDDMRegressor(**model).create(), db='pickle', dbname=dbname)
    else:
        mc = pm.MCMC(hddm.model.HDDM(**model).create(), db='pickle', dbname=dbname)
    
    if not load:
        try:
            os.remove(dbname)
        except OSError:
            pass

    mc.sample(30000, burn=10000, thin=2)
    mc.db.close()
    print "*************************************\n%s"%name
    kabuki.group.print_group_stats(mc.stats())
    print "DIC: %f" % mc.dic
    print "logp: %f" % mc.logp
    sys.stdout.flush()
    return mc.stats()
    # else:
    #     #print "Loading %s" %name
    #     db = pm.database.hdf5.load(dbname)
    #     return db


def load_parallel_chains(model_class, data, tag, kwargs, chains=None, test_convergance=True, combine=True):
    tag_names = create_tag_names(tag, chains=chains)
    models = []
    for tag_name in tag_names:
        model = model_class(data, **kwargs)
        model.mcmc_load_from_db(tag_name)
        models.append(model)

    if test_convergance:
        Rhat = test_chain_convergance(models)
        print Rhat
    
    if combine:
        m = combine_chains(models, model_class, data, kwargs)
        return m
    
    return models

def combine_chains(models, model_class, data, kwargs):
    """Combine multiple model runs into one final model (make sure that chains converged)."""
    # Create model that will contain the other traces
    m = copy(models[0])

    # Loop through models and combine chains
    for model in models[1:]:
        m._set_traces(m.group_params, mcmc_model=model.mcmc_model, add=True)
        m._set_traces(m.group_params_tau, mcmc_model=model.mcmc_model, add=True)
        m._set_traces(m.subj_params, mcmc_model=model.mcmc_model, add=True)

    return m
        
def run_parallel_chains(model_class, data, tag, load=False, cpus=None, chains=None, **kwargs):
    import multiprocessing
    if cpus is None:
        cpus = multiprocessing.cpu_count()

    tag_names = create_tag_names(tag, chains=chains)
    # Parallel call
    if not load:
        rnds = np.random.rand(len(tag_names))*10000
        pool = multiprocessing.Pool(processes=cpus)
        pool.map(call_mcmc, [(model_class, data, tag_name, rnd, kwargs) for tag_name,rnd in zip(tag_names, rnds)])

    models = load_parallel_chains(model_class, data, tag_names, kwargs)

    return models
