#!/usr/bin/python
from __future__ import division
import numpy as np
import numpy.lib.recfunctions as rec
import pymc as pm
import matplotlib.pyplot as plt
from copy import copy

import hddm

def scale(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))


# Model classes
class Base(object):
    """Base class for the hierarchical bayesian drift diffusion
    model."""
    def __init__(self, data, load=None, no_bias=False, trace_subjs=True, save_stats_to=None, debug=False):
        # Flip sign for lower boundary RTs
        self.data = hddm.utils.flip_errors(data)

        self.model = None
        self.mcmc_model = None
        self.map_model = None
        self.params_est = {}
        self.params_est_std = {}
        self.stats = {}
        self.no_bias = no_bias
	self.colors = ('r','b','g','y','c')
        self.trace_subjs = trace_subjs
        self.save_stats_to = save_stats_to
        self.debug = debug
            
        #if load:
        #    self.mcmc_load_from_db(dbname=load)


    def _set_group_params(self):
        raise NotImplementedError("This method has to be overloaded")
    
    def _set_model(self):
        raise NotImplementedError("This method has to be overloaded")

    def _set_all_params(self):
        self._set_group_params()

    def plot_demo(self):
        from hddm import demo
        ddmplot = demo.DDMPlot()
        ddmplot.data = self.data['rt'].flatten()
        ddmplot.external_params = self.params_est
        ddmplot.configure_traits()
        
    def plot_global(self, params_true=None, params=None):
        """Plot real and estimated RT model"""
        # Separate upper and lower boundary responses
        resps_upper = self.data['rt'][self.data['response']==1]
        resps_lower = np.abs(self.data['rt'][self.data['response']==0])

        self._plot(resps_upper, resps_lower, params_true=params_true, params=params)

    def _get_analytical(self, x, params):
        pdf_upper = hddm.wfpt.pdf_array(x=x, v=params['v'], a=params['a'], z=params['z'], ter=params['t'], err=.0001)
        pdf_lower = hddm.wfpt.pdf_array(x=-x, v=params['v'], a=params['a'], z=params['z'], ter=params['t'], err=.0001)
        pdf = np.concatenate((pdf_lower[::-1], pdf_upper)) # Reverse pdf_lower and concatenate

        return pdf

    def _plot(self, resps_upper, resps_lower, bins=40, hrange=(0,4), params=None, params_true=None, reps=100, title=None, label=None, c1=None, c2=None, plot_estimated=True, interpolate=False, k=2):
        """Plot real and estimated RT model. A set of parameters (params) may be provided."""
        import scipy.interpolate
        from scipy.interpolate import InterpolatedUnivariateSpline
        
        if params is None:
            params = self.params_est
        if not c1:
            c1 = 'r'
        if not c2:
            c2 = 'b'
        if not label:
            label = ''

        x = np.linspace(hrange[0], hrange[1], bins)
        x_br = np.linspace(-hrange[1], hrange[1], bins*2)
        xs = np.linspace(hrange[0], hrange[1], bins*10)
        xs_br = np.linspace(-hrange[1], hrange[1], bins*20)

        if interpolate:
            histo_upper = InterpolatedUnivariateSpline(x, np.histogram(resps_upper, bins=bins, range=hrange)[0], k=k)(xs)
            histo_lower = InterpolatedUnivariateSpline(x, np.histogram(resps_lower, bins=bins, range=hrange)[0], k=k)(xs)
        else:
            histo_upper = np.histogram(resps_upper, bins=bins, range=hrange)[0]
            histo_lower = np.histogram(resps_lower, bins=bins, range=hrange)[0]

        histo = np.concatenate((histo_lower[::-1], histo_upper))
        if interpolate:
            plt.plot(xs_br, scale(histo), label="empirical %s"%label, color=c1)
        else:
            plt.plot(x_br, scale(histo), label="empirical %s"%label, color=c1)

        if plot_estimated:
            # Calculate wiener PDFs for both boundaries.
            if self.model_type == 'simple':
                pdf = self._get_analytical(params, xs)
            else:
                pdf = hddm.likelihoods.get_avg_likelihood(xs, params)

            plt.plot(xs_br, scale(pdf), label="analytical %s"%label, color=c2)

        if params_true: # Calculate likelihood for known params
            if self.model_type == 'simple':
                pdf = self._get_analytical(params_true, xs)
            else:
                pdf_true = hddm.likelihoods.get_avg_likelihood(xs, params_true)

            plt.plot(xs_br, scale(pdf_true), label="true %s"%label, color='g')
            
        [ytick.set_visible(False) for ytick in plt.yticks()[1]] # Turn y ticks off
        #plt.legend(loc=0)
        if title:
            plt.title(title)
    
    def map(self):
        """Compute Maximum A Posterior estimates."""
        # Prepare and fit MAP
        self._prepare(map_=True)

        # Write estimates to params_est.
        for param_name in self.param_names:
            self.params_est[param_name] = self.group_params[param_name].value

        return self

    def _prepare(self, dbname=None, map_=True, load=False, verbose=0):
        """Compute posterior model by markov chain monte carlo estimation."""

        ############################################################
        # Try multiple times to create model. Sometimes bad initial
        # parameters are chosen randomly that yield -Inf
        # log-likelihood which causes PyMC to choke.
        model_yields_zero_prob = True
        tries = 0
        if not self.debug:
            while (model_yields_zero_prob):
                try:
                    self._set_all_params()
                    dists = self._set_model()
                    model_yields_zero_prob = False
                except pm.ZeroProbability:
                    tries += 1
                    if tries > 20:
                        raise pm.ZeroProbability("Model creation failed")
        else:
            self._set_all_params()
            self._set_model()

        # Set model parameter values to MAP estimates
        if map_ and not load:
            self.map_model = pm.MAP(self.model)
            self.map_model.fit()

        # Save future samples to database if needed.
        if not load:
            if dbname is None:
                self.mcmc_model = pm.MCMC(self.model, verbose=verbose)
            else:
                self.mcmc_model = pm.MCMC(self.model, db='sqlite', dbname=dbname, verbose=verbose)
        else:
            # Open database
            #db = pm.database.pickle.load(dbname)
            db = pm.database.sqlite.load(dbname)
        
            # Create mcmc instance reading from the opened database
            self.mcmc_model = pm.MCMC(self.model, db=db, verbose=verbose)

            # Take the traces from the database and feed them into our
            # distribution variables (needed for _gen_stats())
            self._set_traces(self.group_params)
            
            if self.is_subj_model:
                self._set_traces(self.group_params_tau)
                self._set_traces(self.subj_params)

            self._gen_stats()
            
        return self
    
    def _sample(self, samples=10000, burn=5000, thin=2, verbose=0, dbname=None):
        """Draw posterior samples. Requires self.model to be set.
        """
        try:
            self.mcmc_model.sample(samples, burn=burn, thin=thin, verbose=verbose)
        except NameError:
            raise NameError("mcmc_model not set, call ._prepare()")

        self._gen_stats()

        if dbname is not None:
            self.mcmc_model.db.commit()

        return self

    def norm_approx(self):
        # Set model parameter values to Normal Approximations
        self._prepare()
        
        self.norm_approx_model = pm.NormApprox(self.model)
        self.norm_approx_model.fit()

        return self
            

    def mcmc(self, samples=10000, burn=5000, thin=2, verbose=0, dbname=None, map_=True):
        """Main method for sampling. Creates and initializes the model and starts sampling.
        """
        # Set and initialize model
        self._prepare(dbname=dbname, map_=map_)
        # Draw samples
        self._sample(samples=samples, burn=burn, thin=thin, verbose=verbose, dbname=dbname)

        return self
    
    def _set_traces(self, params, mcmc_model=None, add=False):
        """Externally set the traces of group_params. This is needed
        when loading a model from a previous sampling run saved to a
        database.
        """
        if not mcmc_model:
            mcmc_model = self.mcmc_model
        for param_name, param_inst in params.iteritems():
            if param_name == 'z' and self.no_bias:
                continue
            try:
                if add:
                    param_inst.trace._trace[0] = np.concatenate((param_inst.trace._trace[0],
                                                                 mcmc_model.trace(param_name)()))
                else:
                    param_inst.trace = mcmc_model.trace(param_name)
            except AttributeError: # param_inst is an array
                if self.trace_subjs:
                    for i, subj_param in enumerate(param_inst):
                        if add:
                            subj_param.trace._trace[0] = np.concatenate((subj_param.trace._trace[0],
                                                                         mcmc_model.trace('%s_%i'%(param_name,i))()))
                        else:
                            subj_param.trace = mcmc_model.trace('%s_%i'%(param_name,i))

    def mcmc_load_from_db(self, dbname):
        """Load samples from a database created by an earlier model
        run (e.g. by calling .mcmc(dbname='test'))
        """
        # Set up model
        self._prepare(dbname=dbname, load=True)


        return self


    
    def _gen_stats(self):
        for param_name in self.param_names:
            if param_name == 'z' and self.no_bias:
                continue
            self.params_est[param_name] = np.mean(self.mcmc_model.trace(param_name)())
            self.params_est_std[param_name] = np.std(self.mcmc_model.trace(param_name)())

        # Save stats to output file
        if self.save_stats_to is not None:
            print "Saving stats to %s" % self.save_stats_to
            with open(self.save_stats_to, 'w') as fd:
                for name, value in self.params_est.iteritems():
                    fd.write('%s: %f\n'%(name, value))
                    
        return self

class Multi(Base):
    """Hierarchical Drift-Diffusion Model.

    This class can generate different hddms:
    - simple DDM (without inter-trial variabilities)
    - full averaging DDM (with inter-trial variabilities)
    - subject param DDM (each subject get's it's own param, see EJ's book 8.3)
    - parameter dependent on data (e.g. drift rate is dependent on stimulus
    """
    def __init__(self, data,
                 depends_on=None, stats_on=None, model_type=None, is_subj_model=True,
                 trace_subjs=True, load=None, save_stats_to=None,
                 debug=False, no_bias=True, normalize_v=True, init_EZ=True, pool_depends=True):

        super(Multi, self).__init__(data, save_stats_to=save_stats_to, trace_subjs=trace_subjs,
                                    debug=debug, no_bias=no_bias)

        if model_type is None:
            self.model_type = 'simple'
        else:
            self.model_type = model_type
        
        self.param_factory = ParamFactory(self.model_type,
                                          data=self.data,
                                          trace_subjs=trace_subjs,
                                          normalize_v=normalize_v,
                                          no_bias=no_bias,
                                          init=init_EZ)
            
        
        # Initialize
        if depends_on is None:
            self.depends_on = {}
        else:
            self.depends_on = copy(depends_on)

        self.pool_depends = pool_depends

        self.is_subj_model = is_subj_model

        if self.is_subj_model:
            self.subjs = np.unique(self.data['subj_idx'])
            self.num_subjs = self.subjs.shape[0]

        # Define parameters for the simple and full averaged ddm.
        if self.model_type == 'simple' or self.model_type == 'simple_gpu':
            self.group_param_names = ['a', 'v', 'z', 't']
        elif self.model_type == 'full_avg' or self.model_type == 'full':
            self.group_param_names = ['a', 'v', 'V', 'z', 'Z', 't', 'T']
        elif self.model_type == 'lba':
            self.group_param_names = ['a', 'z', 't', 'V', 'v0', 'v1']
        else:
            raise ValueError('Model %s not recognized' % self.model_type)

        self.param_names = copy(self.group_param_names)
        self.group_params = {}
        self.root_params = {}
        self.group_params_tau = {}
        self.root_params_tau = {}
        
        if load:
            self.mcmc_load_from_db(dbname=load)

    def _set_all_params(self):
        self._set_group_params()
        if self.is_subj_model:
            self._set_subj_params()

        return self

    def _set_dependent_group_param(self, param):
        """Set group parameters that only depend on individual classes of data."""
        depends_on = self.depends_on[param]
        uniq_data_dep = np.unique(self.data[depends_on])

        if self.pool_depends:
            # Create a global parameter that is parent to all dependent group parameters.
            self.root_params[param] = self.param_factory.get_root_param(param)
            self.root_params_tau[param] = self.param_factory.get_tau_param(param)

        for uniq_date in uniq_data_dep:
            tag = str(uniq_date)
            param_tag = '%s_%s'%(param, tag)
            if self.pool_depends:
                self.group_params[param_tag] = self.param_factory.get_child_param(param,
                                                                                  parent_mean=self.root_params[param],
                                                                                  parent_tau=self.root_params_tau[param],
                                                                                  tag=tag,
                                                                                  plot=True)
            else:
                self.group_params[param_tag] = self.param_factory.get_root_param(param, tag=tag)

        return self

    def _set_group_params(self):
        """Set group level distributions. One distribution for each DDM parameter."""
        for param in self.group_param_names: # Loop through param names
            if param in self.depends_on.keys():
                self._set_dependent_group_param(param)
            else:
                # Parameter does not depend on data
                self.group_params[param] = self.param_factory.get_root_param(param)
        
        return self

    def _set_subj_params(self):
        """Set individual subject distributions. Each subject is
        assigned one set of DDM parameter distributions which have the
        group level parameters as their parents"""
        # For each global param, create n subj_params
        self.subj_params = {}

        # Initialize
        for param_name, param_inst in self.group_params.iteritems():
            self.subj_params[param_name] = np.empty(self.num_subjs, dtype=object)
            
        for param_name, param_inst in self.group_params.iteritems():
            # Create tau parameter for global param
            param_inst_tau = self.param_factory.get_tau_param(param_name)
            self.group_params_tau[param_name] = param_inst_tau
            # Create num_subjs individual subject ddm parameter
            for subj_idx,subj in enumerate(self.subjs):
                self.subj_params[param_name][subj_idx] = self.param_factory.get_subj_param(param_name,
                                                                                           param_inst,
                                                                                           param_inst_tau,
                                                                                           int(subj))
        return self
    
    def _set_model(self):
        """Create and set up the complete DDM."""
        if self.is_subj_model:
            params = copy(self.subj_params) # use subj parameters to feed into ddm 
        else:
            params = copy(self.group_params) # use group parameters to feed into ddm

        self.idx = 0 # DDM counter

        # call recursive function to create dependency model
        depends_on = copy(self.depends_on)
        ddm = self._set_model_rec(data=self.data, depends_on=depends_on, params=params)

        # Set and return all distributions belonging to the DDM.
        if self.is_subj_model:
            self.model = ddm + self.group_params.values() + self.group_params_tau.values() + self.subj_params.values() + \
                         self.root_params.values() + self.root_params_tau.values()
        else:
            self.model = ddm + self.group_params.values() + self.root_params.values() + self.root_params_tau.values()

        return self
        
    def _set_model_rec(self, data, depends_on, params):
        """Recursive function called by _set_model() to generate the DDM."""
        if depends_on: # If depends are present
            ddm_list = []
            param_name = depends_on.keys()[0] # Get first param from depends_on
            depends_on_col = depends_on.pop(param_name) # Take out param
            depend_elements = np.unique(data[depends_on_col])
            for depend_element in depend_elements:
                data_dep = data[data[depends_on_col] == depend_element]
                # Set the appropriate param
                if self.is_subj_model:
                    params[param_name] = self.subj_params[param_name+'_'+str(depend_element)]
                else:
                    params[param_name] = self.group_params[param_name+'_'+str(depend_element)]
                # Recursive call with one less dependency and the sliced data.
                ddm = self._set_model_rec(data_dep, depends_on=depends_on, params=params)
                ddm_list += ddm
            return ddm_list
                
        else: # Data does not depend on anything (anymore)
            ddm = self._create_ddm(data, params)
            return [ddm]

    def _plot(self):
        raise NotImplementedError("TODO")

    def _create_ddm(self, data, params):
        """Create and return a DDM on [data] with [params].
        """
        if self.is_subj_model:
            ddm = np.empty(self.num_subjs, dtype=object)
            for i,subj in enumerate(self.subjs):
                data_subj = data[data['subj_idx'] == subj] # Select data belong to subj

                ddm = self.param_factory.get_model("ddm_%i_%i"%(self.idx, i), data_subj, params, idx=i)
        else: # Do not use subj params, but group ones
            ddm = self.param_factory.get_model("ddm_%i"%self.idx, data, params)

        self.idx+=1
        
        return ddm

    def summary(self, delimiter=None):
        """Return summary statistics of the fit model."""
        if delimiter is None:
            delimiter = '\n'

        s = 'Model type: %s%s'%(self.model_type, delimiter)
        for param, depends_on in self.depends_on.iteritems():
            s+= 'DDM parameter "%s" depends on: %s%s' %(param, ','.join(depends_on), delimiter)

        s += delimiter + 'General model stats:' + delimiter
        for name, value in self.stats.iteritems():
            s += '%s: %f%s'%(name, value, delimiter) 

        s += delimiter + 'Group parameter\t\t\tMean\t\tStd' + delimiter
        for name, value in self.params_est.iteritems():
            s += '%s\t\t\t\t%f\t%f%s'%(name, value, self.params_est_std[name], delimiter)

        return s

    def summary_subjs(self, delimiter=None):
        if delimiter is None:
            delimiter = '\n'

        s = 'Group parameter\t\t\tMean\t\tStd' + delimiter
        for subj, params in self.params_est_subj.iteritems():
            s += 'Subject: %i%s' % (subj, delimiter)
            for name,value in params.iteritems():
                s += '%s\t\t\t\t%f\t%f%s'%(name, value, self.params_est_subj_std[subj][name], delimiter)
            s += delimiter
            
        return s
    
    def _gen_stats(self):
        """Generate summary statistics of fit model."""
        self.stats['logp'] = self.mcmc_model.logp
        self.stats['dic'] = self.mcmc_model.dic

        self.params_est_subj = {}
        self.params_est_subj_std = {}
        
        for param_name in self.group_params.iterkeys():
            if param_name == 'z' and self.no_bias:
                continue
            self.params_est[param_name] = np.mean(self.mcmc_model.trace(param_name)())
            self.params_est_std[param_name] = np.std(self.mcmc_model.trace(param_name)())

        for param_name in self.root_params.iterkeys():
            if param_name == 'z' and self.no_bias:
                continue
            self.params_est[param_name] = np.mean(self.mcmc_model.trace(param_name)())
            self.params_est_std[param_name] = np.std(self.mcmc_model.trace(param_name)())

        if self.trace_subjs and self.is_subj_model:
            for name,params in self.subj_params.iteritems():
                for subj_idx,subj_dist in enumerate(params):
                    if not self.params_est_subj.has_key(subj_idx):
                        # Init
                        self.params_est_subj[subj_idx] = {}
                        self.params_est_subj_std[subj_idx] = {}

                    if subj_dist is None:
                        continue # z is none in non-bias case

                    self.params_est_subj[subj_idx][name] = np.mean(subj_dist.trace())
                    self.params_est_subj_std[subj_idx][name] = np.std(subj_dist.trace())
            
        

    def save_stats(self, fname):
        """Save stats to output file."""
        print "Saving stats to %s" % fname
        s = self.summary()
        with open(fname, 'w') as fd:
            fd.write(s)
                
        return self

    def _get_analytical_lba(self, params, x):
        pdf_upper = hddm.likelihoods.LBA_like(value=x, v0=params['v0'], v1=params['v1'], a=params['a'], z=params['z'], ter=params['t'], err=.0001)
        pdf_lower = hddm.likelihoods.LBA_like(value=-x, v0=params['v0'], v1=params['v1'], a=params['a'], z=params['z'], ter=params['t'], err=.0001)
        pdf = np.concatenate((pdf_lower[::-1], pdf_upper)) # Reverse lower pdf and concatenate

        return pdf

class ParamFactory(object):
    def __init__(self, model_type, data=None, trace_subjs=True, normalize_v=True, no_bias=True, fix_sv=None, init=True):
        self.trace_subjs = trace_subjs
        self.model_type = model_type
        self.no_bias = no_bias
        self.fix_sv = fix_sv
        self.data = data
        
        # Set function map
        self._models = {'simple': self._get_simple,
                        'simple_gpu': self._get_simple_gpu,
                        'full_avg': self._get_full_avg,
                        'full': self._get_full,
                        'lba':self._get_lba}

        if self.model_type != 'lba':
            self.param_ranges = {'a_lower': .5,
                                 'a_upper': 4.5,
                                 'z_lower': .1,
                                 'z_upper': 2.5,
                                 't_lower': .1,
                                 't_upper': 1.,
                                 'v_lower': -3.,
                                 'v_upper': 3.,
                                 'T_lower': 0.,
                                 'T_upper': 1.,
                                 'Z_lower': 0.,
                                 'Z_upper': 1.,
                                 'e_lower': -.5,
                                 'e_upper': .5
                                 }
            if not init:
                # Default param ranges
                self.init_params = {}
            else:
                # Compute ranges based on EZ method
                param_ranges = hddm.utils.EZ_param_ranges(self.data)
                # Overwrite set parameters
                for param,value in param_ranges.iteritems():
                    self.param_ranges[param] = value
                self.init_params = hddm.utils.EZ_subjs(self.data)
                
            self.normalize_v = False
            self.fix_sv = None
            
        else:
            # LBA model
            self.normalize_v = normalize_v

            self.param_ranges = {'a_lower': .2,
                                 'a_upper': 4.,
                                 'v_lower': 0.1,
                                 'v_upper': 3.,
                                 'z_lower': .0,
                                 'z_upper': 2.,
                                 't_lower': .05,
                                 't_upper': 2.,
                                 'V_lower': .2,
                                 'V_upper': 2.}
            
            if self.normalize_v:
                self.param_ranges['v_lower'] = 0.
                self.param_ranges['v_upper'] = 1.

        
    def get_model(self, *args, **kwargs):
        return self._models[self.model_type](*args, **kwargs)
    
    def get_root_param(self, param, tag=None):
        """Create and return a prior distribution for [param]. [tag] is
        used in case of dependent parameters.
        """
        if tag is None:
            tag = ''
        else:
            tag = '_' + tag

        if param in self.init_params:
            init_val = self.init_params[param]
        else:
            init_val = None
            
        if param == 'V' and self.fix_sv is not None: # drift rate variability
            return pm.Lambda("V%s"%tag, lambda x=self.fix_sv: x)

        elif param == 'z' and self.no_bias: # starting point position (bias)
            return None # z = a/2.

        else:
            return pm.Uniform("%s%s"%(param, tag),
                              lower=self.param_ranges['%s_lower'%param],
                              upper=self.param_ranges['%s_upper'%param],
                              value=init_val)

    def get_tau_param(self, param, tag=None):
        if tag is None:
            tag = '_tau'
        else:
            tag = tag + '_tau'

        return pm.Uniform(param + tag, lower=0, upper=800, plot=False)

    def get_subj_param(self, param_name, parent_mean, parent_tau, subj_idx):
        if len(param_name) != 1: # if there is a tag attached to the param
            param = param_name[0]
            tag = param_name[1:] + '_' + str(subj_idx) # create new name for the subj parameter
        else:
            param = param_name
            tag = '_' + str(subj_idx)

        init_param_name = '%s_%i'%(param_name,subj_idx)
        if init_param_name in self.init_params:
            init_val = self.init_params[init_param_name]
        else:
            init_val = None

        return self.get_child_param(param, parent_mean, parent_tau, tag=tag, init_val=init_val)

    def get_child_param(self, param, parent_mean, parent_tau, tag=None, init_val=None, plot=False):
        if tag is None:
            tag = ''

        if not tag.startswith('_'):
            tag = '_'+tag
    
        if param == 'V' and self.fix_sv is not None:
            return pm.Lambda("V%s"%tag, lambda x=parent_mean: parent_mean,
                             plot=plot, trace=self.trace_subjs)

        elif param == 'z' and self.no_bias:
            return None

        elif param == 'e' or param.startswith('v'):
            return pm.Normal("%s%s"%(param,tag),
                             mu=parent_mean,
                             tau=parent_tau,
                             plot=plot, trace=self.trace_subjs,
                             value=init_val)

        else:
            return pm.TruncatedNormal("%s%s"%(param,tag),
                                      a=self.param_ranges['%s_lower'%param],
                                      b=self.param_ranges['%s_upper'%param],
                                      mu=parent_mean, tau=parent_tau,
                                      plot=plot, trace=self.trace_subjs,
                                      value=init_val)



    def _get_simple(self, name, data, params, idx=None):
        if idx is None:
            return hddm.likelihoods.WienerSimple(name,
                                                value=data['rt'].flatten(), 
                                                v=params['v'], 
                                                ter=params['t'], 
                                                a=params['a'], 
                                                z=params['z'],
                                                observed=True)
        else:
            return hddm.likelihoods.WienerSimple(name,
                                value=data['rt'].flatten(), 
                                v=params['v'][idx], 
                                ter=params['t'][idx], 
                                a=params['a'][idx], 
                                z=params['z'][idx],
                                observed=True)


    def _get_simple_gpu(self, name, data, params, idx=None):
        if idx is None:
            return hddm.likelihoods.WienerGPUSingle(name,
                                   value=data['rt'].flatten(), 
                                   v=params['v'], 
                                   ter=params['t'], 
                                   a=params['a'], 
                                   z=params['z'],
                                   observed=True)
        else:
            return hddm.likelihoods.WienerGPUSingle(name,
                                   value=data['rt'].flatten(), 
                                   v=params['v'][idx], 
                                   ter=params['t'][idx], 
                                   a=params['a'][idx],
                                   z=params['z'][idx],
                                   observed=True)

    def _get_full_avg(self, name, data, params, idx=None):
        if idx is None:
            return hddm.likelihoods.WienerAvg(name,
                             value=data['rt'].flatten(), 
                             v=params['v'], 
                             sv=params['V'],
                             ter=params['t'],
                             ster=params['T'], 
                             a=params['a'],
                             z=params['z'],
                             sz=params['Z'],
                             observed=True)

        else:
            return hddm.likelihoods.WienerAvg(name,
                             value=data['rt'].flatten(), 
                             v=params['v'][idx], 
                             sv=params['V'][idx],
                             ter=params['t'][idx],
                             ster=params['T'][idx], 
                             a=params['a'][idx],
                             z=params['z'][idx],
                             sz=params['Z'][idx],
                             observed=True)


    def _get_full(self, name, data, params, idx=None):
        if idx is None:
            trials = data.shape[0]
            ddm[i] = np.empty(trials, dtype=object)
            z_trial = np.empty(trials, dtype=object)
            v_trial = np.empty(trials, dtype=object)
            ter_trial = np.empty(trials, dtype=object)
            for trl in range(trials):
                z_trial[trl] = hddm.likelihoods.CenterUniform("z_%i"%trl,
                                             center=params['z'],
                                             width=params['sz'],
                                             plot=False, observed=False, trace=False)
                v_trial[trl] = pm.Normal("v_%i"%trl,
                                         mu=params['v'],
                                         tau=1/(params['sv']**2),
                                         plot=False, observed=False, trace=False)
                ter_trial[trl] = hddm.likelihoods.CenterUniform("ter_%i"%trl,
                                               center=params['ter'],
                                               width=params['ster'],
                                               plot=False, observed=False, trace=False)
                ddm[i][trl] = hddm.likelihoods.Wiener2("ddm_%i_%i"%(trl, i),
                                      value=data['rt'].flatten()[trl],
                                      v=v_trial[trl],
                                      ter=ter_trial[trl], 
                                      a=param['a'],
                                      z=z_trial[trl],
                                      observed=True, trace=False)

            return ddm

        else:
            trials = data.shape[0]
            ddm = np.empty(trials, dtype=object)
            z_trial = np.empty(trials, dtype=object)
            v_trial = np.empty(trials, dtype=object)
            ter_trial = np.empty(trials, dtype=object)
            for trl in range(trials):
                z_trial[trl] = hddm.likelihoods.CenterUniform("z_%i"%trl,
                                             center=params['z'],
                                             width=params['sz'],
                                             plot=False, observed=False)
                v_trial[trl] = pm.Normal("v_%i"%trl,
                                         mu=params['v'],
                                         tau=1/(params['sv']**2),
                                         plot=False, observed=False)
                ter_trial[trl] = hddm.likelihoods.CenterUniform("ter_%i"%trl,
                                               center=params['ter'],
                                               width=params['ster'],
                                               plot=False, observed=False)
                ddm[trl] = hddm.likelihoods.Wiener2("ddm_%i"%trl,
                                   value=data['rt'].flatten()[trl],
                                   v=v_trial[trl],
                                   ter=ter_trial[trl], 
                                   a=param['a'],
                                   z=z_trial[trl],
                                   observed=True)

            return ddm

    def _get_lba(self, name, data, params, idx=None):
        if idx is None:
            return hddm.likelihoods.LBA(name,
                                        value=data['rt'].flatten(),
                                        a=params['a'],
                                        z=params['z'],
                                        ter=params['t'],
                                        v0=params['v0'],
                                        v1=params['v1'],
                                        sv=params['V'],
                                        normalize_v=self.normalize_v,
                                        observed=True)
        else:
            return hddm.likelihoods.LBA(name,
                                        value=data['rt'].flatten(),
                                        a=params['a'][idx],
                                        z=params['z'][idx],
                                        ter=params['t'][idx],
                                        v0=params['v0'][idx],
                                        v1=params['v1'][idx],
                                        sv=params['V'][idx],
                                        normalize_v=self.normalize_v,
                                        observed=True)    

class MultiGpu(Multi):
    def _set_model(self):
        """Create and set up the complete DDM."""
        #if self.is_subj_model:
        #    params = copy(self.subj_params) # use subj parameters to feed into ddm 
        #else:
        #    params = copy(self.group_params) # use group parameters to feed into ddm

        #for param in params.itervalues():
        #    param = np.empty(self.data.shape, dtype=object)

        self.idx = 0 # DDM counter
        params_out = {}
        for param in self.group_param_names:
            params_out[param] = np.empty(self.data.shape[0], dtype=object)
            
        # call function to create dependency model
        depends_on = copy(self.depends_on)
        params_out = self._set_model_rec(depends_on=depends_on, params_out=params_out)

        # Set and return all distributions belonging to the DDM.
        ddm = hddm.likelihoods.WienerGPUGlobal('ddm',
                              value=self.data['rt'].flatten(),
                              a=params_out['a'],
                              z=params_out['z'],
                              v=params_out['v'],
                              ter=params_out['t'])

        if self.is_subj_model:
            self.model = [ddm] + self.group_params.values() + self.group_params_tau.values() + self.subj_params.values()
        else:
            self.model = [ddm] + self.group_params.values()

        return self

    def _set_model_rec(self, depends_on, params_out):
        """Recursive function called by _set_model() to generate the DDM."""
        # Set non-dependent parameters
        for param_name in self.group_params:
            if param_name[0] in depends_on.keys():
                continue # Skip dependent parameters
            if self.is_subj_model:
                for i,subj in enumerate(self.subjs):
                    idx = (self.data['subj_idx'] == subj).flatten()
                    num_elements = np.sum(idx)
                    params_out[param_name[0]][idx] = np.repeat(self.subj_params[param_name[0]][i], num_elements).flatten()
            else:
                params_out[param_name[0]] = np.repeat(self.group_params[param_name[0]],
                                                      num_elements).flatten()
        # Set dependent parameters
        for param_name,depends_on_col in depends_on.iteritems(): # If depends are present
            depend_elements = np.unique(self.data[depends_on_col])
            for depend_element in depend_elements:
                # Set the appropriate param
                if self.is_subj_model:
                    for i,subj in enumerate(self.subjs):
                        idx = ((self.data[depends_on_col] == depend_element) & (self.data['subj_idx'] == subj)).flatten()
                        num_elements = np.sum(idx)
                        params_out[param_name[0]][idx] = np.repeat(self.subj_params[param_name[0]+'_'+str(depend_element)][i], num_elements).flatten()
                else:
                    idx = self.data[depends_on_col] == depend_element
                    num_elements = np.sum(idx)
                    params_out[param_name[0]][idx] = np.repeat(self.group_params[param_name[0]+'_'+str(depend_element)],
                                                               num_elements).flatten()
        return params_out

    
class MultiEffect(Multi):
    true_vals = {'theta': 'high', 'dbs':'on'}
    dual_effect = False
    def _set_group_params(self):
        """Set group level distributions. One distribution for each DDM parameter."""
        for group_param_name in self.group_param_names: # Loop through param names
            self.group_params[group_param_name] = self._get_group_param(group_param_name)

            # Generate effect priors
            if group_param_name in self.depends_on.keys(): # Parameter depends on data
                depends_on = self.depends_on[group_param_name]
                for depend in depends_on:
                    self.group_params['e_%s_%s'%(group_param_name, depend)] = self._get_group_param('e', tag=group_param_name+'_'+str(depend))
                if len(depends_on):
                    self.group_params['e_%s_inter'%(group_param_name)] = self._get_group_param('e', tag=group_param_name+'_inter')

        return self

    def _set_subj_params(self):
        """Set individual subject distributions. Each subject is
        assigned one set of DDM parameter distributions which have the
        group level parameters as their parents"""
        # For each global param, create n subj_params
        self.subj_params = {}
        self.group_params_tau = {}
        for param_name, param_inst in self.group_params.iteritems():
            # Create tau parameter for global param
            param_inst_tau = self._get_group_param_tau(param_name)
            self.group_params_tau[param_name] = param_inst_tau

            # Create num_subjs individual subject ddm parameter
            self.subj_params[param_name] = np.empty(self.num_subjs, dtype=object) # init
            for subj_idx in range(self.num_subjs):
                self.subj_params[param_name][subj_idx] = self._get_subj_param(param_name, param_inst, param_inst_tau, subj_idx)

        return self

        
    def _set_model(self, data, depends_on, params):
        """Recursive function called by _set_model() to generate the DDM."""
        if depends_on: # If depends are present
            ddm_list = []
            param_name = depends_on.keys()[0] # Get first param from depends_on
            depends_on_col = depends_on.pop(param_name) # Take out param
            depend_elements = np.unique(data[depends_on_col])
            for depend_element in depend_elements:
                #depend_element = depend_elem[0] # Circumvent convention of structured arrays.
                data_dep = data[data[depends_on_col] == depend_element]
                # Set the appropriate param
                if self.is_subj_model:
                    debug_here()
                    effect_name = 'e_det_%s_%s_%s'%(param_name[0], depends_on_col, depend_element)
                    if not self.subj_params.has_key(effect_name):
                        # Create effect distributions
                        self.subj_params[effect_name] = np.empty(self.num_subjs, dtype=object) # init
                        for i in range(self.num_subjs):
                            if len(depend_element) == 1:
                                self.subj_params[effect_name][i] = self.get_effect1(base=self.subj_params[param_name[0]][i],
                                                                                    e1_val=(depend_element==self.true_vals[depends_on_col]),
                                                                                    e1_dist=self.subj_params['e_%s_%s'%(depends_on_col, param_name[0])][i])
                            elif len(depend_element) == 2:
                                self.subj_params[effect_name][i] = self.get_effect2(base=self.subj_params[param_name[0]][i],
                                                                                    e1_val=(depend_element[0]==self.true_vals[depends_on_col[0]]),
                                                                                    e1_dist=self.subj_params['e_%s_%s'%(param_name[0],depends_on_col[0])][i],
                                                                                    e2_val=(depend_element[1]==self.true_vals[depends_on_col[1]]),
                                                                                    e2_dist=self.subj_params['e_%s_%s'%(param_name[0],depends_on_col[1])][i],
                                                                                    e_inter_dist=self.subj_params['e_%s_inter'%(param_name[0])][i])
                            else:
                                raise NotImplementedError('Only 2 effects per parameter are currently supported')
                            
                    params[param_name[0]] = self.subj_params[effect_name]

                else:
                    raise NotImplementedError('Can only use this functionality with a subject model at this time.')
                    params[param_name[0]] = self.group_params[param_name[0]+'_'+str(depend_element)]
                # Recursive call with one less dependency and the sliced data.
                ddm = self._set_model(data_dep, depends_on=depends_on, params=params)
                ddm_list += ddm
            return ddm_list
                
        else: # Data does not depend on anything (anymore)
            ddm = self.create_ddm(data, params)
            return [ddm]

    def get_effect1(self, base, e1_val, e1_dist):
        if e1_val:
            return pm.Lambda('e_ef_%s_e1' % base.__name__,
                             lambda base=base, e1=e1_dist: base+e1_dist)
        else:
            return pm.Lambda('e_ef_%s' % base.__name__,
                             lambda base=base, e1=e1_dist: base+e1_dist)


    def get_effect2(self, base, e1_val, e2_val, e1_dist, e2_dist, e_inter_dist):
        if e1_val and e2_val:
            return pm.Lambda('e_%s_e1_e2' % base.__name__,
                             lambda base=base, e1=e1_dist, e2=e2_dist, e_inter=e_inter_dist: base+e1_dist+e2_dist+e_inter_dist)

        elif e1_val and not e2_val:
            return pm.Lambda('e_%s_e1' % base.__name__,
                             lambda base=base, e1=e1_dist: base+e1_dist)

        elif not e1_val and e2_val:
            return pm.Lambda('e_%s_e2' % base.__name__,
                             lambda base=base, e2=e2_dist: base+e2_dist)

        elif not e1_val and not e2_val:
            if self.dual_effect:
                return pm.Lambda('e_%s' % base.__name__,
                                 lambda base=base, e_inter=e_inter_dist: base+e_inter_dist)
            else:
                return pm.Lambda('e_%s' % base.__name__,
                                 lambda base=base: base)
        else:
            raise ValueError('Combination %s and %s not valid'%(dbs, theta))
