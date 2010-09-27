#!/usr/bin/python
from __future__ import division
import numpy as np
import numpy.lib.recfunctions as rec
import pymc as pm
import matplotlib.pyplot as plt
import platform
from copy import copy
import matplotlib.pyplot as plt
import subprocess
from ddm_likelihoods import *

try:
    from IPython.Debugger import Tracer; debug_here = Tracer()
except:
    def debug_here(): pass

if platform.architecture()[0] == '64bit':
    import wfpt64 as wfpt
    sampler_exec = 'construct-samples64'
else:
    import wfpt
    sampler_exec = 'construct-samples'

def scale(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))


# Model classes
class HDDM_base(object):
    """Base class for the hierarchical bayesian drift diffusion
    model."""
    def __init__(self, data, load=None, no_bias=False, trace_subjs=True, col_names=None, save_stats_to=None, debug=False):
        self.col_names = {'subj_idx':'subj_idx', 'rt': 'rt', 'response': 'response'}
        # Overwrite user suppplied column names
        if col_names is not None:
            for key, val in col_names.iteritems():
                self.col_names[key] = val

        # Flip sign for lower boundary RTs
        self._prepare_data(data)

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

    def _prepare_data(self, data):
        """Flip sign for lower boundary responses."""
        # Copy data
        self.data = np.array(data)
        # Flip sign for lower boundary responses
        idx = self.data[self.col_names['response']] == 0
        self.data[self.col_names['rt']][idx] = -self.data[self.col_names['rt']][idx]

        return self

    def _set_group_params(self):
        raise NotImplementedError("This method has to be overloaded")
    
    def _set_model(self):
        raise NotImplementedError("This method has to be overloaded")

    def _set_all_params(self):
        self._set_group_params()

    def plot_brownian(self):
        import brownian
        ddmplot = brownian.DDMPlot()
        ddmplot.data = self.data['rt'].flatten()
        ddmplot.external_params = self.params_est
        ddmplot.configure_traits()
        
    def plot_global(self, params_true=None, params=None):
        """Plot real and estimated RT model"""
        # Separate upper and lower boundary responses
        resps_upper = self.data[self.col_names['rt']][self.data[self.col_names['response']]==1]
        resps_lower = np.abs(self.data[self.col_names['rt']][self.data[self.col_names['response']]==0])

        self._plot(resps_upper, resps_lower, params_true=params_true, params=params)

    def _plot(self, resps_upper, resps_lower, bins=40, hrange=(0,4), params=None, params_true=None, reps=100, title=None, label=None, c1=None, c2=None, plot_estimated=True, interpolate=False, k=2):
        """Plot real and estimated RT model. A set of parameters (params) may be provided."""
        import scipy.interpolate
        
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
            histo_upper = scipy.interpolate.InterpolatedUnivariateSpline(x, np.histogram(resps_upper, bins=bins, range=hrange)[0], k=k)(xs)
            histo_lower = scipy.interpolate.InterpolatedUnivariateSpline(x, np.histogram(resps_lower, bins=bins, range=hrange)[0], k=k)(xs)
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
                pdf_upper = wfpt.pdf_array(x=xs, v=params['v'], a=params['a'], z=params['z'], ter=params['t'], err=.0001)
                pdf_lower = wfpt.pdf_array(x=-xs, v=params['v'], a=params['a'], z=params['z'], ter=params['t'], err=.0001)
                pdf = np.concatenate((pdf_lower[::-1], pdf_upper))
            else:
                pdf = get_avg_likelihood(xs, params)

            plt.plot(xs_br, scale(pdf), label="analytical %s"%label, color=c2)

        if params_true: # Calculate likelihood for known params
            if self.model_type == 'simple':
                pdf_upper_true = wfpt.pdf_array(x=xs, v=params_true['v'], a=params_true['a'], z=params_true['z'], ter=params_true['t'], err=.0001)
                pdf_lower_true = wfpt.pdf_array(x=-xs, v=params_true['v'], a=params_true['a'], z=params_true['z'], ter=params_true['t'], err=.0001)
                pdf_true = np.concatenate((pdf_lower_true[::-1], pdf_upper_true))
            else:
                pdf_true = get_avg_likelihood(xs, params_true)

            plt.plot(xs_br, scale(pdf_true), label="true %s"%label, color='g')
            
        [ytick.set_visible(False) for ytick in plt.yticks()[1]] # Turn y ticks off
        #plt.legend(loc=0)
        if title:
            plt.title(title)
    
    def map(self):
        """Compute Maximum A Posterior estimates."""
        # Prepare and fit MAP
        self._prepare(map=True)

        # Write estimates to params_est.
        for param_name in self.param_names:
            self.params_est[param_name] = self.group_params[param_name].value

        return self

    def _prepare(self, dbname=None, map_=True, load=False, verbose=0):
        """Compute posterior model by markov chain monte carlo estimation."""

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
                self.mcmc_model = pm.MCMC(self.model, db='pickle', dbname=dbname, verbose=verbose)
        else:
            # Open database
            db = pm.database.pickle.load(dbname)
            #db = pm.database.sqlite.load(dbname)
        
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
        
    def mcmc(self, samples=10000, burn=5000, thin=2, verbose=0, dbname=None, map_=True):
        """Main method for sampling. Creates and initializes the model and starts sampling.
        """
        # Set and initialize model
        self._prepare(dbname=dbname, map_=map_)
        # Draw samples
        self._sample(samples=samples, burn=burn, thin=thin, verbose=verbose, dbname=dbname)
        
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
                    param_inst.trace._trace[0] = np.concatenate((param_inst.trace._trace[0], mcmc_model.trace(param_name)()))
                else:
                    param_inst.trace = mcmc_model.trace(param_name)
            except AttributeError: # param_inst is an array
                if self.trace_subjs:
                    for i, subj_param in enumerate(param_inst):
                        if add:
                            subj_param.trace._trace[0] = np.concatenate((subj_param.trace._trace[0], mcmc_model.trace('%s_%i'%(param_name,i))()))
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

class HDDM_multi(HDDM_base):
    """Hierarchical Drift-Diffusion Model.

    This class can generate different hddms:
    - simple DDM (without inter-trial variabilities)
    - full averaging DDM (with inter-trial variabilities)
    - subject param DDM (each subject get's it's own param, see EJ's book 8.3)
    - parameter dependent on data (e.g. drift rate is dependent on stimulus
    """
    def __init__(self, data, depends_on=None, stats_on=None, model_type=None, is_subj_model=True, trace_subjs=True, load=None, col_names=None, save_stats_to=None, debug=False, no_bias=True):
        super(HDDM_multi, self).__init__(data, col_names=col_names, save_stats_to=save_stats_to, trace_subjs=trace_subjs, debug=debug, no_bias=no_bias)

        # Initialize
        if depends_on is None:
            self.depends_on = {}
        else:
            self.depends_on = copy(depends_on)
            
        #if stats_on is None:
        #    self.stats_on = []
        #else:
        #    self.stats_on = stats_on
            
        if model_type is None:
            self.model_type = 'simple'
        else:
            self.model_type = model_type

        self.is_subj_model = is_subj_model

        if self.is_subj_model:
            self.subjs = np.unique(self.data[self.col_names['subj_idx']])
            self.num_subjs = self.subjs.shape[0]

        # Define parameters for the simple and full averaged ddm.
        if self.model_type == 'simple' or self.model_type == 'simple_gpu':
            self.group_param_names = ['a', 'v', 'z', 't']
        elif self.model_type == 'full_avg' or self.model_type == 'full':
            self.group_param_names = ['a', 'v', 'V', 'z', 'Z', 't', 'T']
        else:
            raise ValueError('Model %s not recognized' % self.model_type)

        #if self.no_bias:
        #    self.group_param_names.remove('z')

        # Set function pointers
        self._models = {'simple': self._get_simple,
                        'simple_gpu': self._get_simple_gpu,
                        'full_avg': self._get_full_avg,
                        'full': self._get_full}


        self.param_names = copy(self.group_param_names)
        self.group_params = {}

        self.param_ranges = {'a_lower': 1.,
                             'a_upper': 3.,
                             'z_lower': .1,
                             'z_upper': 2.,
                             't_lower': .1,
                             't_upper': 1.,
                             'v_lower': -3.,
                             'v_upper': 3.}
        if load:
            self.mcmc_load_from_db(dbname=load)

    def _set_all_params(self):
        self._set_group_params()
        if self.is_subj_model:
            self._set_subj_params()

        return self
    
    def _set_group_params(self):
        """Set group level distributions. One distribution for each DDM parameter."""
        for group_param_name in self.group_param_names: # Loop through param names
            if group_param_name in self.depends_on.keys(): # Parameter depends on data
                depends_on = self.depends_on[group_param_name]
                uniq_data_dep = np.unique(self.data[depends_on])
                for uniq_date in uniq_data_dep:
                    tag = str(uniq_date)
                    self.group_params[group_param_name + '_' + tag] = self._get_group_param(group_param_name, tag=tag)
            else: # Parameter does not depend on data
                self.group_params[group_param_name] = self._get_group_param(group_param_name)
        
        return self

    def _set_subj_params(self):
        """Set individual subject distributions. Each subject is
        assigned one set of DDM parameter distributions which have the
        group level parameters as their parents"""
        # For each global param, create n subj_params
        self.subj_params = {}
        self.group_params_tau = {}

        # Initialize
        for param_name, param_inst in self.group_params.iteritems():
            self.subj_params[param_name] = np.empty(self.num_subjs, dtype=object)
            
        for param_name, param_inst in self.group_params.iteritems():
            # Create tau parameter for global param
            param_inst_tau = self._get_group_param_tau(param_name)
            self.group_params_tau[param_name] = param_inst_tau
            # Create num_subjs individual subject ddm parameter
            for subj_idx,subj in enumerate(self.subjs):
                self.subj_params[param_name][subj_idx] = self._get_subj_param(param_name, param_inst, param_inst_tau, int(subj))
                
        return self

    def _get_group_param(self, param, tag=None):
        """Create and return a prior distribution for [param]. [tag] is
        used in case of dependent parameters.
        """
        if tag is None:
            tag = ''
        else:
            tag = '_' + tag
        if param == 'a':
            return pm.Uniform("a%s"%tag, lower=self.param_ranges['a_lower'], upper=self.param_ranges['a_upper'])
        elif param.startswith('v'):
            return pm.Uniform("%s%s"%(param, tag), lower=self.param_ranges['v_lower'], upper=self.param_ranges['v_upper'], observed=False)
        elif param == 'V':
            return pm.Lambda("V%s"%tag, lambda x=self.fix_sv: x)
        elif param == 'z':
            if self.no_bias:
                return None
            else:
                return pm.Uniform('z%s'%tag, lower=self.param_ranges['z_lower'], upper=self.param_ranges['z_upper'])
        elif param == 'Z':
            return pm.Uniform("Z%s"%tag, lower=0, upper=1.)
        elif param == 't':
            return pm.Uniform("t%s"%tag, lower=self.param_ranges['t_lower'], upper=self.param_ranges['t_upper'], value=.2, observed=False)
        elif param == 'T':
            return pm.Uniform("T%s"%tag, lower=0, upper=1)
        elif param == 'e':
            return pm.Uniform("e%s"%tag, lower=-.7, upper=.7)
        else:
            raise ValueError("Param %s not recognized" % param)

        return self

    def _get_group_param_tau(self, param, tag=None):
        if tag is None:
            tag = '_tau'
        else:
            tag = tag + '_tau'

        return pm.Uniform(param + tag, lower=0, upper=800, plot=False)

    def _get_subj_param(self, param_name, parent_mean, parent_tau, subj_idx):
        if len(param_name) != 1: # if there is a tag attached to the param
            param = param_name[0]
            tag = param_name[1:] + '_' + str(subj_idx) # create new name for the subj parameter
        else:
            param = param_name
            tag = '_' + str(subj_idx)

        if param == 'a':
            return pm.TruncatedNormal("a%s"%tag, a=self.param_ranges['a_lower'], b=self.param_ranges['a_upper'], mu=parent_mean, tau=parent_tau, plot=False, trace=self.trace_subjs)
        elif param.startswith('v'):
            return pm.Normal("%s%s"%(param,tag), mu=parent_mean, tau=parent_tau, value=.5, observed=False, plot=False, trace=self.trace_subjs)
        elif param == 'V':
            return pm.Lambda("V%s"%tag, lambda x=parent_mean: parent_mean, plot=False, trace=self.trace_subjs)
        elif param == 'z':
            if self.no_bias:
                return None
            else:
                return pm.TruncatedNormal('z%s'%tag, a=self.param_ranges['z_lower'], b=self.param_ranges['z_upper'], mu=parent_mean, tau=parent_tau, plot=False, trace=self.trace_subjs)
        elif param == 'Z':
            return pm.TruncatedNormal("Z%s"%tag, a=0, b=1., mu=parent_mean, tau=parent_tau, plot=False, trace=self.trace_subjs)
        elif param == 't':
            return pm.TruncatedNormal("t%s"%tag, a=self.param_ranges['t_lower'], b=self.param_ranges['t_upper'], mu=parent_mean, tau=parent_tau, observed=False, plot=False, trace=self.trace_subjs)
        elif param == 'T':
            return pm.TruncatedNormal("T%s"%tag, a=0, b=1, mu=parent_mean, tau=parent_tau, plot=False, trace=self.trace_subjs)
        elif param == 'e':
            return pm.Normal("e%s"%tag, mu=parent_mean, tau=parent_tau, plot=False, trace=self.trace_subjs)
        else:
            raise ValueError("Param %s not recognized" % param)
        
    
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
            self.model = ddm + self.group_params.values() + self.group_params_tau.values() + self.subj_params.values()
        else:
            self.model = ddm + self.group_params.values()

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

    def _get_simple(self, name, data, params, idx=None):
        if idx is None:
            return WienerSimple(name,
                                value=data[self.col_names['rt']].flatten(), 
                                v=params['v'], 
                                ter=params['t'], 
                                a=params['a'], 
                                z=params['z'],
                                observed=True)
        else:
            return WienerSimple(name,
                                value=data[self.col_names['rt']].flatten(), 
                                v=params['v'][idx], 
                                ter=params['t'][idx], 
                                a=params['a'][idx], 
                                z=params['z'][idx],
                                observed=True)


    def _get_simple_gpu(self, name, data, params, idx=None):
        if idx is None:
            return WienerGPUSingle(name,
                                   value=data[self.col_names['rt']].flatten(), 
                                   v=params['v'], 
                                   ter=params['t'], 
                                   a=params['a'], 
                                   z=params['z'],
                                   observed=True)
        else:
            return WienerGPUSingle(name,
                                   value=data[self.col_names['rt']].flatten(), 
                                   v=params['v'][idx], 
                                   ter=params['t'][idx], 
                                   a=params['a'][idx],
                                   z=params['z'][idx],
                                   observed=True)

    def _get_full_avg(self, name, data, params, idx=None):
        if idx is None:
            return WienerAvg(name,
                             value=data[self.col_names['rt']].flatten(), 
                             v=params['v'], 
                             sv=params['V'],
                             ter=params['t'],
                             ster=params['T'], 
                             a=params['a'],
                             z=params['a']/2.,
                             sz=params['Z'],
                             observed=True)

        else:
            return WienerAvg(name,
                             value=data[self.col_names['rt']].flatten(), 
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
                z_trial[trl] = CenterUniform("z_%i"%trl, center=params['z'], width=params['sz'], plot=False, observed=False, trace=False)
                v_trial[trl] = pm.Normal("v_%i"%trl, mu=params['v'], tau=1/(params['sv']**2), plot=False, observed=False, trace=False)
                ter_trial[trl] = CenterUniform("ter_%i"%trl, center=params['ter'], width=params['ster'], plot=False, observed=False, trace=False)
                ddm[i][trl] = Wiener2("ddm_%i_%i"%(trl, i),
                                      value=self.data[self.col_names['rt']].flatten()[trl],
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
                z_trial[trl] = CenterUniform("z_%i"%trl, center=params['z'], width=params['sz'], plot=False, observed=False)
                v_trial[trl] = pm.Normal("v_%i"%trl, mu=params['v'], tau=1/(params['sv']**2), plot=False, observed=False)
                ter_trial[trl] = CenterUniform("ter_%i"%trl, center=params['ter'], width=params['ster'], plot=False, observed=False)
                ddm[trl] = Wiener2("ddm_%i"%trl,
                                   value=self.data[self.col_names['rt']].flatten()[trl],
                                   v=v_trial[trl],
                                   ter=ter_trial[trl], 
                                   a=param['a'],
                                   z=z_trial[trl],
                                   observed=True)

            return ddm

    def _create_ddm(self, data, params):
        """Create and return a DDM on [data] with [params].
        """
        if self.is_subj_model:
            ddm = np.empty(self.num_subjs, dtype=object)
            for i,subj in enumerate(self.subjs):
                data_subj = data[data[self.col_names['subj_idx']] == subj] # Select data belong to subj

                ddm = self._models[self.model_type]("ddm_%i_%i"%(self.idx, i), data_subj, params, idx=i)
        else: # Do not use subj params, but group ones
            ddm = self._models[self.model_type]("ddm_%i"%self.idx, data, params)

        self.idx+=1
        
        return ddm


    def _gen_stats(self):
        for param_name in self.group_params.iterkeys():
            if param_name == 'z' and self.no_bias:
                continue
            self.params_est[param_name] = np.mean(self.mcmc_model.trace(param_name)())
            self.params_est_std[param_name] = np.std(self.mcmc_model.trace(param_name)())

        self.stats['logp'] = self.mcmc_model.logp
        self.stats['dic'] = self.mcmc_model.dic
        
        # Save stats to output file
        if self.save_stats_to is not None:
            print "Saving stats to %s" % self.save_stats_to
            with open(self.save_stats_to, 'w') as fd:
                fd.write('Mean group estimates:\n')
                for name, value in self.params_est.iteritems():
                    fd.write('%s: %f\n'%(name, value))
                fd.write('\nStandard deviations of group parameters:\n')
                for name, value in self.params_est_std.iteritems():
                    fd.write('%s: %f\n'%(name, value))
                fd.write('\nGeneral model stats:\n')
                for name, value in self.stats.iteritems():
                    fd.write('%s: %f\n'%(name, value))
        return self

class HDDM_multi_lba(HDDM_multi):
    def __init__(self, *args, **kwargs):
        # Fetch out lba specific parameters
        if kwargs.has_key('fix_sv'):
            self.fix_sv = kwargs['fix_sv']
            del kwargs['fix_sv']
        else:
            self.fix_sv = None
        
        if kwargs.has_key('normalize_v'):
            self.normalize_v = kwargs['normalize_v']
            del kwargs['normalize_v']
        else:
            self.normalize_v = True

        super(HDDM_multi_lba, self).__init__(*args, **kwargs)

        self.model_type = 'lba'
        self.no_bias = False # Does not make sense for LBA
        self.resps = np.unique(self.data['response'])
        self.nresps = self.resps.shape[0]
        self.group_param_names = ['a', 'V', 'z', 't'] + ['v%i'%i for i in self.resps]

        self._models = {'lba': self._get_lba}

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

    def _get_group_param(self, param, tag=None):
        """Create and return a prior distribution for [param]. [tag] is
        used in case of dependent parameters.
        """
        if tag is None:
            tag = ''
        else:
            tag = '_' + tag
            
        if param == 'a':
            return pm.Uniform("a%s"%tag, lower=self.param_ranges['a_lower'], upper=self.param_ranges['a_upper'])
        elif param.startswith('v'):
            if self.normalize_v:
                # Normalize the sum of the drifts to 1.
                if param.startswith('v0'):
                    return pm.Uniform("%s%s"%(param, tag), lower=self.param_ranges['v_lower'], upper=self.param_ranges['v_upper'])
                else:
                    return pm.Lambda("%s%s"%(param, tag), lambda x=self.group_params['v0%s'%tag]: 1-x)
            else:
                return pm.Uniform("%s%s"%(param, tag), lower=self.param_ranges['v_lower'], upper=self.param_ranges['v_upper'])
        elif param == 'V':
            if self.fix_sv is None:
                return pm.Uniform("V%s"%tag, lower=self.param_ranges['V_lower'], upper=self.param_ranges['V_upper'])
            else:
                return pm.Lambda("V%s"%tag, lambda x=self.fix_sv: x)
        elif param == 'z':
            if self.no_bias:
                return None
            return pm.Uniform('z%s'%tag, lower=self.param_ranges['z_lower'], upper=self.param_ranges['z_upper'])
        elif param == 'Z':
            return pm.Uniform("Z%s"%tag, lower=0, upper=1.)
        elif param == 't':
            return pm.Uniform("t%s"%tag, lower=self.param_ranges['t_lower'], upper=self.param_ranges['t_upper'], value=.2, observed=False)
        elif param == 'T':
            return pm.Uniform("T%s"%tag, lower=0, upper=1)
        elif param == 'e':
            return pm.Uniform("e%s"%tag, lower=-.7, upper=.7, value=0)
        else:
            raise ValueError("Param %s not recognized" % param)

        return self

    def _get_subj_param(self, param_name, parent_mean, parent_tau, subj_idx):
        if len(param_name) != 1: # if there is a tag attached to the param
            param = param_name[0]
            tag = param_name[1:] + '_' + str(subj_idx) # create new name for the subj parameter
        else:
            param = param_name
            tag = '_' + str(subj_idx)

        if param == 'a':
            return pm.TruncatedNormal("a%s"%tag, a=self.param_ranges['a_lower'], b=self.param_ranges['a_upper'],
                                      mu=parent_mean, tau=parent_tau, plot=False, trace=self.trace_subjs)
        elif param == 'v':
            if self.normalize_v:
                # Normalize the sum of the drifts to 1.
                # Determine if v0 or v1 has been set before (dict is not ordered, so we can't know)
                if param_name.startswith('v0'):
                    other_param = 'v1'
                else:
                    other_param = 'v0'
                other_param_complete = '%s%s'%(other_param, param_name[2:])

                if self.subj_params[other_param_complete][subj_idx] is not None:
                    # Other parameter has already been set.
                    return pm.Lambda("%s%s"%(param, tag), lambda x=self.subj_params[other_param_complete][subj_idx]: 1-x, plot=False, trace=self.trace_subjs)
                else:
                    # Create new v parameter
                    return pm.TruncatedNormal("%s%s"%(param,tag), a=self.param_ranges['v_lower'], b=self.param_ranges['v_upper'], mu=parent_mean, tau=parent_tau, observed=False, plot=False, trace=self.trace_subjs)
            else: 
                return pm.TruncatedNormal("%s%s"%(param,tag), a=self.param_ranges['v_lower'], b=self.param_ranges['v_upper'], mu=parent_mean, tau=parent_tau, value=.5, observed=False, plot=False, trace=self.trace_subjs)
        elif param == 'V':
            if self.fix_sv is None:
                return pm.TruncatedNormal("V%s"%tag, a=self.param_ranges['V_lower'], b=self.param_ranges['V_upper'], mu=parent_mean, tau=parent_tau, plot=False, trace=self.trace_subjs)
            else:
                return pm.Lambda("V%s"%tag, lambda x=parent_mean: parent_mean, plot=False, trace=self.trace_subjs)
        elif param == 'z':
            if self.no_bias:
                return None
            else:
                return pm.TruncatedNormal('z%s'%tag, a=self.param_ranges['z_lower'], b=self.param_ranges['z_upper'], mu=parent_mean, tau=parent_tau, plot=False, trace=self.trace_subjs)
        elif param == 'Z':
            return pm.TruncatedNormal("Z%s"%tag, a=0, b=1., mu=parent_mean, tau=parent_tau, plot=False, trace=self.trace_subjs)
        elif param == 't':
            return pm.TruncatedNormal("t%s"%tag, a=self.param_ranges['t_lower'], b=self.param_ranges['t_upper'], mu=parent_mean, tau=parent_tau, observed=False, plot=False, trace=self.trace_subjs)
        elif param == 'T':
            return pm.TruncatedNormal("T%s"%tag, a=0, b=1, mu=parent_mean, tau=parent_tau, plot=False, trace=self.trace_subjs)
        elif param == 'e':
            return pm.Normal("e%s"%tag, mu=parent_mean, tau=parent_tau, plot=False, trace=self.trace_subjs)
        else:
            raise ValueError("Param %s not recognized" % param)

    def _get_lba(self, name, data, params, idx=None):
        if idx is None:
            return LBA(name,
                       value=data[self.col_names['rt']].flatten(),
                       resps=data[self.col_names['response']].flatten(),
                       a=params['a'],
                       z=params['z'],
                       ter=params['t'],
                       v=np.array([params['v%i'%i] for i in self.resps]).flatten(),
                       sv=params['V'],
                       observed=True)
        else:
            return LBA(name,
                       value=data[self.col_names['rt']].flatten(),
                       resps=data[self.col_names['response']].flatten(),
                       a=params['a'][idx],
                       z=params['z'][idx],
                       ter=params['t'][idx],
                       v=np.array([params['v%i'%i][idx] for i in self.resps]).flatten(),
                       sv=params['V'][idx],
                       observed=True)                       

class HDDM_multi_gpu(HDDM_multi):
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
        ddm = WienerGPUGlobal('ddm',
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

    
class HDDM_multi_effect(HDDM_multi):
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

    
# Model classes
class HDDM_full_avg(HDDM_base):
    """Basic hierarchical bayesian drift diffusion model. Estimates
    posterior parameter model for one subject. To calculate
    the likelihood, the wiener PDF from Navarro & Fuss 2009 is used."""
    def __init__(self, *args, **kwargs):
        super(HDDM_full_avg, self).__init__(*args, **kwargs)

        self.model_type = 'full_avg'

    def _set_group_params(self):
        """Define standard param model"""
        v = pm.Uniform("v", lower=-2, upper=2)
        sv = pm.Uniform("sv", lower=0.001, upper=1.)
        ster = pm.Uniform("ster", lower=0, upper=1)
        ter = pm.Uniform("ter", lower=0, upper=1)
        a = pm.Uniform("a", lower=1, upper=3)
        sz = pm.Uniform("sz", lower=0, upper=1.)
        if self.no_bias:
            z = pm.Lambda('z', lambda x=a: x/2.)
        else:
            z = pm.Uniform("z", lower=sz/2., upper=a-sz/2.)

        self.group_params = {'v':v, 'sv': sv, 'ter': ter, 'ster': ster, 
                              'z': z, 'sz': sz, 'a': a}

        self.param_names = self.group_params.keys()
        
        return self

    def _set_model(self):
        """Generate model for the whole DDM."""
        # Define Param Model
        self.model = copy(self.group_params)
        # Define wiener DDM distribution
        ddm = WienerAvg("ddm", value=self.data['rt'].flatten(), 
                        v=self.model['v'], 
                        sv=self.model['sv'], 
                        ter=self.model['ter'], 
                        ster=self.model['ster'], 
                        a=self.model['a'], 
                        z=self.model['z'], 
                        sz=self.model['sz'], 
                        observed=True)

        self.model['ddm'] = ddm
        return self

class HDDM_simple(HDDM_base):
    """Simple hierarchical bayesian drift diffusion model. Estimates
    posterior parameter model for one subject. To calculate
    the likelihood, the wiener likelihood from Navarro & Fuss 2009 is used."""
    def _set_group_params(self):
        """Define standard param model"""
        self.group_params = {}
        self.group_params['v'] = pm.Uniform("v", lower=-2, upper=2)
        self.group_params['ter'] = pm.Uniform("ter", lower=0, upper=1)
        self.group_params['a'] = pm.Uniform("a", lower=1, upper=4)
        if self.no_bias:
            self.group_params['z'] = None # pm.Lambda('z', lambda x=self.group_params['a']: x/2.)
            #pass
        else:
            self.group_params['z'] = pm.Uniform("z", lower=0, upper=self.group_params['a'])

        self.param_names = self.group_params.keys()

        self.model_type = 'simple'
        
        return self

    def _set_model(self):
        """Generate model for the whole DDM."""
        # Define Param Model
        self._set_group_params()
        self.model = copy(self.group_params)
        # Define wiener DDM distribution
        ddm = WienerSimple("ddm", value=self.data['rt'].flatten(), 
                           v=self.model['v'], 
                           ter=self.model['ter'], 
                           a=self.model['a'], 
                           z=self.model['z'], 
                           observed=True)

        self.model['ddm'] = ddm

        return self

class HDDM_lba(HDDM_base):
    """Simple hierarchical bayesian drift diffusion model. Estimates
    posterior parameter model for one subject. To calculate
    the likelihood, the wiener likelihood from Navarro & Fuss 2009 is used."""
    def _set_group_params(self):
        """Define standard param model"""
        self.group_params = {}
        self.group_params['v0'] = pm.Uniform("v0", lower=-2, upper=2)
        self.group_params['v1'] = pm.Uniform("v1", lower=-2, upper=2)
        self.group_params['ter'] = pm.Uniform("ter", lower=0, upper=1)
        self.group_params['a'] = pm.Uniform("a", lower=1, upper=4)
        if self.no_bias:
            self.group_params['z'] = None # pm.Lambda('z', lambda x=self.group_params['a']: x/2.)
            #pass
        else:
            self.group_params['z'] = pm.Uniform("z", lower=0, upper=self.group_params['a'])

        self.param_names = self.group_params.keys()

        self.model_type = 'simple'
        
        return self

    def _set_model(self):
        """Generate model for the whole DDM."""
        # Define Param Model
        self._set_group_params()
        self.model = copy(self.group_params)
        # Define wiener DDM distribution
        ddm = WienerSimple("ddm", value=self.data['rt'].flatten(), 
                           v=self.model['v'], 
                           ter=self.model['ter'], 
                           a=self.model['a'], 
                           z=self.model['z'], 
                           observed=True)

        self.model['ddm'] = ddm

        return self



class HDDM_full(HDDM_base):
    def _set_group_params(self):
        """Define standard param model"""
        v = pm.Uniform("v", lower=-2, upper=2)
        sv = pm.Uniform("sv", lower=0, upper=800.)
        ster = pm.Uniform("ster", lower=0, upper=.3) #pm.Beta("ster", alpha=1, beta=1)
        ter = pm.Uniform("ter", lower=0.01, upper=1)
        sz = pm.Uniform("sz", lower=0.01, upper=3.)
        a = pm.Uniform("a", lower=1, upper=4)#, value=5, observed=False)

        z = pm.Uniform("z", lower=.5, upper=2)# , value=.5, observed=False)

        self.group_params = {'sv': sv, 'ster': ster, 'sz': sz, 'a': a, 'v': v, 
                              'ter': ter, 'z': z}
        self.param_names = self.group_params.keys()
        self.model_type = 'full'
        
        return self

    def _set_model(self):
        """Generate model for the whole DDM."""
        # Define Param Model
        self.model = copy(self.group_params)
        # Define wiener DDM distribution
        z_center = self.group_params['z']
        sz = self.group_params['sz']
        v_center = self.group_params['v']
        sv = self.group_params['sv']
        ter_center = self.group_params['ter']
        ster = self.group_params['ster']

        trials = self.data.shape[0]
        print trials
        ddm = np.empty(trials, dtype=object)
        z_trial = np.empty(trials, dtype=object)
        v_trial = np.empty(trials, dtype=object)
        ter_trial = np.empty(trials, dtype=object)
        for i in range(trials):
            print i
            debug_here()
            #z_trial[i] = CenterUniform("z_%i"%i, center=z_center, width=sz, plot=False, observed=False, trace=False)
            #v_trial[i] = pm.Normal("v_%i"%i, mu=v_center, tau=sv, plot=False, observed=False, trace=False)
            #ter_trial[i] = CenterUniform("ter_%i"%i, center=ter_center, width=ster, plot=False, observed=False, trace=False)
            ddm[i] = Wiener2("ddm_%i"%i,
                             value=self.data['rt'][i],
                             v=self.model['v'], #v_trial[i],
                             ter=self.model['ter'],#ter_trial[i], 
                             a=self.model['a'],
                             z=self.model['z'], #z_trial[i],
                             observed=True, trace=False)

        self.model['z_trial'] = z_trial
        self.model['ter_trial'] = ter_trial
        self.model['v_trial'] = v_trial
        self.model['ddm'] = ddm
        return self

def get_subj_hddm(base=HDDM_simple):
    class HDDM_subjs(base):
        """Hierarchical bayesian drift-diffusion model for multiple
        subjects. In essence, individual subject parameters are itself
        samples from global parameter model."""
        def __init__(self, *args, **kwargs):
            super(HDDM_subjs, self).__init__(*args, **kwargs)
            self.is_subj_model = True
            self.num_subjs = np.unique(self.data['subj_idx']).shape[0]
            for i in range(self.num_subjs):
                idx = self.data['subj_idx'] == i

        def _set_group_params_tau(self, group_params=None):
            """Generate corresponding variance (tau) model for every global param distribution."""
            if group_params is None:
                group_params = self.group_params

            self.group_params_tau = {}
            for param_name, param in self.group_params.iteritems():
                if param_name[0] == 's': # if param is for variance
                    self.group_params_tau[param_name] = pm.Uniform(param_name+'_tau',
                                                                    lower=10,
                                                                    upper=600, plot=False)
                else:
                    self.group_params_tau[param_name] = pm.Uniform(param_name+'_tau',
                                                                    lower=0,
                                                                    upper=600, plot=False)
            return self

        def _set_subj_params(self, group_params=None, group_params_tau=None):
            """Generate param model for each parameter for each individual subject."""
            if group_params is None:
                group_params = self.group_params
            if group_params_tau is None:
                group_params_tau = self.group_params_tau

            self.subj_params = {}
            # Define individual subject param model
            for param_name, param in group_params.iteritems():
                self.subj_params[param_name] = np.empty(self.num_subjs, dtype=object)

            for i in range(self.num_subjs):
                # Define variance params
                for param_name, param in group_params.iteritems():
                    if param_name[0] == 's': # if variance param, use truncated normal (values can't be negative)
                        self.subj_params[param_name][i] = pm.TruncatedNormal('%s_%i'%(param_name,i), a=0, b=1.,
                                                                             mu=param,
                                                                             tau=group_params_tau[param_name],
                                                                             plot=False, trace=self.trace_subjs)

                # Set custom params for v, ter, a and z
                try:
                    self.subj_params['v'][i] = pm.Normal('v_%i'%i,
                                                         mu = group_params['v'],
                                                         tau = group_params_tau['v'], plot=False, trace=self.trace_subjs)
                except KeyError:
                    pass
                
                try:
                    self.subj_params['ter'][i] = pm.TruncatedNormal('ter_%i'%i,
                                                                    mu = group_params['ter'],
                                                                    tau = group_params_tau['ter'],
                                                                    a = 0,
                                                                    b = 1, plot=False, trace=self.trace_subjs)
                except KeyError:
                    pass

                try:
                    self.subj_params['a'][i] = pm.TruncatedNormal('a_%i'%i, a=1, b=4,
                                                                  mu=group_params['a'],
                                                                  tau=group_params_tau['a'],
                                                                  plot=False, trace=self.trace_subjs)
                except KeyError:
                    pass

                try:
                    if not self.no_bias:
                        self.subj_params['z'][i] = pm.TruncatedNormal('z_%i'%i, a=0,
                                                                     b=self.subj_params['a'][i],
                                                                     mu=group_params['z'],
                                                                     tau=group_params_tau['z'], plot=False, trace=self.trace_subjs)
                    else:
                        self.subj_params['z'][i] = None #pm.Lambda('z_%i'%i, lambda x=self.subj_params['a'][i]: x/2., plot=False, trace=self.trace_subjs)
                except KeyError:
                    pass

        def _set_all_params(self):
            self._set_group_params()
            self._set_group_params_tau()

            # Set individual params
            self._set_subj_params()
            return self
        
        def plot_subjs(self):
            """Plot real and estimated RT model for each subject."""
            size = np.ceil(np.sqrt(self.num_subjs))
            for subj in range(self.num_subjs):
                # Get subj parameters
                subj_params = self.params_est_ind[subj]
                # Get response times for subj
                idx = (self.data['subj_idx'] == subj) & (self.data['response']==1)
                resps_upper = self.data['rt'][idx]
                idx = (self.data['subj_idx'] == subj) & (self.data['response']==0)
                resps_lower = np.abs(self.data['rt'][idx])

                plt.subplot(size, size, subj+1)
                self._plot(resps_upper, resps_lower, params=subj_params)

            return self

        def _gen_stats(self):
            super(HDDM_subjs, self)._gen_stats()

            self.params_est_ind = []
            # Set individual subject parameters
            params_est_subj = {}
            for subj in range(self.num_subjs):
                # Get parameters from model stats
                for param_name in self.param_names:
                    if param_name == 'z' and self.no_bias:
                        continue
                    params_est_subj[param_name] = np.mean(self.mcmc_model.trace(param_name+'_'+str(subj))())

                self.params_est_ind.append(copy(params_est_subj))

            return self
        
    return HDDM_subjs

class HDDM_full_subj(get_subj_hddm(base=HDDM_full)):
    def _set_model(self):
        """Generate model for the whole DDM."""
        # Define Param Model
        self.model = copy(self.group_params)

        # Init arrays
        z_trial = np.empty(self.num_subjs, dtype=object)
        v_trial = np.empty(self.num_subjs, dtype=object)
        ter_trial = np.empty(self.num_subjs, dtype=object)
        ddm = np.empty(self.num_subjs, dtype=object)
        for i in range(self.num_subjs):
            # Relable for conviency
            z_center = self.subj_params['z'][i]
            sz = self.subj_params['sz'][i]
            v_center = self.subj_params['v'][i]
            sv = self.subj_params['sv'][i]
            ter_center = self.subj_params['ter'][i]
            ster = self.subj_params['ster'][i]

            # Init arrays
            trials = self.data[self.data['subj_idx']==i].shape[0]
            ddm[i] = np.empty(trials, dtype=object)
            z_trial[i] = np.empty(trials, dtype=object)
            v_trial[i] = np.empty(trials, dtype=object)
            ter_trial[i] = np.empty(trials, dtype=object)

            for j in range(trials):
                z_trial[i][j] = CenterUniform("z_%i_%i"%(i,j), center=z_center, width=sz, plot=False, observed=False, trace=False)
                v_trial[i][j] = pm.Normal("v_%i_%i"%(i,j), mu=v_center, tau=sv, plot=False, observed=False, trace=False)
                ter_trial[i][j] = CenterUniform("ter_%i_%i"%(i,j), center=ter_center, width=ster, plot=False, observed=False, trace=False)
                ddm[i][j] = Wiener2("ddm_%i_%i"%(i,j),
                                    value=self.data[self.data['subj_idx']==i]['rt'].flatten()[j],
                                    v=v_trial[i][j],
                                    ter=ter_trial[i][j], 
                                    a=self.model['a'],
                                    z=z_trial[i][j],
                                    observed=True)

        self.model['z_trial'] = z_trial
        self.model['ter_trial'] = ter_trial
        self.model['v_trial'] = v_trial
        self.model['ddm'] = ddm

        return self

    
class HDDM_full_avg_subj(get_subj_hddm(base=HDDM_full_avg)):
    def _set_model(self):
        """Generate model for the HDDM."""
        # Define DDM likelihood model for each subject
        ddm_subjs = np.empty(self.num_subjs, dtype=object)
        for i in range(self.num_subjs):
            data_subj = self.data['rt'][self.data['subj_idx'] == i]
            ddm_subjs[i] = WienerAvg("ddm%i" % i,
                                     value = data_subj,
                                     v = self.subj_params['v'][i],
                                     sv = self.subj_params['sv'][i],
                                     z = self.subj_params['z'][i],
                                     sz = self.subj_params['sz'][i],
                                     ter = self.subj_params['ter'][i],
                                     ster = self.subj_params['ster'][i],
                                     a = self.subj_params['a'][i],
                                     observed=True)

        # Combine all model
        self.model = self.group_params.values() + self.group_params_tau.values() + self.subj_params.values() + [ddm_subjs]
        return self

class HDDM_simple_subjs(get_subj_hddm(base=HDDM_simple)):
    def _set_model(self):
        """Generate model for the HDDM."""
        # Define DDM likelihood model for each subject
        ddm_subjs = np.empty(self.num_subjs, dtype=object)
        for i in range(self.num_subjs):
            data_subj = self.data['rt'][self.data['subj_idx'] == i]
            ddm_subjs[i] = WienerSimple("ddm%i" % i,
                                        value = data_subj,
                                        v = self.subj_params['v'][i],
                                        z = self.subj_params['z'][i],
                                        ter = self.subj_params['ter'][i],
                                        a = self.subj_params['a'][i], 
                                        observed=True)

        # Combine all model
        self.model = self.group_params.values() + self.group_params_tau.values() + self.subj_params.values() + [ddm_subjs]
        return self

def difference_prior(delta):
    # See Wagenmakers et al 2010, equation 14
    if type(delta) is int:
        if delta<=0:
            return 1+delta
        else:
            return 1-delta
    else:
        out = copy(delta)
        out[delta <= 0] = 1+delta[delta <= 0]
        out[delta > 0] = 1-delta[delta > 0]
        return out

def interpolate_trace(x, trace, range=(-1,1), bins=100):
    import scipy.interpolate

    x_histo = np.linspace(range[0], range[1], bins)
    histo = np.histogram(trace, bins=bins, range=range, normed=True)[0]
    interp = scipy.interpolate.InterpolatedUnivariateSpline(x_histo, histo)(x)

    return interp

def uniform(x, lower, upper):
    y = np.ones(x.shape, dtype=np.float)/(upper-lower)
    #y[x<lower] = 0.
    #y[x>upper] = 0.

    return y

def savage_dickey(post_trace, range=(-1,1), bins=100, plot=False, title=None, savefig=None, prior_trace=None, prior_y=None, plot_prior=True, label=None):
    # Calculate Savage-Dickey density ratio test, see Wagenmakers et al 2010
    # Estimate density of posterior
    # Calculate normalized histogram (density)
    x = np.linspace(range[0], range[1], bins)
    if prior_trace is not None:
        prior0 = interpolate_trace(0, prior_trace, range=range, bins=bins)
        prior = interpolate_trace(x, prior_trace, range=range, bins=bins)
    elif prior_y is not None:
        import scipy.interpolate
        prior0 = scipy.interpolate.InterpolatedUnivariateSpline(x, prior_y)(0)
        prior = prior_y
    else:
        assert ValueError, "Supply either prior_trace or prior_y keyword arguments"
        
    posterior0 = interpolate_trace(0, post_trace, range=range, bins=bins)
    posterior = interpolate_trace(x, post_trace, range=range, bins=bins)

    # Calculate Savage-Dickey density ratio at x=0
    sav_dick = posterior0 / prior0

    if plot:
        if label is None:
            label='posterior'
        plt.plot(x, posterior, label=label, lw=2.)
        if plot_prior:
            plt.plot(x, prior, label='prior', lw=2.)
        plt.axvline(x=0, lw=1., color='k')
        plt.ylim(ymin=0)
        if title:
            plt.title(title)
        if savefig:
            plt.savefig('plots/'+savefig+'.png')

    return sav_dick #, prior, posterior, prior0, posterior0
    

def call_mcmc((model_class, data, dbname, rnd, kwargs)):
    # Randomize seed
    np.random.seed(int(rnd))

    model = model_class(data, **kwargs)
    model.mcmc(dbname=dbname)
    model.mcmc_model.db.close()

def create_tag_names(tag, chains=None):
    import multiprocessing
    if chains is None:
        chains = multiprocessing.cpu_count()
    tag_names = []
    # Create copies of data and the corresponding db names
    for chain in range(chains):
        tag_names.append("db/mcmc%s%i.pickle"% (tag,chain))

    return tag_names

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

def R_hat(samples):
    n, num_chains = samples.shape # n=num_samples
    chain_means = np.mean(samples, axis=1)
    # Calculate between-sequence variance
    between_var = n * np.var(chain_means, ddof=1)

    chain_var = np.var(samples, axis=1, ddof=1)
    within_var = np.mean(chain_var)

    marg_post_var = ((n-1.)/n) * within_var + (1./n) * between_var # 11.2
    R_hat_sqrt = np.sqrt(marg_post_var/within_var)

    return R_hat_sqrt

def test_chain_convergance(models):
    # Calculate R statistic to check for chain convergance (Gelman at al 2004, 11.4)
    params = models[0].group_params
    R_hat_param = {}
    for param_name in params.iterkeys():
        # Calculate mean for each chain
        num_samples = models[0].group_params[param_name].trace().shape[0] # samples
        num_chains = len(models)
        samples = np.empty((num_chains, num_samples))
        for i,model in enumerate(models):
            samples[i,:] = model.group_params[param_name].trace()

        R_hat_param[param_name] = R_hat(samples)

    return R_hat_param

def load_gene_data(exclude_missing=None, exclude_inst_stims=True):
    pos_stims = ('A', 'C', 'E')
    neg_stims = ('B', 'D', 'F')
    fname = 'data/gene/Gene_tst1_RT.csv'
    data = np.recfromcsv(fname)
    data['subj_idx'] = data['subj_idx']-1 # Offset subj_idx to start at 0

    data['rt'] = data['rt']/1000. # Time in seconds
    # Remove outliers
    data = data[data['rt'] > .25]

    # Exclude subjects for which there is no particular gene.
    if exclude_missing is not None:
        if isinstance(exclude_missing, (tuple, list)):
            for exclude in exclude_missing:
                data = data[data[exclude] != '']
        else:
            data = data[data[exclude_missing] != '']

    # Add convenience columns
    # First stim and second stim
    cond1 = np.empty(data.shape, dtype=np.dtype([('cond1','S1')]))
    cond2 = np.empty(data.shape, dtype=np.dtype([('cond2','S1')]))
    cond_class = np.empty(data.shape, dtype=np.dtype([('conf','S2')]))
    contains_A = np.empty(data.shape, dtype=np.dtype([('contains_A',np.bool)]))
    contains_B = np.empty(data.shape, dtype=np.dtype([('contains_B',np.bool)]))
    include = np.empty(data.shape, dtype=np.bool)
    
    for i,cond in enumerate(data['cond']):
        cond1[i] = cond[0]
        cond2[i] = cond[1]
        # Condition contains A or B, with AB trials excluded
        contains_A[i] = (((cond[0] == 'A') or (cond[1] == 'A')) and not ((cond[0] == 'B') or (cond[1] == 'B')),)
        contains_B[i] = (((cond[0] == 'B') or (cond[1] == 'B')) and not ((cond[0] == 'A') or (cond[1] == 'A')),)

        if cond[0] in pos_stims and cond[1] in pos_stims:
            cond_class[i] = 'WW'
        elif cond[0] in neg_stims and cond[1] in neg_stims:
            cond_class[i] = 'LL'
        else:
            cond_class[i] = 'WL'

        # Exclude instructed stims
        include[i] = cond.find(data['instructed1'][i])
        
    # Append rows to data
    data = rec.append_fields(data, names=['cond1','cond2','conf', 'contains_A', 'contains_B'],
                             data=(cond1,cond2,cond_class,contains_A,contains_B), dtypes=['S1','S1','S2','B1','B1'], usemask=False)

    return data[include]

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

def csv2rec(fname):
    return np.lib.io.recfromcsv(fname)

def parse_config_file(fname, load=False):
    import ConfigParser
    
    config = ConfigParser.ConfigParser()
    config.read(fname)
    
    #####################################################
    # Parse config file
    data_fname = config.get('data', 'load')
    save = config.get('data', 'save')
    #format_ = config.get('data', 'format')

    data = np.recfromcsv(data_fname)
    
    try:
        model_type = config.get('model', 'type')
    except ConfigParser.NoOptionError:
        model_type = 'simple'

    try:
        is_subj_model = config.getboolean('model', 'is_subj_model')
    except ConfigParser.NoOptionError:
        is_subj_model = True

    try:
        no_bias = config.getboolean('model', 'no_bias')
    except ConfigParser.NoOptionError:
        no_bias = True

    try:
        debug = config.getboolean('model', 'debug')
    except ConfigParser.NoOptionError:
        debug = False

    try:
        dbname = config.get('mcmc', 'dbname')
    except ConfigParser.NoOptionError:
        dbname = None

    if model_type == 'simple' or model_type == 'simple_gpu':
        group_param_names = ['a', 'v', 'z', 't']
    elif model_type == 'full_avg' or model_type == 'full':
        group_param_names = ['a', 'v', 'V', 'z', 'Z', 't', 'T']
    elif model_type == 'lba':
        group_param_names = ['a', 'v', 'z', 't', 'V']
    else:
        raise NotImplementedError('Model type %s not implemented'%model_type)

    # Get depends
    depends = {}
    for param_name in group_param_names:
        try:
            # Multiple depends can be listed (separated by a comma)
            depends[param_name] = config.get('depends', param_name).split(',')
        except ConfigParser.NoOptionError:
            pass

    # MCMC values
    try:
        samples = config.getint('mcmc', 'samples')
    except ConfigParser.NoOptionError:
        samples = 10000
    try:
        burn = config.getint('mcmc', 'burn')
    except ConfigParser.NoOptionError:
        burn = 5000
    try:
        thin = config.getint('mcmc', 'thin')
    except ConfigParser.NoOptionError:
        thin = 3
    try:
        verbose = config.getint('mcmc', 'verbose')
    except ConfigParser.NoOptionError:
        verbose = 0

    print "Creating model..."
    if model_type != 'lba':
        m = HDDM_multi(data, is_subj_model=is_subj_model, no_bias=no_bias, depends_on=depends, save_stats_to=save, debug=debug)
    else:
        m = HDDM_multi_lba(data, is_subj_model=is_subj_model, no_bias=no_bias, depends_on=depends, save_stats_to=save, debug=debug)

    if not load:
        print "Sampling... (this can take some time)"
        m.mcmc(samples=samples, burn=burn, thin=thin, verbose=verbose, dbname=dbname)
    else:
        m.mcmc_load_from_db(dbname=dbname)

    return m

if __name__=='__main__':
    import sys
    parse_config_file(sys.argv[1])
