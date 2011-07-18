"""
.. module:: HDDM
   :platform: Agnostic
   :synopsis: Definition of HDDM models.

.. moduleauthor:: Thomas Wiecki <thomas.wiecki@gmail.com>
                  Imri Sofer <imrisofer@gmail.com>


"""

from __future__ import division
import numpy as np
import numpy.lib.recfunctions as rec
import pymc as pm
import matplotlib.pyplot as plt
from copy import copy

import hddm

import kabuki

from kabuki.hierarchical import Parameter

class HDDM(kabuki.Hierarchical):
    """Implements the hierarchical Ratcliff drift-diffusion model
    using the Navarro & Fuss likelihood and numerical integration over
    the variability parameters.

    :Arguments:
        data : numpy.recarray
            Input data with a row for each trial.
             Must contain the following columns:
              * 'rt': Reaction time of trial in seconds.
              * 'response': Binary response (e.g. 0->error, 1->correct)
             May contain:
              * 'subj_idx': A unique ID (int) of the subject.
              * Other user-defined columns that can be used in depends on keyword.
    :Example:
        >>> data, params = hddm.generate.gen_rand_data() # gen data
        >>> model = hddm.HDDM(data) # create object
        >>> mcmc = model.mcmc() # Create pymc.MCMC object
        >>> mcmc.sample() # Sample from posterior
    :Optional:
        include : iterable
            Optional inter-trial variability parameters to include.
             Can be any combination of 'V', 'Z' and 'T'. Passing the string
            'all' will include all three.
            
            Note: Including 'Z' and/or 'T' will increase run time significantly!
 
            is_group_model : bool
                If True, this results in a hierarchical
                model with separate parameter distributions for each
                subject. The subject parameter distributions are
                themselves distributed according to a group parameter
                distribution.
    
                If not provided, this parameter is set to True if data
                provides a column 'subj_idx' and False otherwise.
        
            depends_on : dict
                Specifies which parameter depends on data
                of a column in data. For each unique element in that
                column, a separate set of parameter distributions will be
                created and applied. Multiple columns can be specified in
                a sequential container (e.g. list)
    
                :Example: 
    
                    >>> hddm.HDDM(data, depends_on={'v':'difficulty'})
                    
                    Separate drift-rate parameters will be estimated
                    for each difficulty. Requires 'data' to have a
                    column difficulty.
        

            bias : bool 
                Whether to allow a bias to be estimated. This
                is normally used when the responses represent
                left/right and subjects could develop a bias towards
                responding right. This is normally never done,
                however, when the 'response' column codes
                correct/error.
    
            plot_tau : bool
                 Plot group variability parameters when calling pymc.Matplot.plot()
                 (i.e. variance of Normal distribution.)
    
            wiener_params : dict
                 Parameters for wfpt evaluation and
                 numerical integration.
                 
                 :Parameters: 
                     * err: Error bound for wfpt (default 1e-4)
                     * nT: Maximum depth for numerical integration for T (default 2)
                     * nZ: Maximum depth for numerical integration for Z (default 2)
                     * use_adaptive: Whether to use adaptive numerical integration (default True)
                     * simps_err: Error bound for Simpson integration (default 1e-3)

    """
        
    def __init__(self, data, bias=False,
                 include=(), wiener_params=None, **kwargs):
        
        # Flip sign for lower boundary RTs
        self.data = hddm.utils.flip_errors(data)

        include = set(include)
        
        if include is not None:
            if include == 'all':
                [include.add(param) for param in ('T','V','Z')]
            else:
                [include.add(param) for param in include]

        if bias:
            include.add('z')

        if wiener_params is None:
            self.wiener_params = {'err': 1e-4, 'nT':2, 'nZ':2,
                                  'use_adaptive':1,
                                  'simps_err':1e-3}
            self.wfpt = hddm.likelihoods.WienerFullIntrp
        else:
            self.wiener_params = wiener_params
            wp = self.wiener_params
            self.wfpt = hddm.likelihoods.general_WienerFullIntrp_variable(err=wp['err'], nT=wp['nT'], nZ=wp['nZ'], use_adaptive=wp['use_adaptive'], simps_err=wp['simps_err'])

        self.params = self.get_params()

        super(hddm.model.HDDM, self).__init__(data, include=include, **kwargs)

    def get_params(self):
        """Returns list of model parameters.
        """
        # These boundaries are largely based on a meta-analysis of
        # reported fit values. 
        # See: Matzke & Wagenmakers 2009
        params = [Parameter('a', True, lower=.8, upper=4),
                  Parameter('v', True, lower=-6., upper=6.), 
                  Parameter('t', True, lower=.1, upper=.9, init=.1), # Change lower to .2 as in MW09?
                  Parameter('z', True, lower=.2, upper=0.8, init=.5, 
                            default=.5, optional=True),
                  Parameter('V', True, lower=0., upper=3.5, default=0, 
                            optional=True),
                  Parameter('Z', True, lower=0., upper=1.3, init=.1, 
                            default=0, optional=True),
                  Parameter('T', True, lower=0., upper=0.8, init=.1, 
                            default=0, optional=True),
                  Parameter('wfpt', False)]
        
        return params
    
    def get_root_node(self, param):
        """Create and return a uniform prior distribution for root
        parameter 'param'.

        This is used for the group distributions.

        """
        return pm.Uniform(param.full_name,
                          lower=param.lower,
                          upper=param.upper,
                          value=param.init,
                          verbose=param.verbose)

    def get_tau_node(self, param):
        """Create and return a Uniform prior distribution for the
        variability parameter 'param'.
        
        Note, that we chose a Uniform distribution rather than the
        more common Gamma (see Gelman 2006: "Prior distributions for
        variance parameters in hierarchical models").

        This is used for the variability fo the group distribution.

        """
        return pm.Uniform(param.full_name, lower=0., upper=1., value=.1, plot=self.plot_tau)

    def get_child_node(self, param):
        """Create and return a Normal (in case of an effect or
        drift-parameter) or Truncated Normal (otherwise) distribution
        for 'param' centered around param.root with standard deviation
        param.tau and initialization value param.init.

        This is used for the individual subject distributions.

        """
        if param.name.startswith('e') or param.name.startswith('v'):
            return pm.Normal(param.full_name,
                             mu=param.root,
                             tau=param.tau**-2,
                             plot=self.plot_subjs,
                             value=param.init)

        else:
            return pm.TruncatedNormal(param.full_name,
                                      a=param.lower,
                                      b=param.upper,
                                      mu=param.root, 
                                      tau=param.tau**-2,
                                      plot=self.plot_subjs,
                                      value=param.init)
    
    def get_rootless_child(self, param, params):
        """Create and return the wiener likelihood distribution
        supplied in 'param'. 

        'params' is a dictionary of all parameters on which the data
        depends on (i.e. condition and subject).

        """
        if param.name == 'wfpt':
            return self.wfpt(param.full_name,
                             value=param.data['rt'].flatten(),
                             v = params['v'],
                             a = params['a'],
                             z = self.get_node('z',params),
                             t = params['t'],
                             Z = self.get_node('Z',params),
                             T = self.get_node('T',params),
                             V = self.get_node('V',params),
                             observed=True)

        else:
            raise KeyError, "Rootless parameter named %s not found." % param.name

if __name__ == "__main__":
    import doctest
    doctest.testmod()
