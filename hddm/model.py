"""
.. module:: HDDM
   :platform: Agnostic
   :synopsis: Definition of HDDM models.

.. moduleauthor:: Thomas Wiecki <thomas.wiecki@gmail.com>
                  Imri Sofer <imrisofer@gmail.com>


"""

from __future__ import division
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

import hddm

import kabuki

from kabuki.hierarchical import Parameter
from copy import copy, deepcopy
from time import time
from matplotlib.mlab import rec_drop_fields

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

            plot_var : bool
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
        data = hddm.utils.flip_errors(data)

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
        else:
            self.wiener_params = wiener_params
        wp = self.wiener_params

        self.wfpt = deepcopy(hddm.likelihoods.wfpt_like)

        self.wfpt.rv.wiener_params = wp

        self.kwargs = kwargs

        super(hddm.HDDM, self).__init__(data, include=include, **kwargs)

    def get_params(self):
        """Returns list of model parameters.
        """
        # These boundaries are largely based on a meta-analysis of
        # reported fit values.
        # See: Matzke & Wagenmakers 2009
        params = [Parameter('a', lower=.3, upper=4),
                  Parameter('v', lower=-15., upper=15., init=0.),
                  Parameter('t', lower=.1, upper=.9, init=.1), # Change lower to .2 as in MW09?
                  Parameter('z', lower=.2, upper=0.8, init=.5,
                            default=.5, optional=True),
                  Parameter('V', lower=0., upper=3.5, default=0,
                            optional=True),
                  Parameter('Z', lower=0., upper=1.0, init=.1,
                            default=0, optional=True),
                  Parameter('T', lower=0., upper=0.8, init=.1,
                            default=0, optional=True),
                  Parameter('wfpt', is_bottom_node=True)]

        return params

    def get_subj_node(self, param):
        """Create and return a Normal (in case of an effect or
        drift-parameter) or Truncated Normal (otherwise) distribution
        for 'param' centered around param.group with standard deviation
        param.var and initialization value param.init.

        This is used for the individual subject distributions.

        """
        if param.name.startswith('e') or param.name.startswith('v'):
            return pm.Normal(param.full_name,
                             mu=param.group,
                             tau=param.var**-2,
                             plot=self.plot_subjs,
                             trace=self.trace_subjs,
                             value=param.init)

        else:
            return pm.TruncatedNormal(param.full_name,
                                      a=param.lower,
                                      b=param.upper,
                                      mu=param.group,
                                      tau=param.var**-2,
                                      plot=self.plot_subjs,
                                      trace=self.trace_subjs,
                                      value=param.init)

    def get_bottom_node(self, param, params):
        """Create and return the wiener likelihood distribution
        supplied in 'param'.

        'params' is a dictionary of all parameters on which the data
        depends on (i.e. condition and subject).

        """
        if param.name == 'wfpt':
            return self.wfpt(param.full_name,
                             value=param.data['rt'].flatten(),
                             v=params['v'],
                             a=params['a'],
                             z=self.get_node('z',params),
                             t=params['t'],
                             Z=self.get_node('Z',params),
                             T=self.get_node('T',params),
                             V=self.get_node('V',params),
                             observed=True)

        else:
            raise KeyError, "Groupless parameter named %s not found." % param.name


class HDDMContaminant(HDDM):
    """Contaminant HDDM Super class

    :Optional:
        init : bool
            Use EZ to initialize parameters (default: True)

    """
    def __init__(self, *args, **kwargs):
        super(hddm.model.HDDMContaminant, self).__init__(*args, **kwargs)
        self.cont_res = None


    def get_bottom_node(self, param, params):
        if param.name == 'wfpt':
            return self.wfpt(param.full_name,
                             value=param.data['rt'].flatten(),
                             cont_x=params['x'],
                             v=params['v'],
                             a=params['a'],
                             z=self.get_node('z',params),
                             t=params['t'],
                             Z=self.get_node('Z',params),
                             T=self.get_node('T',params),
                             V=self.get_node('V',params),
                             t_min=self.t_min,
                             t_max=self.t_max,
                             observed=True)

        elif param.name == 'x':
            rts = param.data['rt']
            outlier = np.empty(rts.shape, dtype=np.bool)
            outlier[np.abs(rts) < params['t'].value] = True
            outlier[np.abs(rts) >= params['t'].value] = False
            return pm.Bernoulli(param.full_name, p=params['pi'], size=len(param.data['rt']), plot=False, value=outlier)

        else:
            raise KeyError, "Groupless subj parameter %s not found" % param.name

    def cont_report(self, cont_threshold=0.5, plot=True):
        """Create conaminate report

        :Arguments:
            cont_threshold : float
                the threshold tthat define an outlier (default: 0.5)
            plot : bool
                should the result be plotted (default: True)

        :Returns:
            report : dict
                * cont_idx : index of outlier above threshold
                * probs : Probability RT is an outlier
                * rts : RTs of outliers

        """
        hm = self
        data_dep = hm._get_data_depend()

        self.cont_res = {}
        if self.is_group_model:
            subj_list = self._subjs
        else:
            subj_list = [0]

        conds = hm.params_dict['x'].subj_nodes.keys()


        #loop over subjects
        for subj_idx, subj in enumerate(subj_list):
            n_cont = 0
            rts = np.empty(0)
            probs = np.empty(0)
            cont_idx = np.empty(0, np.int)
            print "#$#$#$# outliers for subject %s #$#$#$#" % subj
            #loop over conds
            for cond in conds:
                print "*********************"
                print "looking at %s" % cond
                nodes = hm.params_dict['x'].subj_nodes[cond]
                if self.is_group_model:
                    node = nodes[subj_idx]
                else:
                    node = nodes
                m = np.mean(node.trace(), 0)

                #look for outliers with high probabilty
                idx = np.where(m > cont_threshold)[0]
                n_cont += len(idx)
                if idx.size > 0:
                    print "found %d probable outliers in %s" % (len(idx), cond)
                    wfpt = list(node.children)[0]
                    data_idx = [x for x in data_dep if x[3]==cond][0][0]['data_idx']
                    for i_cont in range(len(idx)):
                        print "rt: %8.5f prob: %.2f" % (wfpt.value[idx[i_cont]], m[idx[i_cont]])
                    cont_idx = np.r_[cont_idx, data_idx[idx]]
                    rts = np.r_[rts, wfpt.value[idx]]
                    probs = np.r_[probs, m[idx]]

                    #plot outliers
                    if plot:
                        plt.figure()
                        mask = np.ones(len(wfpt.value), dtype=bool)
                        mask[idx] = False
                        plt.plot(wfpt.value[mask], np.zeros(len(mask) - len(idx)), 'b.')
                        plt.plot(wfpt.value[~mask], np.zeros(len(idx)), 'ro')
                        plt.title(wfpt.__name__)
                        plt.xlabel('RTs for lower (negative) and upper (positive) boundary responses.')
                #report the next higest probability outlier
                next_outlier = max(m[m < cont_threshold])
                print "probability of the next most probable outlier: %.2f" % next_outlier

            print "!!!!!**** %d probable outliers were found in the data ****!!!!!" % n_cont
            single_cont_res = {}
            single_cont_res['cont_idx'] = cont_idx
            single_cont_res['rts'] = rts
            single_cont_res['probs'] = probs
            if self.is_group_model:
                self.cont_res[subj] = single_cont_res
            else:
                self.cont_res = single_cont_res

        if plot:
            plt.show()


        return self.cont_res


    def remove_conts(self, cont_threshold=.5):
        """
        Return the data without the contaminants.

        :Arguments:
            cutoff : float
                the probablity that defines an outlier (deafult 0.5)

        :Returns:
            cleaned_data : ndarray

        """
        # Generate cont_report
        if not self.cont_res:
            self.cont_report(cont_threshold = cont_threshold)

        new_data = []
        if self.is_group_model:
            subj_list = self._subjs
        else:
            subj_list = [0]

        # Loop over subjects and append cleaned data
        for subj in subj_list:
            if self.is_group_model:
                cont_res = self.cont_res[subj]
                data_subj = self.data[self.data['subj_idx'] == subj]
            else:
                cont_res = self.cont_res
                data_subj = self.data
            idx = np.ones(len(data_subj), bool)
            idx[cont_res['cont_idx'][cont_res['probs'] >= cont_threshold]] = 0
            new_data.append(data_subj[idx])

        data_all = np.concatenate(new_data)
        data_all['rt'] = np.abs(data_all['rt'])

        return data_all


class HDDMContUnif(HDDMContaminant):
    """Contaminant HDDM Uniform class

    Outliers are modeled using a uniform distribution over responses
    and reaction times.

    :Optional:
        init : bool
            Use EZ to initialize parameters (default: True)

    """
    def __init__(self, *args, **kwargs):
        super(hddm.model.HDDMContUnif, self).__init__(*args, **kwargs)
        self.params = self.params[:-1] + \
                 [Parameter('pi', lower=0.01, upper=0.1),
                  Parameter('x', is_bottom_node=True),
                  Parameter('wfpt', is_bottom_node=True)]

        self.t_min = 0
        self.t_max = max(abs(self.data['rt']))
        wp = self.wiener_params
        self.wfpt = hddm.likelihoods.general_WienerCont(err=wp['err'],
                                                        nT=wp['nT'],
                                                        nZ=wp['nZ'],
                                                        use_adaptive=wp['use_adaptive'],
                                                        simps_err=wp['simps_err'])

    def get_bottom_node(self, param, params):
        if param.name == 'wfpt':
            return self.wfpt(param.full_name,
                             value=param.data['rt'].flatten(),
                             cont_x=params['x'],
                             v=params['v'],
                             a=params['a'],
                             z=self.get_node('z',params),
                             t=params['t'],
                             Z=self.get_node('Z',params),
                             T=self.get_node('T',params),
                             V=self.get_node('V',params),
                             t_min=self.t_min,
                             t_max=self.t_max,
                             observed=True)

        elif param.name == 'x':
            rts = param.data['rt']
            outlier = np.empty(rts.shape, dtype=np.bool)
            outlier[np.abs(rts) < params['t'].value] = True
            outlier[np.abs(rts) >= params['t'].value] = False
            return pm.Bernoulli(param.full_name, p=params['pi'], size=len(param.data['rt']), plot=False, value=outlier)

        else:
            raise KeyError, "Groupless subj parameter %s not found" % param.name



if __name__ == "__main__":
    import doctest
    doctest.testmod()
