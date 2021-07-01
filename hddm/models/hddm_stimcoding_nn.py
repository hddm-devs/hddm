from copy import copy
import numpy as np
from collections import OrderedDict

from kabuki.hierarchical import Knode
from kabuki.utils import stochastic_from_dist
from hddm.models import HDDM
from hddm.keras_models import load_mlp
from hddm.cnn.wrapper import load_cnn
import hddm

import wfpt
from functools import partial

class HDDMnnStimCoding(HDDM):
    """HDDMnn model that can be used when stimulus coding and estimation
    of bias (i.e. displacement of starting point z) is required.

    In that case, the 'resp' column in your data should contain 0 and
    1 for the chosen stimulus (or direction), not whether the response
    was correct or not as you would use in accuracy coding. You then
    have to provide another column (referred to as stim_col) which
    contains information about which the correct response was.

    HDDMnnStimCoding distinguishes itself from the HDDMStimCoding class by allowing you
    to specify a variety of generative models. Likelihoods are based on Neural Networks.

    :Arguments:
        model: str <default='ddm>
            String that determines which model you would like to fit your data to.
            Currently available models are: 'ddm', 'full_ddm', 'angle', 'weibull', 'ornstein', 'levy'
        
        network_type: str <default='mlp>
            String that defines which kind of network to use for the likelihoods. There are currently two 
            options: 'mlp', 'cnn'. CNNs should be treated as experimental at this point.

        nbin: int <default=512>
            Relevant only if network type was chosen to be 'cnn'. CNNs can be trained on coarser or
            finer binnings of RT space. At this moment only networks with 512 bins are available.

        include: list <default=None>
            A list with parameters we wish to include in the fitting procedure. Generally, per default included
            in fitting are the drift parameter 'v', the boundary separation parameter 'a' and the non-decision-time 't'. 
            Which parameters you can include depends on the model you specified under the model parameters.

        split_param : {'v', 'z'} <default='z'>
            There are two ways to model stimulus coding in the case where both stimuli
            have equal information (so that there can be no difference in drift):
            * 'z': Use z for stimulus A and 1-z for stimulus B
            * 'v': Use drift v for stimulus A and -v for stimulus B

        stim_col : str
            Column name for extracting the stimuli to use for splitting.

        drift_criterion : bool <default=False>
            Whether to estimate a constant factor added to the drift-rate.
            Requires split_param='v' to be set.

    """
    def __init__(self, *args, **kwargs):
        kwargs['nn'] = True
        self.network_type = kwargs.pop('network_type', 'mlp')
        self.network = None
        self.stim_col = kwargs.pop('stim_col', 'stim')
        self.split_param = kwargs.pop('split_param', 'z')
        self.drift_criterion = kwargs.pop('drift_criterion', False)
        self.model = kwargs.pop('model', 'ddm')
        self.w_outlier = kwargs.pop('w_outlier', 0.1)

        # AF-TODO: MAYBE NOT NECESSARY ?
        self.is_informative = kwargs.pop('informative', False)

        self.nbin = kwargs.pop('nbin', 512)

        if self.nbin == 512:
            self.cnn_pdf_multiplier = 51.2
        elif self.nbin == 256:
            self.cnn_pdf_multiplier = 25.6
        
        #print(kwargs['include'])
        # Attach likelihood corresponding to model
        if self.network_type == 'mlp':
            self.network = load_mlp(model = self.model)
            network_dict = {'network': self.network}
            self.wfpt_nn = hddm.likelihoods_mlp.make_mlp_likelihood_complete(model = self.model, **network_dict)
    
        if self.network_type == 'cnn':
            self.network = load_cnn(model = self.model, nbin=self.nbin)
            network_dict = {'network': self.network}
            self.wfpt_nn = hddm.likelihoods_cnn.make_cnn_likelihood(model = self.model, pdf_multiplier = self.cnn_pdf_multiplier, **network_dict)
        
        if self.split_param == 'z':
            assert not self.drift_criterion, "Setting drift_criterion requires split_param='v'."
            #print("Setting model to be non-informative")
            kwargs['informative'] = False

            # Add z if it is split parameter but not included in 'include'
            if 'include' in kwargs and 'z' not in kwargs['include']:
                kwargs['include'].append('z')
            else:
                #print('passing through here...')
                if 'include' not in kwargs:
                    kwargs['include'] = ['z']
                else:
                    pass

            #print("Adding z to includes.")

        # Get unique stimulus values for the stimcoding relevant column (has to be of length 2!)
        self.stims = np.asarray(np.sort(np.unique(args[0][self.stim_col])))
        assert len(self.stims) == 2, "%s must contain two stimulus types" % self.stim_col

        super(HDDMnnStimCoding, self).__init__(*args, **kwargs)
        #print(self.p_outlier)

    def _create_stochastic_knodes(self, include):
        knodes = super(HDDMnnStimCoding, self)._create_stochastic_knodes(include)
        #print('knodes')
        #print(knodes)
        if self.drift_criterion:
            knodes.update(self._create_family_normal_normal_hnormal('dc',
                                                                     value = 0,
                                                                     g_mu = 0,
                                                                     g_tau = 3**-2,
                                                                     std_std = 2))
        return knodes

    def _create_wfpt_parents_dict(self, knodes):
        wfpt_parents = super(HDDMnnStimCoding, self)._create_wfpt_parents_dict(knodes)
        
        # SPECIFIC TO STIMCODING
        if self.drift_criterion: 
            wfpt_parents['dc'] = knodes['dc_bottom']

        #print('wfpt parents: ')
        #print(wfpt_parents)
        return wfpt_parents

    def _create_wfpt_knode(self, knodes):
        
        wfpt_parents = self._create_wfpt_parents_dict(knodes)
        #print('wfpt parents in _create_wfpt_knode', wfpt_parents)
        # Here we use a special Knode (see below) that either inverts v or z
        # depending on what the correct stimulus was for that trial type.
        
        return KnodeWfptStimCoding(self.wfpt_nn, 
                                   'wfpt', # TD: ADD wfpt class we need
                                   observed = True, 
                                   col_name = ['rt', 'response'], # Note: This is different from vanilla stimcoding class where it was set to col_name = 'rt',
                                   depends = [self.stim_col],
                                   split_param = self.split_param,
                                   stims = self.stims,
                                   stim_col = self.stim_col,
                                   **wfpt_parents)
class KnodeWfptStimCoding(Knode):
    def __init__(self, *args, **kwargs):
        self.split_param = kwargs.pop('split_param')
        self.stims = kwargs.pop('stims')
        self.stim_col = kwargs.pop('stim_col')

        super(KnodeWfptStimCoding, self).__init__(*args, **kwargs)

    def create_node(self, name, kwargs, data):
        
        # the addition of "depends=['stim']" in the call of
        # KnodeWfptInvZ in HDDMStimCoding makes that data are
        # submitted splitted by the values of the variable stim the
        # following lines check if the variable stim is equal to the
        # value of stim for which z' = 1-z and transforms z if this is
        # the case (similar to v)

        dc = kwargs.pop('dc', None)
        
        # Data supplied here is split so that stim_col has only one value !
        if all(data[self.stim_col] == self.stims[1]): # AF NOTE: Reversed this, previously self.stims[0], compare what is expected as data to my simulator...
            # 
            if self.split_param == 'z':
                #print('printing kwargs from create_node')
                #print(kwargs)
                kwargs['z'] = 1 - kwargs['z']
            elif self.split_param == 'v' and dc is None:
                kwargs['v'] = - kwargs['v']
            elif self.split_param == 'v' and dc != 0:
                kwargs['v'] = - kwargs['v'] + dc # 
            else:
                raise ValueError('split_var must be either v or z, but is %s' % self.split_var)

            return self.pymc_node(name, **kwargs)
        else:
            if dc is not None:
                kwargs['v'] = kwargs['v'] + dc

            return self.pymc_node(name, **kwargs)