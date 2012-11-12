from copy import copy
import numpy as np

from kabuki.hierarchical import Knode
from hddm.models import HDDM

class HDDMStimCoding(HDDM):
    """HDDM model that can be used when stimulus coding and estimation
    of bias (i.e. displacement of starting point z) is required.

    In that case, the 'resp' column in your data should contain 0 and
    1 for the chosen stimulus (or direction), not whether the response
    was correct or not as you would use in accuracy coding. You then
    have to provide another column (referred to as stim_col) which
    contains information about which the correct response was.

    :Arguments:
        split_param : str ('v' or 'z')
            There are two ways to model stimulus coding in the case where both stimuli
            have equal information (so that there can be no difference in drift):
            * 'z': Use z for stimulus A and 1-z for stimulus B
            * 'v': Use drift v for stimulus A and -v for stimulus B

        stim_col : str
            Column name for extracting the stimuli to use for splitting.

    """
    def __init__(self, *args, **kwargs):
        self.stim_col = kwargs.pop('stim_col', 'stim')
        self.split_param = kwargs.pop('split_param', 'z')
        if self.split_param == 'z' and 'include' in kwargs:
            if 'z' not in kwargs['include']:
                kwargs['include'].append('z')
                print "Adding z to includes."
        else:
            kwargs['include'] = ['z']
            print "Adding z to includes."
        #assert self.stim_col in self.data.columns, "Can not find column named %s" % self.stim_col
        self.stims = np.sort(np.unique(args[0][self.stim_col]))
        assert len(self.stims) == 2, "%s must contain two stimulus types" % self.stim_col

        super(HDDMStimCoding, self).__init__(*args, **kwargs)


    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)
        # Here we use a special Knode (see below) that either inverts v or z
        # depending on what the correct stimulus was for that trial type.
        return KnodeWfptStimCoding(self.wfpt_class, 'wfpt',
                                   observed=True, col_name='rt',
                                   depends=[self.stim_col],
                                   split_param=self.split_param,
                                   stims=self.stims,
                                   stim_col=self.stim_col,
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
        if all(data[self.stim_col] == self.stims[0]):
            if self.split_param == 'z':
                z = copy(kwargs['z'])
                kwargs['z'] = 1-z
            elif self.split_param == 'v':
                v = copy(kwargs['v'])
                kwargs['v'] = -v
            else:
                raise ValueError('split_var must be either v or z, but is %s' % self.split_var)

            return self.pymc_node(name, **kwargs)
        else:
            return self.pymc_node(name, **kwargs)
