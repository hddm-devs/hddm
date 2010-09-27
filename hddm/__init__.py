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

import models
import likelihoods

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
