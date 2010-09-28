#!/usr/bin/python
from __future__ import division
import wx

import matplotlib
# We want matplotlib to use a wxPython backend
if __name__ == "__main__":
    matplotlib.use('WXAgg')

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
import demo
import generate

from models import *

try:
    from IPython.Debugger import Tracer; debug_here = Tracer()
except:
    def debug_here(): pass

if platform.architecture()[0] == '64bit':
    import wfpt64 as wfpt
    import lba64 as lba
    sampler_exec = 'construct-samples64'
else:
    import wfpt
    import lba
    sampler_exec = 'construct-samples'
