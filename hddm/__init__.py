#!/usr/bin/python
from __future__ import division

try:
    import demo
except ImportError, err:
    print "WARNING: demo.py could not be imported, disabling plot_demo(). Reason: " + str(err)

import numpy as np
import numpy.lib.recfunctions as rec
import pymc as pm
import matplotlib.pyplot as plt
from copy import copy
import subprocess

import model
import likelihoods
import generate
import utils
try:
    import mpi
except ImportError:
    def mpi():
        raise ImportError, "Can't import mpi"

from model import *
from utils import plot_posteriors, plot_post_pred

try:
    import wfpt
    import wfpt_full
    import wfpt_switch
except:
    import wfpt32 as wfpt
    import wfpt_full32 as wfpt_full
    import wfpt_switch32 as wfpt_switch

#import lba

from kabuki.utils import load_csv, save_csv
try:
    from IPython.core.debugger import Tracer; debug_here = Tracer()
except:
    def debug_here(): pass
