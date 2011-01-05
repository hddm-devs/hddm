#!/usr/bin/python
from __future__ import division

try:
    import demo
except ImportError, err:
    print "Demo applet could not be imported: " + str(err)

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
import generate
import utils

from models import *

import wfpt
import lba

try:
    from IPython.Debugger import Tracer; debug_here = Tracer()
except:
    def debug_here(): pass
