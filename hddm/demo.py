"""
brownian() implements one dimensional Brownian motion (i.e. the Wiener process).
"""
import wx

import matplotlib
# We want matplotlib to use a wxPython backend
if __name__ == "__main__":
    matplotlib.use('WXAgg')

import matplotlib.pyplot as plt

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from enthought.traits.api import HasTraits, Instance, Range,\
                                Array, on_trait_change, Property,\
                                cached_property, Bool, Button, Tuple,\
                                Any, Int, Str, Float, Delegate, Dict
from enthought.traits.ui.view import View, Item
from enthought.traits.ui.wx.editor import Editor
from enthought.traits.ui.basic_editor_factory import BasicEditorFactory

from math import sqrt

import numpy as np
import pylab as pl
import time

import enthought.traits.ui

try:
    from IPython.Debugger import Tracer; debug_here = Tracer()
except:
    pass

import hddm

def timer(method):
    def time_me(self, *args,**kwargs):
        t0 = time.clock()
        res = method(self,*args,**kwargs)
        print "%s took %f secs" % (method.__name__, time.clock() - t0)
        return res
    
    return time_me
    
class DDM(HasTraits):
    """Drift diffusion model"""
    # Paremeters
    z = Range(0,1.,.5)
    sz = Range(0,5.,.1)
    v = Range(-3,3.,.5)
    sv = Range(0.0,2.,0.01)
    ter = Range(0,2.,.3)
    ster = Range(0,2.,.1)
    a = Range(0.,10.,2.)

    params = Property(Array, depends_on=['z', 'sz', 'v', 'sv', 'ter', 'ster', 'a'])

    # Distributions
    drifts = Property(Tuple, depends_on=['params'])
    rts = Property(Tuple, depends_on=['drifts'])
    params_dict = Property(Dict)

    mask = Property(Array)
    
    histo = Property(Array, depends_on=['params'])

    # Total time.
    T = Float(5.0)
    # Number of steps.
    steps = Int(1000)
    # Time step size
    dt = Property(Float, depends_on=['T', 'steps'])
    # Number of realizations to generate.
    num_samples = Int(1000)
    # Number of drifts to plot
    iter_plot = Int(30)
    # Number of histogram bins
    bins = Int(100)
    view = View('z', 'sz', 'v', 'sv', 'ter', 'ster', 'a', 'steps', 'num_samples', 'iter_plot')

    def _get_dt(self):
        return self.steps / self.T

    def _get_params_dict(self):
        return {'v':self.v, 'V':self.sv, 'z':self.z, 'Z':self.sz, 't':self.ter, 'T':self.ster, 'a':self.a}

    @cached_property
    def _get_drifts(self):
        return hddm.generate.simulate_drifts(self.params_dict, self.num_samples, self.steps, self.T)

    @cached_property
    def _get_rts(self):
        return hddm.generate.find_thresholds(self.drifts, self.a)

    @cached_property
    def _get_mask(self):
        return hddm.generate.gen_mask(self.drifts, self.a)
    
    @cached_property
    def _get_histo(self):
        return np.histogram(self.rts, bins=2*self.bins, range=(-self.steps,self.steps))[0]
    
    def _get_params(self):
        return np.array([self.a, self.v, self.ter, self.sz, self.sv, self.ster])

    
class DDMPlot(HasTraits):
    from MPLTraits import MPLFigureEditor
    
    figure = Instance(Figure, ())
    ddm = Instance(DDM, ())
    plot_histogram = Bool(False)
    plot_simple = Bool(True)
    plot_full_mc = Bool(False)
    plot_lba = Bool(True)
    plot_drifts = Bool(False)
    plot_data = Bool(False)
    
    x_analytical = Property(Array)
    
    full_mc = Property(Array)
    simple = Property(Array)
    lba = Property(Array)

    x_raster = Int(100)

    data = Array() # Can be set externally
    external_params = Dict()

    go = Button('Go')
    
    view = View(Item('figure', editor=MPLFigureEditor(),
		     show_label=False),
		Item('ddm'),
                Item('plot_histogram'),
                Item('plot_drifts'),
                Item('plot_simple'),
                Item('plot_full_mc'),
                Item('plot_lba'),
                Item('plot_data'),
		Item('go'),
		#style='custom',
		width=800,
		height=600,
                resizable=True)
    
    def __init__(self, data=None, params=None):
	super(DDMPlot, self).__init__()

        # Create plot    
        self.num_axes = 3
	self.ax1 = self.figure.add_subplot(411)
	self.ax2 = self.figure.add_subplot(412, sharex=self.ax1)
	self.ax3 = self.figure.add_subplot(413, sharex=self.ax1)

        self.set_figure()
        #self.update_plot()

        # Set external data if necessary
        if data is not None:
            self.data = data
        if params is not None:
            self._set_external_params(params)

    def _external_params_changed(self, new):
        try:
            self.ddm.a = new['a']
        except KeyError:
            pass
        try:
            self.ddm.z_bias = new['z']
        except KeyError:
            pass
        try:
            self.ddm.sv = new['V']
        except KeyError:
            pass
        try:
            self.ddm.ter = new['t']
        except KeyError:
            pass
        try:
            self.ddm.ster = new['T']
        except KeyError:
            pass
        try:
            self.ddm.sz = new['Z']
        except KeyError:
            pass
        try:
            self.ddm.v = new['v']
        except KeyError:
            pass
        try:
            self.ddm.v = new['v0']
        except KeyError:
            pass
        try:
            self.ddm.sz = new['v1']
        except KeyError:
            pass
            
    def _data_changed(self):
        self.plot_data = True
        
    @cached_property
    def _get_x_analytical(self):
        time = self.ddm.steps/self.ddm.dt
        return np.linspace(-time, time, self.x_raster)

    @timer
    def _get_full_mc(self):
        return hddm.wfpt.wiener_like_full_mc(x=self.x_analytical,
                                             v=self.ddm.v,
                                             V=self.ddm.sv,
                                             z=self.ddm.z,
                                             Z=self.ddm.sz,
                                             t=self.ddm.ter,
                                             T=self.ddm.ster,
                                             a=self.ddm.a, err=.0001, reps=50)

    @timer
    def _get_simple(self):
        return hddm.wfpt.pdf_array(x=self.x_analytical,
                                   a=self.ddm.a,
                                   z=self.ddm.z,
                                   v=self.ddm.v,
                                   t=self.ddm.ter, err=.000001)

    @timer
    def _get_lba(self):
        return hddm.likelihoods.LBA_like(self.x_analytical,
                                         a=self.ddm.a,
                                         z=self.ddm.z_bias,
                                         v0=self.ddm.v, 
                                         v1=self.ddm.sz,
                                         t=self.ddm.ter,
                                         V=self.ddm.sv, logp=False)

    def _go_fired(self):
        self.update_plot()

    def plot_histo(self, x, y, color):
        y_scaled = hddm.utils.scale(y)
        # y consists of lower and upper boundary responses
        # mid point tells us where to split
        assert y.shape[0]%2==0, "x_analytical has to be even. Shape is %s "%str(y.shape)
        mid = y.shape[0]/2
        self.figure.axes[0].plot(x, y_scaled[mid:], color=color, lw=2.)
        # [::-1] -> reverse ordering
        self.figure.axes[2].plot(x, -y_scaled[:mid][::-1], color=color, lw=2.)

    @on_trait_change('ddm.params')
    def update_plot(self):
        if self.figure.canvas is None:
            return
        
	# Clear all plots
	for i in range(self.num_axes):
	    self.figure.axes[i].clear()

        # Set x axis values 
        x = np.linspace(0,self.ddm.steps/self.ddm.dt,self.ddm.bins)
        # Set x axis values for analyticals
        time = self.ddm.steps/self.ddm.dt
        x_anal = np.linspace(0, time, self.x_raster/2)

        # Plot normalized histograms of simulated data
        if self.plot_histogram:
            self.plot_histo(x, self.ddm.histo, color='g')

        # Plot normalized histograms of empirical data
        if self.plot_data:
            range_ = self.ddm.steps/self.ddm.dt
            histo = np.histogram(self.data, bins=2*self.ddm.bins, range=(-range_,range_))[0]
            self.plot_histo(x, histo, color='y')

        # Plot analyitical full averaged likelihood function
        if self.plot_full_mc:
            self.plot_histo(x_anal, self.full_mc, color='r')

        # Plot analytical simple likelihood function
        if self.plot_simple:
            self.plot_histo(x_anal, self.simple, color='b')

        if self.plot_lba:
            y_scaled = hddm.utils.scale(self.lba)
            self.plot_histo(x_anal, self.lba, color='k')

        if self.plot_drifts:
            t = np.linspace(0.0, self.ddm.steps/self.ddm.dt, self.ddm.steps)
            for k in range(self.ddm.iter_plot):
                self.figure.axes[1].plot(t[self.ddm.mask[k]],
                                         self.ddm.drifts[k][self.ddm.mask[k]], 'b')
	# Draw boundaires
	#self.figure.axes[1].plot(t, np.ones(t.shape)*self.ddm.a, 'k')
	#self.figure.axes[1].plot(t, np.zeros(t.shape), 'k')
        #self.figure.axes[1].set_ylim((0,10))
        
        self.set_figure()

	wx.CallAfter(self.figure.canvas.draw)

    def set_figure(self):
        # Set axes limits
        self.figure.axes[0].set_ylim((0,1.05))
        self.figure.axes[2].set_ylim((-1.05, 0))
        self.figure.axes[0].set_xlim((0, self.ddm.T))
        self.figure.axes[2].set_xlim((0, self.ddm.T))
        self.figure.axes[1].set_ylim((0, self.ddm.a))

        self.figure.subplots_adjust(hspace=0)
        for ax in self.ax1, self.ax2:
            plt.setp(ax.get_xticklabels(), visible=False)

        for ax in self.ax1, self.ax3:
            plt.setp(ax.get_yticklabels(), visible=False)
        self.ax3.set_xlabel('time (secs)')

if __name__ == "__main__":
    ddm = DDMPlot()
    ddm.configure_traits()
