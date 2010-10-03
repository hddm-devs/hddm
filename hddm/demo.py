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

import enthought.traits.ui

try:
    from IPython.Debugger import Tracer; debug_here = Tracer()
except:
    pass

import hddm


class DDM(HasTraits):
    """Drift diffusion model"""
    # Paremeters
    z_bias = Range(0,10.,1.)
    z = Property(Float, depends_on=['a','z_bias','no_bias'])
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
    
    histo_lower = Property(Array, depends_on=['params'])
    histo_upper = Property(Array, depends_on=['params'])

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
    no_bias = Bool(True)
    view = View('z_bias', 'sz', 'v', 'sv', 'ter', 'ster', 'a', 'steps', 'num_samples', 'no_bias', 'iter_plot')

    def _get_dt(self):
        return self.steps / self.T

    def _get_z(self):
        if self.no_bias:
            return self.a/2.
        else:
            return self.z_bias

    def _get_params_dict(self):
        return {'v':self.v, 'sv':self.sv, 'z':self.z, 'sz':self.sz, 'ter':self.ter, 'ster':self.ster, 'a':self.a}

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
    def _get_histo_upper(self):
        rts = self.rts[self.rts>0]
        return np.histogram(rts, bins=self.bins, range=(0,self.steps))[0]

    @cached_property
    def _get_histo_lower(self):
        rts = -self.rts[self.rts<0]
        return np.histogram(rts, bins=self.bins, range=(0,self.steps))[0]
    
    def _get_params(self):
        return np.array([self.a, self.v, self.ter, self.sz, self.sv, self.ster])

    
class DDMPlot(HasTraits):
    from MPLTraits import MPLFigureEditor
    
    figure = Instance(Figure, ())
    ddm = Instance(DDM, ())
    plot_histogram = Bool(False)
    plot_simple = Bool(True)
    plot_full_avg = Bool(False)
    plot_lba = Bool(True)
    plot_drifts = Bool(False)
    plot_data = Bool(False)
    
    x_analytical = Property(Array)

    full_avg = Property(Array)
    simple = Property(Array)
    lba = Property(Array)


    data = Array() # Can be set externally
    external_params = Dict()

    go = Button('Go')
    
    view = View(Item('figure', editor=MPLFigureEditor(),
		     show_label=False),
		Item('ddm'),
                Item('plot_histogram'),
                Item('plot_drifts'),
                Item('plot_simple'),
                Item('plot_full_avg'),
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
        return np.linspace(-time, time, 2000)
    
    def _get_full_avg(self):
        return hddm.likelihoods.wiener_like_full_avg(t=self.x_analytical,
                                                     v=self.ddm.v,
                                                     sv=self.ddm.sv,
                                                     z=self.ddm.z,
                                                     sz=self.ddm.sz,
                                                     ter=self.ddm.ter,
                                                     ster=self.ddm.ster,
                                                     a=self.ddm.a, err=.0001, reps=100)

    def _get_simple(self):
        return hddm.wfpt.pdf_array(x=self.x_analytical,
                                   a=self.ddm.a,
                                   z=self.ddm.z,
                                   v=self.ddm.v,
                                   ter=self.ddm.ter, err=.000001)

    def _get_lba(self):
        return hddm.likelihoods.LBA_like(self.x_analytical,
                                        a=self.ddm.a,
                                        z=self.ddm.z_bias,
                                        v0=self.ddm.v, 
					v1=self.ddm.sz,
                                        ter=self.ddm.ter,
                                        sv=self.ddm.sv, logp=False)

    def _go_fired(self):
        self.update_plot()

    def plot_analytical(self, y, color):
        time = self.ddm.steps/self.ddm.dt
        upper = self.x_analytical > 0
        lower = self.x_analytical < 0

        x = np.linspace(0, time, 1000)

        self.figure.axes[0].plot(x, y[upper], color=color, lw=2.)
        self.figure.axes[2].plot(x, -y[lower][::-1], color=color, lw=2.)

    @on_trait_change('ddm.params')
    def update_plot(self):
        if self.figure.canvas is None:
            return
        
	# Clear all plots
	for i in range(self.num_axes):
	    self.figure.axes[i].clear()

        x = np.linspace(0,self.ddm.steps/self.ddm.dt,self.ddm.bins)

        # Plot normalized histograms of simulated data
        if self.plot_histogram:
            upper_scaled, lower_scaled = hddm.utils.scale_multi(self.ddm.histo_upper, self.ddm.histo_lower)
            self.figure.axes[0].plot(x, upper_scaled, color='g', lw=2.)
            self.figure.axes[2].plot(x, -lower_scaled, color='g', lw=2.)

        # Plot normalized histograms of empirical data
        if self.plot_data:
            histo_upper = np.histogram(self.data[self.data > 0], bins=self.ddm.bins, range=(0,self.ddm.steps/self.ddm.dt))[0]
            histo_lower = np.histogram(-self.data[self.data < 0], bins=self.ddm.bins, range=(0,self.ddm.steps/self.ddm.dt))[0]
            upper_scaled, lower_scaled = hddm.utils.scale_multi(histo_upper, histo_lower)
            self.figure.axes[0].plot(x, upper_scaled, color='y', lw=2.)
            self.figure.axes[2].plot(x, -lower_scaled, color='y', lw=2.)

        # Plot analyitical full averaged likelihood function
        if self.plot_full_avg:
            y_avg_scaled = hddm.utils.scale(self.full_avg)
            self.plot_analytical(y_scaled, color='r')

        # Plot analytical simple likelihood function
        if self.plot_simple:
            y_scaled = hddm.utils.scale(self.simple)
            self.plot_analytical(y_scaled, color='b')

        if self.plot_lba:
            y_scaled = hddm.utils.scale(self.lba)
            self.plot_analytical(y_scaled, color='k')

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
        self.figure.axes[0].set_xlim((0,self.ddm.T))
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
