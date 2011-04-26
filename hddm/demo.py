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
import scipy as sp
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
    sz = Range(0,1.,.0)
    v = Range(-3,3.,-2.)
    sv = Range(0.0,2.,0.0)
    ter = Range(0,2.,.3)
    ster = Range(0,2.,.0)
    a = Range(0.,10.,2.)
    switch = Bool(True)
    t_switch = Range(0,2.,.9)
    v_switch = Range(-3.,3.,1.)
    intra_sv = Range(0.,10.,1.)
    urgency = Range(.1,10.,1.)
    
    params = Property(Array, depends_on=['z', 'sz', 'v', 'sv', 'ter', 'ster', 'a', 'switch', 't_switch', 'v_switch', 'intra_sv', 'urgency'])

    # Distributions
    drifts = Property(Tuple, depends_on=['params'])
    rts = Property(Tuple, depends_on=['drifts'])
        
    params_dict = Property(Dict)

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
    view = View('z', 'sz', 'v', 'sv', 'ter', 'ster', 'a', 't_switch', 'v_switch', 'intra_sv', 'urgency', 'steps', 'num_samples', 'iter_plot', 'switch')

    def _get_dt(self):
        return self.steps / self.T

    def _get_params_dict(self):
        d = {'v':self.v, 'V':self.sv, 'z':self.z, 'Z':self.sz, 't':self.ter, 'T':self.ster, 'a':self.a}
        if self.switch:
            d['v_switch'] = self.v_switch
            d['t_switch'] = self.t_switch
        return d

    @cached_property
    def _get_drifts(self):
        return hddm.generate.simulate_drifts(self.params_dict, self.num_samples, self.steps, self.T, intra_sv=self.intra_sv)

    @cached_property
    def _get_rts(self):
        return hddm.generate.find_thresholds(self.drifts, self.a)

    @cached_property
    def _get_rts(self):
        return hddm.generate.find_thresholds(self.drifts, self.a)
    
    @cached_property
    def _get_histo(self):
        return hddm.utils.histogram(self.rts/self.dt, bins=2*self.bins, range=(-self.T,self.T), density=True)[0]
    
    def _get_params(self):
        return np.array([self.a, self.v, self.ter, self.sz, self.sv, self.ster])

    
class DDMPlot(HasTraits):
    from MPLTraits import MPLFigureEditor
    
    figure = Instance(Figure, ())
    ddm = Instance(DDM, ())
    plot_histogram = Bool(False)
    plot_simple = Bool(True)
    plot_full_mc = Bool(False)
    plot_full_interp = Bool(True)
    plot_lba = Bool(False)
    plot_drifts = Bool(False)
    plot_data = Bool(False)
    plot_density = Bool(True)
    plot_density_dist = Bool(False)
    plot_mean_rt = Bool(False)
    plot_switch = Bool(True)
    
    x_analytical = Property(Array)
    
    full_mc = Property(Array)
    full_intrp = Property(Array)
    simple = Property(Array)
    lba = Property(Array)
    switch = Property(Array)

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
                Item('plot_full_interp'),
                Item('plot_lba'),
                Item('plot_data'),
                Item('plot_density'),
#                Item('plot_density_dist'),
                Item('plot_mean_rt'),
		Item('go'),
		#style='custom',
		width=800,
		height=600,
                resizable=True)
    
    def __init__(self, data=None, params=None):
	super(DDMPlot, self).__init__()
        plt.hot()

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
                                             a=self.ddm.a, err=.0001, reps=100)

    @timer
    def _get_simple(self):
        pdf = hddm.wfpt.pdf_array(self.x_analytical,
                                  self.ddm.v,
                                  self.ddm.a,
                                  self.ddm.z,
                                  self.ddm.ter, .001)
        return pdf

    @timer
    def _get_switch(self):
        switch_pdf = lambda x: hddm.wfpt.switch_pdf(x,
                                                    1.,
                                                    self.ddm.v,
                                                    self.ddm.v_switch,
                                                    self.ddm.a,
                                                    self.ddm.z,
                                                    self.ddm.ter,
                                                    self.ddm.t_switch,
                                                    1e-4)
        pdf = np.array(map(switch_pdf, self.x_analytical))
        return pdf
    
    @timer
    def _get_full_intrp(self):
        full_pdf = lambda x: hddm.wfpt.full_pdf(x,
                                                v=self.ddm.v,
                                                V=self.ddm.sv,
                                                a=self.ddm.a,
                                                z=self.ddm.z,
                                                Z=self.ddm.sz,
                                                t=self.ddm.ter,
                                                T=self.ddm.ster,
                                                err=1e-4)

        pdf = np.array(map(full_pdf, self.x_analytical))
        
        return pdf


    @timer
    def _get_lba(self):
        return hddm.likelihoods.LBA_like(self.x_analytical,
                                         a=self.ddm.a,
                                         z=self.ddm.z,
                                         v0=self.ddm.v, 
                                         v1=self.ddm.sz,
                                         t=self.ddm.ter,
                                         V=self.ddm.sv, logp=False)

    def _go_fired(self):
        self.update_plot()

    def plot_histo(self, x, y, color, max_perc=None):
        # y consists of lower and upper boundary responses
        # mid point tells us where to split
        assert y.shape[0]%2==0, "x_analytical has to be even. Shape is %s "%str(y.shape)
        mid = y.shape[0]/2
        self.figure.axes[0].plot(x, y[mid:], color=color, lw=2.)
        # Compute correct EV
        mean_correct_rt = np.sum(x*y[mid:])/np.sum(y[mid:])
        # [::-1] -> reverse ordering
        self.figure.axes[2].plot(x, -y[:mid][::-1], color=color, lw=2.)
        # Compute error EV
        mean_error_rt = np.sum(x*y[:mid][::-1])/np.sum(y[:mid][::-1])

        if self.plot_mean_rt:
            self.figure.axes[0].axvline(mean_correct_rt, color=color)
            self.figure.axes[2].axvline(mean_error_rt, color=color)
        
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
            self.plot_histo(x, self.ddm.histo, color='g', max_perc=.99)

        # Plot normalized histograms of empirical data
        if self.plot_data:
            range_ = self.ddm.steps/self.ddm.dt
            histo = hddm.utils.histogram(self.data, bins=2*self.ddm.bins, range=(-range_,range_), dens=True)[0]
            self.plot_histo(x, histo, color='y')

        # Plot analyitical full averaged likelihood function
        if self.plot_full_mc:
            self.plot_histo(x_anal, self.full_mc, color='r')

        # Plot analytical simple likelihood function
        if self.plot_simple:
            self.plot_histo(x_anal, self.simple, color='b')

        # Plot analytical simple likelihood function
        if self.plot_switch:
            self.plot_histo(x_anal, self.switch, color='r')
            
        if self.plot_lba:
            self.plot_histo(x_anal, self.lba, color='k')

        if self.plot_full_interp:
            self.plot_histo(x_anal, self.full_intrp, color='y')

        if self.plot_density_dist:
            t = np.linspace(0.0, self.ddm.steps/self.ddm.dt, self.ddm.steps)
            dens_upper = sp.stats.norm.pdf(self.ddm.a, loc=((t-self.ddm.ter)*self.ddm.v+(self.ddm.a*self.ddm.z)), scale=((t-self.ddm.ter)*self.ddm.intra_sv + self.ddm.sz)**self.ddm.urgency)
            dens_lower = sp.stats.norm.pdf(0., loc=((t-self.ddm.ter)*self.ddm.v+(self.ddm.a*self.ddm.z)), scale=((t-self.ddm.ter)*self.ddm.intra_sv + self.ddm.sz)**self.ddm.urgency)
            self.plot_histo(t, np.concatenate((dens_lower[::-1], dens_upper)), color='k')
                
        if self.plot_density:
            t = np.linspace(0.0, self.ddm.steps/self.ddm.dt, self.ddm.steps)
            x,y = np.meshgrid(t, np.linspace(0, self.ddm.a, 100))
            # Compute normal density
            t_rel = (t-self.ddm.ter)
            t_rel[t_rel<0] = 0 # Set t smaller than ter to 0
            dens = sp.stats.norm.pdf(y, loc=((t-self.ddm.ter)*self.ddm.v+(self.ddm.z*self.ddm.a)), scale=(np.sqrt(t_rel)*self.ddm.intra_sv + self.ddm.sz)**self.ddm.urgency)
            ## Normalize density
            dens_max = np.max(dens, axis=0)
            dens_max[dens_max == 0] = 1.
            dens_norm = dens / dens_max
            self.figure.axes[1].contourf(x,y,dens_norm)

        if self.plot_drifts:
            t = np.linspace(0.0, self.ddm.steps/self.ddm.dt, self.ddm.steps)
            for k in range(self.ddm.iter_plot):
                reached_thresh = np.where((self.ddm.drifts[k] > self.ddm.a) | (self.ddm.drifts[k] < 0))[0]
                if reached_thresh.shape == (0,):
                    duration = -1
                else:
                    duration = reached_thresh[0]
                self.figure.axes[1].plot(t[:duration],
                                         self.ddm.drifts[k][:duration], 'b', alpha=.5)
    
        self.set_figure()

	wx.CallAfter(self.figure.canvas.draw)

    def set_figure(self):
        # Set axes limits
        # TODO: Fix boundary now that we are using true densities
        total_max = 1.
        if len(self.figure.axes[0].get_lines()) != 0:
            for line in self.figure.axes[0].get_lines():
                max_point = np.max(line.get_ydata())
                if max_point > total_max:
                    total_max = max_point
            for line in self.figure.axes[2].get_lines():
                max_point = np.abs(np.min(line.get_ydata()))
                if max_point > total_max:
                    total_max = max_point
        print total_max
        self.figure.axes[0].set_ylim((0,total_max))
        self.figure.axes[2].set_ylim((-total_max, 0))
        self.figure.axes[0].set_xlim((0, self.ddm.T))
        self.figure.axes[2].set_xlim((0, self.ddm.T))
        self.figure.axes[1].set_ylim((0, self.ddm.a))

        self.figure.subplots_adjust(hspace=0)
        for ax in self.ax1, self.ax2:
            plt.setp(ax.get_xticklabels(), visible=False)

        #for ax in self.ax3:
        #    plt.setp(ax.get_yticklabels(), visible=False)
        self.ax3.set_xlabel('time (secs)')

if __name__ == "__main__":
    ddm = DDMPlot()
    ddm.configure_traits()
