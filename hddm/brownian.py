"""
brownian() implements one dimensional Brownian motion (i.e. the Wiener process).
"""
from __future__ import division
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
from scipy.stats import uniform, norm
import numpy as np
import pylab as pl

import enthought.traits.ui

try:
    from IPython.Debugger import Tracer; debug_here = Tracer()
except:
    pass

from fastdm_cdf import cdf
import wfpt
import ddm_likelihoods


def scale(x, max=None, min=None):
    #x = np.array(x, dtype=np.float)
    if max is None:
        max = np.max(x)
    if min is None:
        min = np.min(x)
    return (x-min)/(max-min)

def scale_multi(a1, a2):
    """Scale two arrays to be in range [0,1].
    """
    # Scale appropriately
    if np.max(a1) > np.max(a2):
        a1_scaled = scale(a1)
        a2_scaled = scale(a2, max=np.max(a1), min=np.min(a1))
    else:
        a2_scaled = scale(a2)
        a1_scaled = scale(a1, max=np.max(a2), min=np.min(a2))

    return (a1_scaled, a2_scaled)

class DDM(HasTraits):
    """Drift diffusion model"""
    # Paremeters
    z_bias = Range(0,10.,1.)
    z = Property(Float, depends_on=['a','z_bias','no_bias'])
    sz = Range(0,5.,.1)
    v = Range(-3,3.,.5)
    sv = Range(0.0,2.,0.01)
    t0 = Range(0,2.,.3)
    st0 = Range(0,2.,.1)
    a = Range(0.,10.,2.)

    params = Property(Array, depends_on=['z', 'sz', 'v', 'sv', 't0', 'st0', 'a'])

    # Distributions
    start_delays = Property(Array, depends_on=['ter', 'ster'])
    start_points = Property(Array, depends_on=['z', 'sz'])
    drift_rates = Property(Array, depends_on=['v', 'sv'])
    drifts = Property(Array, depends_on=['v', 'sv'])
    randoms = Property(Array)
    
    rts = Property(Tuple, depends_on=['params'])
    rts_upper = Property(Array, depends_on=['params'])
    rts_lower = Property(Array, depends_on=['params'])

    histo_lower = Property(Array, depends_on=['params'])
    histo_upper = Property(Array, depends_on=['params'])

    x_analytical_cdf = Property(Array, depends_on=['N'])

    # Functions to compute the Cummulative Density Function
    analytical_cdf = Property(Array, depends_on=['params', 'N'])
    empirical_cdf = Property(Array, depends_on=['histo_lower', 'histo_upper'])
    
    # Total time.
    T = Float(5.0)
    # Number of steps.
    N = Int(1000)
    # Time step size
    dt = Property(Float, depends_on=['T', 'N'])
    # Number of realizations to generate.
    num_samples = Int(1000)
    # Number of drifts to plot
    iter_plot = Int(30)
    # Number of histogram bins
    bins = Int(100)
    dt_bins = Property(Float, depends_on=['T', 'bins'])
    no_bias = Bool(True)
    view = View('z_bias', 'sz', 'v', 'sv', 't0', 'st0', 'a', 'N', 'num_samples', 'no_bias', 'iter_plot')

    def _get_z(self):
        if self.no_bias:
            return self.a/2.
        else:
            return self.z_bias
        
    @cached_property
    def _get_drifts(self):
	# This computes the Brownian motion by forming the cumulative sum of
	# the random samples. Add the starting points as a constant offset.
	return np.cumsum(self.randoms, axis=1) + np.tile(self.start_points, (self.N, 1)).T

    @cached_property
    def _get_randoms(self):
	# Generate num_samples samples from a normal distribution and
        # add the drift rates
	x = norm.rvs(loc=0, scale=np.sqrt(self.dt), size=(self.num_samples, self.N))/self.dt  + np.tile(self.drift_rates, (self.N, 1)).T
	# Go through every line and zero out the non-decision component
	for i in range(self.num_samples):
	    x[i,:self.start_delays[i]] = 0
	return x

    @cached_property
    def _get_start_delays(self):
        # Draw starting delays from uniform distribution around [t0-0.5*st0, t0+0.5*st0].
        start_delays = np.abs(uniform.rvs(size=(self.num_samples), scale=(self.st0)) - self.st0/2. + self.t0)
        start_delays *= self.dt
	return np.array(start_delays, dtype='int')

    @cached_property
    def _get_start_points(self):
        # Draw starting points from uniform distribution around [z-0.5*sz, z+0.5*sz]
        return uniform.rvs(size=(self.num_samples), scale=(self.sz)) - self.sz/2. + self.z

    @cached_property
    def _get_drift_rates(self):
        # Draw drift rates from a normal distribution
        return norm.rvs(loc=self.v, scale=self.sv, size=self.num_samples)/self.dt
    
    def _get_dt(self):
        return self.N / self.T

    def _get_dt_bins(self):
        return self.bins / self.T

    @cached_property
    def _get_rts(self):
	# Find lines over the threshold
	rts_upper = []
	rts_lower = []
        self.mask = np.ones(self.drifts.shape, dtype='bool')
        # Go through every drift individually
	for line_x, line_mask in zip(self.drifts, self.mask):
            # Check if upper bound was reached, and where
	    thresh_upper = np.where((line_x > self.a))[0]
            upper = np.Inf
            lower = np.Inf
	    if thresh_upper.shape != (0,):
                # This provides the RT, append
                upper = thresh_upper[0]
		
            # Check if lower bound was reached, and where
	    thresh_lower = np.where((line_x < 0))[0]
	    if thresh_lower.shape != (0,):
                # Lower RT
		lower = thresh_lower[0]

            if upper == lower == np.Inf:
                # Threshold not crossed
                continue
            
            # Determine which one hit the threshold before
            if upper < lower:
                rts_upper.append(upper)
            else:
                rts_lower.append(lower)
            # For plotting, we want to cut off the diffusion processes
            # after they reached the threshold. So get all thresholds,
            # and create a mask with Trues if below threshold. Mask is used
            # for displaying only.
	    thresh = np.where((line_x > self.a) | (line_x < 0))[0]

	    if thresh.shape != (0,):
		line_mask[thresh[0]:] = np.zeros((self.N-thresh[0]), dtype='bool')

        return (rts_upper, rts_lower)

    @cached_property
    def _get_rts_upper(self):
        return self.rts[0]
    
    @cached_property
    def _get_rts_lower(self):
        return self.rts[1]
    
    @cached_property
    def _get_histo_upper(self):
        return np.histogram(self.rts_upper, bins=self.bins, range=(0,self.N))[0]

    @cached_property
    def _get_histo_lower(self):
        return np.histogram(self.rts_lower, bins=self.bins, range=(0,self.N))[0]
        
    @cached_property    
    def _get_x_analytical_cdf(self):
        #return np.linspace((-self.N)/self.dt, (self.N)/self.dt, 2*self.N + 1)
	return np.linspace(-self.T, self.T, 2*self.N)
    
    def _get_params(self):
        return np.array([self.a, self.v, self.t0, self.sz, self.sv, self.st0])

    @cached_property
    def _get_analytical_cdf(self):
        """Compute the response time cumulative distribution analytically using the fast-dm algorithm."""
        x = cdf(self.params, z=self.z, N=self.N, time=self.T)
	# Normalize. For some reason, fast-dm treats non-finished trials in a weird way.
	x = (x-x.min())/(x.max()-x.min())
	return x

    def _get_empirical_cdf(self):
        """Compute the response time cumulative distribution of the simulated data."""
        histo = np.concatenate((self.histo_lower[::-1], self.histo_upper))
        cumulative = np.cumsum(histo)
        normalizer = cumulative[-1]
        return np.array(cumulative, dtype='float')/normalizer
        

class CDFPlot(HasTraits):
    from MPLTraits import MPLFigureEditor

    cdf = Instance(DDM, ())
    figure = Instance(Figure, ())
    axes = Instance(Axes)
    line = Instance(Line2D, depends_on=['cdf'])
    
    view = View(Item('figure', editor=MPLFigureEditor(),
		     show_label=False),
                Item('cdf'),
		style='custom',
		width=800,
		height=600,
		resizable=True)

    def __init__(self):
        super(CDFPlot, self).__init__()
        self.axes = self.figure.add_subplot(111)
        self.canvas = self.figure.axes[0].figure.canvas
        self.line = self.axes.plot(self.cdf.x_analytical_cdf, self.cdf.analytical_cdf)[0]

    def _get_axes(self):
        return 

    def _get_line(self):
        return self.axes.plot(self.cdf.x, self.cdf.analytical_cdf)[0]
        
    @on_trait_change('cdf.analytical_cdf')
    def line_change(self):
        #self.figure.axes[0].lines[0].set_ydata(self.cdf.analytical_cdf)
        #self.figure.axes[0].draw_animated()
        self.line.set_ydata(self.cdf.analytical_cdf)
        self.axes.draw_artist(self.line)
	self.axes.set_ylim((0,1))
        wx.CallAfter(self.figure.canvas.draw)
    
class DDMPlot(HasTraits):
    from MPLTraits import MPLFigureEditor
    
    figure = Instance(Figure, ())
    ddm = Instance(DDM, ())
    plot_histogram = Bool(False)
    plot_simple = Bool(True)
    plot_full_avg = Bool(False)
    plot_CDF = Bool(False)
    plot_lba = Bool(True)
    plot_drifts = Bool(False)
    plot_data = Bool(False)
    
    x_analytical = Property(Array)
    full_avg_lower = Property(Array)
    full_avg_upper = Property(Array)
    simple_upper = Property(Array)
    simple_lower = Property(Array)
    lba_upper = Property(Array)
    lba_lower = Property(Array)

    data = Array() # Can be set externally
    external_params = Dict()

    go = Button('Go')
    
    view = View(Item('figure', editor=MPLFigureEditor(),
		     show_label=False),
		Item('ddm'),
                Item('plot_histogram'),
                Item('plot_simple'),
                Item('plot_full_avg'),
                Item('plot_CDF'),
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
        if self.plot_CDF:
            self.num_axes = 4
        else:
            self.num_axes = 3
	self.ax1 = self.figure.add_subplot(411)
	self.ax2 = self.figure.add_subplot(412, sharex=self.ax1)
	self.ax3 = self.figure.add_subplot(413, sharex=self.ax1)

        if self.plot_CDF:
            self.ax4 = self.figure.add_subplot(414)
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
            self.ddm.t0 = new['t']
        except KeyError:
            pass
        try:
            self.ddm.st0 = new['T']
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
        return np.linspace(0, self.ddm.N/self.ddm.dt, 1000)
    
    def _get_full_avg_upper(self):
        return wfpt.wiener_like_full_avg(t=self.x_analytical, v=self.ddm.v, sv=self.ddm.sv, z=self.ddm.z, sz=self.ddm.sz, ter=self.ddm.t0, ster=self.ddm.st0, a=self.ddm.a, err=.0001, reps=100)
    def _get_full_avg_lower(self):
        return wfpt.wiener_like_full_avg(t=-self.x_analytical, v=self.ddm.v, sv=self.ddm.sv, z=self.ddm.z, sz=self.ddm.sz, ter=self.ddm.t0, ster=self.ddm.st0, a=self.ddm.a, err=.0001, reps=100)

    def _get_simple_upper(self):
        return wfpt.pdf_array(x=self.x_analytical, a=self.ddm.a, z=self.ddm.z, v=self.ddm.v, ter=self.ddm.t0, err=.000001)
    def _get_simple_lower(self):
        return wfpt.pdf_array(x=-self.x_analytical, a=self.ddm.a, z=self.ddm.z, v=self.ddm.v, ter=self.ddm.t0, err=.000001)

    def _get_lba_upper(self):
        return ddm_likelihoods.LBA_like(self.x_analytical,
                                        resps=np.ones_like(self.x_analytical),
                                        a=self.ddm.a,
                                        z=self.ddm.z_bias,
                                        v=[self.ddm.v, self.ddm.sz],
                                        ter=self.ddm.t0, sv=self.ddm.sv, logp=False)

    def _get_lba_lower(self):
        return ddm_likelihoods.LBA_like(self.x_analytical,
                                        resps=np.zeros_like(self.x_analytical),
                                        a=self.ddm.a,
                                        z=self.ddm.z_bias,
                                        v=[self.ddm.v, self.ddm.sz],
                                        ter=self.ddm.t0, sv=self.ddm.sv, logp=False)

    def _go_fired(self):
        self.update_plot()
        
    @on_trait_change('ddm.params')
    def update_plot(self):
        if self.figure.canvas is None:
            return
        
	# Clear all plots
	for i in range(self.num_axes):
	    self.figure.axes[i].clear()

        x = np.linspace(0,self.ddm.N/self.ddm.dt,self.ddm.bins)

        # Plot normalized histograms of simulated data
        if self.plot_histogram:
            upper_scaled, lower_scaled = scale_multi(self.ddm.histo_upper, self.ddm.histo_lower)
            self.figure.axes[0].plot(x, upper_scaled, color='g')
            self.figure.axes[2].plot(x, -lower_scaled, color='g')

        # Plot normalized histograms of empirical data
        if self.plot_data:
            histo_upper = np.histogram(self.data[self.data > 0], bins=self.ddm.bins, range=(0,self.ddm.N/self.ddm.dt))[0]
            histo_lower = np.histogram(-self.data[self.data < 0], bins=self.ddm.bins, range=(0,self.ddm.N/self.ddm.dt))[0]
            upper_scaled, lower_scaled = scale_multi(histo_upper, histo_lower)
            self.figure.axes[0].plot(x, upper_scaled, color='y', lw=2.)
            self.figure.axes[2].plot(x, -lower_scaled, color='y', lw=2.)

        # Plot analyitical full averaged likelihood function
        if self.plot_full_avg:
            y_upper_avg_scaled, y_lower_avg_scaled = scale_multi(self.full_avg_upper, self.full_avg_lower)
            self.figure.axes[0].plot(self.x_analytical, y_upper_avg_scaled, color='r', lw=2.)
            self.figure.axes[2].plot(self.x_analytical, -y_lower_avg_scaled, color='r', lw=2.)

        # Plot analytical simple likelihood function
        if self.plot_simple:
            y_upper_scaled, y_lower_scaled = scale_multi(self.simple_upper, self.simple_lower)
            self.figure.axes[0].plot(self.x_analytical, y_upper_scaled, color='b', lw=2.)
            self.figure.axes[2].plot(self.x_analytical, -y_lower_scaled, color='b', lw=2.)

        if self.plot_lba:
            upper_scaled, lower_scaled = scale_multi(self.lba_upper, self.lba_lower)
            self.figure.axes[0].plot(self.x_analytical, upper_scaled, color='k', lw=2.)
            self.figure.axes[2].plot(self.x_analytical, -lower_scaled, color='k', lw=2.)

        if self.plot_drifts:
            t = np.linspace(0.0, self.ddm.N/self.ddm.dt, self.ddm.N)
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
        if self.plot_CDF:
            self.figure.axes[3].plot(np.linspace(-self.ddm.N/self.ddm.dt,self.ddm.N/self.ddm.dt,2*self.ddm.bins), self.ddm.empirical_cdf)
            self.figure.axes[3].plot(self.ddm.x_analytical_cdf, self.ddm.analytical_cdf)
            self.figure.axes[3].set_xlim((-self.ddm.T,self.ddm.T))
            plt.setp(self.ax4.get_yticklabels(), visible=False)

        self.figure.subplots_adjust(hspace=0)
        for ax in self.ax1, self.ax2:
            plt.setp(ax.get_xticklabels(), visible=False)

        for ax in self.ax1, self.ax3:
            plt.setp(ax.get_yticklabels(), visible=False)
        self.ax3.set_xlabel('time (secs)')

if __name__ == "__main__":
    ddm = DDMPlot()
    ddm.configure_traits()
