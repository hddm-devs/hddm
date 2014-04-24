#!/usr/bin/env python
"""
brownian() implements one dimensional Brownian motion (i.e. the Wiener process).
"""
import wx

import matplotlib

# We want matplotlib to use a wxPython backend
if __name__ == "__main__":
    matplotlib.use('WXAgg')

import matplotlib.pyplot as plt

from matplotlib.figure import Figure

from traits.api import HasTraits, Instance, Range,\
                                Array, on_trait_change, Property,\
                                cached_property, Bool, Button, Tuple,\
                                Any, Int, Str, Float, Delegate, Dict
from traitsui.view import View, Item

import numpy as np
import time

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
    z = Range(0, 1., .5)
    sz = Range(0, 1., .0)
    v = Range(-4.,4.,.5)
    sv = Range(0.0,2.,0.0)
    ter = Range(0,2.,.3)
    ster = Range(0,2.,.0)
    a = Range(0.,10.,2.)
    switch = Bool(False)
    t_switch = Range(0,2.,.3)
    v_switch = Range(-20.,20.,1.)
    intra_sv = Range(0.1,10.,1.)
    urgency = Range(.1,10.,1.)

    params = Property(Array, depends_on=['z', 'sz', 'v', 'sv', 'ter', 'ster', 'a']) #, 'switch', 't_switch', 'v_switch', 'intra_sv'])

    # Distributions
    drifts = Property(Tuple, depends_on=['params'])
    rts = Property(Tuple, depends_on=['drifts'])

    params_dict = Property(Dict)

    histo = Property(Array, depends_on=['params'])

    # Total time.
    T = Float(5.0)
    # Time step size
    dt = Float(1e-4)
    # Number of realizations to generate.
    num_samples = Int(5000)
    # Number of drifts to plot
    iter_plot = Int(50)
    # Number of histogram bins
    bins = Int(200)
    view = View('z', 'sz', 'v', 'sv', 'ter', 'ster', 'a', 'num_samples', 'iter_plot') #, 'switch', 't_switch', 'v_switch', 'intra_sv', 'T')

    def _get_params_dict(self):
        d = {'v': self.v, 'sv': self.sv, 'z': self.z, 'sz': self.sz, 't': self.ter, 'st': self.ster, 'a': self.a}
        if self.switch:
            d['v_switch'] = self.v_switch
            d['t_switch'] = self.t_switch
            d['V_switch'] = self.sv
        return d

    @cached_property
    def _get_drifts(self):
        return hddm.generate._gen_rts_from_simulated_drift(self.params_dict, samples=self.iter_plot, dt=self.dt, intra_sv=self.intra_sv)[1]

    @cached_property
    def _get_rts(self):
        if not self.switch:
            # Use faster cdf method
            return hddm.generate.gen_rts(size=self.num_samples, range_=(-self.T, self.T), structured=False, **self.params_dict)
        else:
            # Simulate individual drifts
            return hddm.generate._gen_rts_from_simulated_drift(self.params_dict, samples=self.num_samples, dt=self.dt, intra_sv=self.intra_sv)[0]
            #return hddm.generate.gen_rts(self.params_dict, samples=self.num_samples, range_=(-5, 5), method='drift')

    @cached_property
    def _get_histo(self):
        n, bins = np.histogram(self.rts, bins=2 * self.bins, range=(-self.T, self.T))
        db = np.array(np.diff(bins), float)
        return n / db / (n.sum())

    def _get_params(self):
        return np.array([self.a, self.v, self.ter, self.sz, self.sv, self.ster])


class DDMPlot(HasTraits):
    from hddm.MPLTraits import MPLFigureEditor

    figure = Instance(Figure, ())
    parameters = Instance(DDM, ())
    plot_histogram = Bool(True)
    plot_simple = Bool(False)
    plot_wiener = Bool(False)
    plot_lba = Bool(False)
    plot_drifts = Bool(True)
    #plot_data = Bool(False)
    #plot_density = Bool(False)
    plot_true_density = Bool(False)
    #plot_density_dist = Bool(False)
    plot_mean_rt = Bool(False)
    plot_switch = Bool(False)
    color_errors = Bool(True)

    x_analytical = Property(Array)

    wiener = Property(Array)
    simple = Property(Array)
    lba = Property(Array)
    switch = Property(Array)

    x_raster = Int(100)

    data = Array()  # Can be set externally
    external_params = Dict()

    go = Button('Go')

    view = View(Item('figure', editor=MPLFigureEditor(),
                     show_label=False),
                Item('parameters'),
                Item('plot_histogram'),
                Item('plot_drifts'),
                Item('plot_wiener'),
                #Item('plot_lba'),
                #Item('plot_data'),
                #Item('plot_density'),
                #Item('plot_true_density'),
                #Item('plot_density_dist'),
                #Item('plot_switch'),
                Item('color_errors'),
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

    def _data_changed(self):
        self.plot_data = True

    @cached_property
    def _get_x_analytical(self):
        return np.linspace(-self.parameters.T, self.parameters.T, self.x_raster)

    @timer
    def _get_switch(self):
        from hddm.sandbox.model import wfpt_switch_like
        pdf = wfpt_switch_like.rv.pdf(self.x_analytical,
                                      self.parameters.v,
                                      self.parameters.v_switch,
                                      self.parameters.sv,
                                      self.parameters.a, self.parameters.z,
                                      self.parameters.ter,
                                      self.parameters.t_switch,
                                      self.parameters.ster)
        return pdf

    @timer
    def _get_wiener(self):
        pdf = hddm.wfpt.pdf_array(self.x_analytical,
                                  self.parameters.v, self.parameters.sv,
                                  self.parameters.a, self.parameters.z,
                                  self.parameters.sz, self.parameters.ter,
                                  self.parameters.ster, 1e-4, logp=False)
        return pdf

    @timer
    def _get_lba(self):
        return hddm.likelihoods.LBA_like(self.x_analytical,
                                         a=self.parameters.a,
                                         z=self.parameters.z,
                                         v0=self.parameters.v,
                                         v1=self.parameters.sz,
                                         t=self.parameters.ter,
                                         V=self.parameters.sv, logp=False)

    def _go_fired(self):
        self.update_plot()

    def plot_histo(self, x, y, color, max_perc=None):
        # y consists of lower and upper boundary responses
        # mid point tells us where to split
        assert y.shape[0] % 2 == 0, "x_analytical has to be even. Shape is %s " % str(y.shape)
        mid = y.shape[0] / 2
        self.figure.axes[0].plot(x, y[mid:], color=color, lw=2.)
        # Compute correct EV
        mean_correct_rt = np.sum(x * y[mid:]) / np.sum(y[mid:])
        # [::-1] -> reverse ordering
        if self.color_errors:
            color='r'
        self.figure.axes[2].plot(x, -y[:mid][::-1], color=color, lw=2.)
        # Compute error EV
        mean_error_rt = np.sum(x * y[:mid][::-1]) / np.sum(y[:mid][::-1])

        if self.plot_mean_rt:
            self.figure.axes[0].axvline(mean_correct_rt, color=color)
            if self.color_errors:
                self.figure.axes[2].axvline(mean_error_rt, color='r')

    def plot_dens(self):
        t = np.linspace(0.0, self.parameters.T, self.x_raster)
        xm, ym = np.meshgrid(t, np.linspace(0, self.parameters.a, 100))
        zm = np.zeros_like(xm)  # np.zeros((t, 100), dtype=np.float)
        #print zs
        i = 0
        for xs, ys in zip(xm, ym):
            j = 0
            for x, y in zip(xs, ys):
                if x <= self.parameters.ter + .05:
                    zm[i, j] = 1.
                else:
                    zm[i, j] = hddm.wfpt_switch.drift_dens(y, x - self.parameters.ter, self.parameters.v, self.parameters.a, self.parameters.z * self.parameters.a)
                j += 1
        self.figure.axes[1].contourf(xm, ym, zm)

    @on_trait_change('parameters.params')
    def update_plot(self):
        if self.figure.canvas is None:
            return

        # Clear all plots
        for i in range(self.num_axes):
            self.figure.axes[i].clear()

        # Set x axis values
        x = np.linspace(0, self.parameters.T, self.parameters.bins)
        # Set x axis values for analyticals
        x_anal = np.linspace(0, self.parameters.T, self.x_raster / 2)

        # Plot normalized histograms of simulated data
        if self.plot_histogram:
            self.plot_histo(x, self.parameters.histo, color='b', max_perc=.99)

        # Plot normalized histograms of empirical data
        #if self.plot_data:
        #    histo = hddm.utils.histogram(self.data, bins=2*self.parameters.bins, range=(-self.parameters.T, self.parameters.T), dens=True)[0]
        #    self.plot_histo(x, histo, color='y')

        # Plot analytical simple likelihood function
        if self.plot_switch and self.parameters.switch:
            self.plot_histo(x_anal, self.switch, color='r')

        if self.plot_lba:
            self.plot_histo(x_anal, self.lba, color='k')

        if self.plot_wiener:
            self.plot_histo(x_anal, self.wiener, color='y')

        if self.plot_true_density:
            self.plot_dens()

        if self.plot_drifts:
            t = np.linspace(0.0, self.parameters.T, self.parameters.T / 1e-4)
            drifts = self.parameters.drifts
            for drift in drifts:
                # Make sure drift is not longer than x-axis
                drift_len = len(drift) if len(drift) < len(t) else len(t)
                if self.color_errors:
                    if drift[drift_len-1] < 0.1:
                        color = 'r'
                    else:
                        color = 'b'
                else:
                    color = 'b'
                self.figure.axes[1].plot(t[:drift_len],
                                         drift[:drift_len], color, alpha=.5)

        # Plot analytical simple likelihood function
        if self.parameters.switch:
            self.figure.axes[1].axvline(self.parameters.ter+self.parameters.t_switch, color='k', linestyle='--', lw=2)

        self.set_figure()

        wx.CallAfter(self.figure.canvas.draw)

        self.figure.savefig('DDM.svg')

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

        self.figure.axes[0].set_ylim((0, total_max))
        self.figure.axes[2].set_ylim((-total_max, 0))
        self.figure.axes[0].set_xlim((0, self.parameters.T))
        self.figure.axes[2].set_xlim((0, self.parameters.T))
        self.figure.axes[1].set_ylim((0, self.parameters.a))

        self.figure.subplots_adjust(hspace=0)
        for ax in self.ax1, self.ax2:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)

        plt.setp(self.ax3.get_yticklabels(), visible=False)

        self.ax3.set_xlabel('time (secs)')

if __name__ == "__main__":
    ddm = DDMPlot()
    ddm.configure_traits()
