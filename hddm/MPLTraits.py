import wx

import matplotlib
# We want matplotlib to use a wxPython backend
if __name__ == "__main__":
    matplotlib.use('WXAgg')

import matplotlib.pyplot
import pylab as pl

pl.ion()

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_wx import NavigationToolbar2Wx

from traits.api import HasTraits, Instance, Range,\
                                Array, on_trait_change, Property,\
                                cached_property, Bool
from traits.api import Any, Instance
from traitsui.wx.editor import Editor
from traitsui.basic_editor_factory import BasicEditorFactory

try:
    from IPython.Debugger import Tracer; debug_here = Tracer()
except:
    pass

class _MPLFigureEditor(Editor):

    scrollable  = True

    def init(self, parent):
        self.control = self._create_canvas(parent)
        self.set_tooltip()

    def update_editor(self):
        pass

    def _create_canvas(self, parent):
        """ Create the MPL canvas. """
        # The panel lets us add additional controls.
        panel = wx.Panel(parent, -1, style=wx.CLIP_CHILDREN)
        sizer = wx.BoxSizer(wx.VERTICAL)
        panel.SetSizer(sizer)
        # matplotlib commands to create a canvas
        mpl_control = FigureCanvas(panel, -1, self.value)
        sizer.Add(mpl_control, 1, wx.LEFT | wx.TOP | wx.GROW)
        toolbar = NavigationToolbar2Wx(mpl_control)
        sizer.Add(toolbar, 0, wx.EXPAND)
        self.value.canvas.SetMinSize((10,10))
        return panel

class MPLFigureEditor(BasicEditorFactory):

    klass = _MPLFigureEditor

def main():
    # Create a window to demo the editor
    from enthought.traits.api import HasTraits
    from enthought.traits.ui.api import View, Item
    from numpy import sin, cos, linspace, pi

    class Test(HasTraits):

        figure = Instance(Figure, ())
	test = Range(0,5,1)
        view = View(Item('figure', editor=MPLFigureEditor(),
                                show_label=False),
		    Item('test'),
		    width=400,
		    height=300,
		    resizable=True)

        def __init__(self):
            super(Test, self).__init__()
            self.axes = self.figure.add_subplot(111)
            self.t = linspace(0, 2*pi, 200)
	    param = self.test
	    self.line, = self.axes.plot(sin(self.t)*(1+0.5*cos(self.test*self.t)), cos(self.t)*(1+0.5*cos(11*self.t)))

	def update_plot(self):
	    self.figure.axes[0].clear()
	    self.figure.axes[0].plot(sin(self.t)*(1+0.5*cos(self.test*self.t)), cos(self.t)*(1+0.5*cos(11*self.t)))
            #self.axes.plot(sin(self.t)*(1+0.5*cos(self.test*self.t)), cos(self.t)*(1+0.5*cos(11*self.t)))
	    #self.axes.redraw_in_frame()
	    wx.CallAfter(self.figure.canvas.draw)



	def _test_changed(self):
	    self.update_plot()


    #wx.EVT_IDLE(wx.GetApp(), callback)
    Test().configure_traits()

#if __name__ == "__main__":
#    main()


