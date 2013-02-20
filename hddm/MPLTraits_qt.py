import PyQt4
from pyface.qt import QtGui, QtCore

import matplotlib
# We want matplotlib to use a QT backend
matplotlib.use('Qt4Agg')
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from traits.api import Any, Instance
from traitsui.qt4.editor import Editor
from traitsui.qt4.basic_editor_factory import BasicEditorFactory

class _MPLFigureEditor(Editor):

   scrollable  = True

   def init(self, parent):
       self.control = self._create_canvas(parent)
       self.set_tooltip()

   def update_editor(self):
       pass

   def _create_canvas(self, parent):
       """ Create the MPL canvas. """
       # matplotlib commands to create a canvas
       mpl_canvas = FigureCanvas(self.value)
       return mpl_canvas

class MPLFigureEditor(BasicEditorFactory):

   klass = _MPLFigureEditor


if __name__ == "__main__":
   # Create a window to demo the editor
   from traits.api import HasTraits
   from traitsui.api import View, Item
   from numpy import sin, cos, linspace, pi

   class Test(HasTraits):

       figure = Instance(Figure, ())

       view = View(Item('figure', editor=MPLFigureEditor(),
                               show_label=False),
                       width=400,
                       height=300,
                       resizable=True)

       def __init__(self):
           super(Test, self).__init__()
           axes = self.figure.add_subplot(111)
           t = linspace(0, 2*pi, 200)
           axes.plot(sin(t)*(1+0.5*cos(11*t)), cos(t)*(1+0.5*cos(11*t)))

   Test().configure_traits()
