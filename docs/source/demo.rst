Import the modules we are going to use. Pandas for the dataframe and
matplotlib for plotting.

.. ipython:: python
   :suppress:

   import numpy as np
   np.random.seed(123456)
   np.set_printoptions(precision=4, suppress=True)

.. ipython:: python

   import pandas as pd
   import matplotlib.pyplot as plt


At the time of this writing, these version were used

.. ipython:: python

   import hddm
   print hddm.__version__

   import kabuki
   print kabuki.__version__

First, we have to load a data set. The easiest way to get your data
into HDDM is via csv. In this example we will be using data collected
in a reinforcement learning experiment. The data file looks as
follows:

.. ipython:: python

   !head PD_PS.csv

We use the ``hddm.load_csv`` function to load this file and create a
pandas dataframe for us.

.. ipython:: python

   data = pd.DataFrame(hddm.load_csv('PD_PS.csv'))
   del data['conf']
   data = hddm.utils.flip_errors(data)
   data.head()

.. ipython:: python

   fig = plt.figure()
   ax = fig.add_subplot(111, xlabel='RT', ylabel='count', title='RT distributions')
   for i, subj_data in data.groupby('subj_idx'):
       @savefig hist_subjs.png width=4in
       ax.hist(subj_data.rt, bins=20, histtype='step')

   plt.show()
