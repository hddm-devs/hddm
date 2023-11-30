Tutorial on Parameter defaults
==============================

As of version ``0.9.8``, ``HDDM`` doesn’t expect that you always
explicitly want to fit the ``v``, ``a`` and ``t`` parameters. You are
now allowed to fix any of these parameters to any default you like. In
this tutorial we show how to fit any given subset of parameters of a
model, while supplying (user picked) default values for the remaining
parameters.

Install (colab)
---------------

.. code:: ipython3

    # package to help train networks
    # !pip install git+https://github.com/AlexanderFengler/LANfactory
    
    # package containing simulators for ssms
    # !pip install git+https://github.com/AlexanderFengler/ssm_simulators
    
    # packages related to hddm
    # !pip install cython
    # !pip install pymc==2.3.8
    # !pip install git+https://github.com/hddm-devs/kabuki
    # !pip install git+https://github.com/hddm-devs/hddm

Load Modules
------------

.. code:: ipython3

    # MODULE IMPORTS ----
    
    # warning settings
    import warnings
    
    warnings.simplefilter(action="ignore", category=FutureWarning)
    
    # Data management
    import pandas as pd
    import numpy as np
    import pickle
    
    # Plotting
    import matplotlib.pyplot as plt
    import matplotlib
    import seaborn as sns
    
    # Stats functionality
    from statsmodels.distributions.empirical_distribution import ECDF
    
    # HDDM
    import hddm
    from hddm.simulators.hddm_dataset_generators import simulator_h_c

Example Models
--------------

``HDDM()``
~~~~~~~~~~

Simulate Data
^^^^^^^^^^^^^

.. code:: ipython3

    #from hddm.simulators.hddm_dataset_generators import simulator_h_c
    from hddm.simulators.basic_simulator import simulator
    from hddm.simulators.hddm_dataset_generators import hddm_preprocess
    
    model = 'ddm_hddm_base'
    
    data = simulator(theta = [1., 1., 0.5, 0.5],
                     model = model,
                     n_samples = 500)
    
    data = hddm_preprocess(data)

Model and Sample
^^^^^^^^^^^^^^^^

Let’s first fit all parameters.

.. code:: ipython3

    hddm_model = hddm.HDDM(data,
                            include = ['v', 'a', 't', 'z'],
                            informative = False,
                            is_group_model = False,
                           )


.. parsed-literal::

    No model attribute --> setting up standard HDDM
    Set model to ddm


.. code:: ipython3

    hddm_model.sample(1000, burn = 500)


.. parsed-literal::

     [-----------------100%-----------------] 1000 of 1000 complete in 8.5 sec



.. parsed-literal::

    <pymc.MCMC.MCMC at 0x7ff3014fc210>



.. code:: ipython3

    hddm_model.gen_stats()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>mean</th>
          <th>std</th>
          <th>2.5q</th>
          <th>25q</th>
          <th>50q</th>
          <th>75q</th>
          <th>97.5q</th>
          <th>mc err</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>a</th>
          <td>0.996413</td>
          <td>0.022212</td>
          <td>0.949557</td>
          <td>0.983287</td>
          <td>0.995682</td>
          <td>1.011092</td>
          <td>1.040408</td>
          <td>0.001349</td>
        </tr>
        <tr>
          <th>v</th>
          <td>1.150145</td>
          <td>0.122256</td>
          <td>0.925338</td>
          <td>1.072607</td>
          <td>1.14161</td>
          <td>1.225006</td>
          <td>1.435445</td>
          <td>0.007297</td>
        </tr>
        <tr>
          <th>t</th>
          <td>0.501954</td>
          <td>0.003282</td>
          <td>0.495503</td>
          <td>0.500019</td>
          <td>0.502353</td>
          <td>0.504147</td>
          <td>0.507753</td>
          <td>0.000225</td>
        </tr>
        <tr>
          <th>z</th>
          <td>0.488272</td>
          <td>0.015967</td>
          <td>0.45511</td>
          <td>0.477788</td>
          <td>0.489069</td>
          <td>0.498717</td>
          <td>0.519293</td>
          <td>0.001011</td>
        </tr>
      </tbody>
    </table>
    </div>



Now we **fix ``a`` to it’s default** as per the ``HDDM``-supplied
``model_config`` dictionary. As shown below, this sets ``a = 2.`` which
corresponds to an overestimation. We expect that, having fixed ``a`` at
such value, we will correspondingly overestimate ``v`` to compensate
(however the fit will end up worse in general).

.. code:: ipython3

    hddm.model_config.model_config['ddm_hddm_base']




.. parsed-literal::

    {'doc': 'Model used internally for simulation purposes. Do NOT use with the LAN extension.',
     'params': ['v', 'a', 'z', 't'],
     'params_trans': [0, 0, 1, 0],
     'params_std_upper': [1.5, 1.0, None, 1.0],
     'param_bounds': [[-5.0, 0.1, 0.05, 0], [5.0, 5.0, 0.95, 3.0]],
     'boundary': <function ssms.basic_simulators.boundary_functions.constant(t=0)>,
     'params_default': [0.0, 2.0, 0.5, 0],
     'hddm_include': ['v', 'a', 't', 'z'],
     'choices': [0, 1],
     'slice_widths': {'v': 1.5,
      'v_std': 1,
      'a': 1,
      'a_std': 1,
      'z': 0.1,
      'z_trans': 0.2,
      't': 0.01,
      't_std': 0.15}}



.. code:: ipython3

    hddm_model_no_a = hddm.HDDM(data,
                            include = ['v', 't', 'z'],
                            informative = False,
                            is_group_model = False,
                           )


.. parsed-literal::

    No model attribute --> setting up standard HDDM
    Set model to ddm


.. parsed-literal::

    /Users/afengler/OneDrive/project_hddm_extension/hddm/hddm/models/base.py:1316: UserWarning:  
     Your include statement misses either the v, a or t parameters. 
    Parameters not explicitly included will be set to the defaults, 
    which you can find in the model_config dictionary!
      "Parameters not explicitly included will be set to the defaults, \n" + \


.. code:: ipython3

    hddm_model_no_a.sample(1000, burn = 500)


.. parsed-literal::

     [-----------------100%-----------------] 1000 of 1000 complete in 5.6 sec



.. parsed-literal::

    <pymc.MCMC.MCMC at 0x7ff301546b50>



.. code:: ipython3

    hddm_model_no_a.gen_stats()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>mean</th>
          <th>std</th>
          <th>2.5q</th>
          <th>25q</th>
          <th>50q</th>
          <th>75q</th>
          <th>97.5q</th>
          <th>mc err</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>v</th>
          <td>2.077952</td>
          <td>0.146506</td>
          <td>1.806508</td>
          <td>1.977741</td>
          <td>2.075691</td>
          <td>2.164686</td>
          <td>2.385388</td>
          <td>0.011489</td>
        </tr>
        <tr>
          <th>t</th>
          <td>0.335338</td>
          <td>0.009595</td>
          <td>0.315019</td>
          <td>0.330216</td>
          <td>0.335275</td>
          <td>0.341575</td>
          <td>0.353736</td>
          <td>0.000553</td>
        </tr>
        <tr>
          <th>z</th>
          <td>0.36132</td>
          <td>0.022235</td>
          <td>0.319808</td>
          <td>0.346506</td>
          <td>0.360754</td>
          <td>0.376082</td>
          <td>0.406029</td>
          <td>0.001815</td>
        </tr>
      </tbody>
    </table>
    </div>



As predicted, ``v`` is now overestimated as well.

Let’s now try to set ``a`` to a default of our liking. We will set it to
the ground-truth and again not include it in the parameters to estimate.
To do so, we supply our own ``model_config`` to the ``HDDM()`` class.

.. code:: ipython3

    from copy import deepcopy
    # copy model_config dictionary so we can change it 
    my_model_config = deepcopy(hddm.model_config.model_config['ddm_hddm_base'])
    
    # setting 'a' to 1.
    my_model_config['params_default'][1] = 1.
    
    hddm_model_no_a_2 = hddm.HDDM(data,
                            include = ['v', 't', 'z'],
                            informative = False,
                            is_group_model = False,
                            model_config = my_model_config
                           )


.. parsed-literal::

    Custom model config supplied as: 
    
    {'doc': 'Model used internally for simulation purposes. Do NOT use with the LAN extension.', 'params': ['v', 'a', 'z', 't'], 'params_trans': [0, 0, 1, 0], 'params_std_upper': [1.5, 1.0, None, 1.0], 'param_bounds': [[-5.0, 0.1, 0.05, 0], [5.0, 5.0, 0.95, 3.0]], 'boundary': <function constant at 0x7ff31c1fab90>, 'params_default': [0.0, 1.0, 0.5, 0], 'hddm_include': ['v', 'a', 't', 'z'], 'choices': [0, 1], 'slice_widths': {'v': 1.5, 'v_std': 1, 'a': 1, 'a_std': 1, 'z': 0.1, 'z_trans': 0.2, 't': 0.01, 't_std': 0.15}}
    No model attribute --> setting up standard HDDM
    Set model to ddm


.. parsed-literal::

    /Users/afengler/OneDrive/project_hddm_extension/hddm/hddm/models/base.py:1316: UserWarning:  
     Your include statement misses either the v, a or t parameters. 
    Parameters not explicitly included will be set to the defaults, 
    which you can find in the model_config dictionary!
      "Parameters not explicitly included will be set to the defaults, \n" + \


.. code:: ipython3

    hddm_model_no_a_2.sample(1000, burn = 500)


.. parsed-literal::

     [-----------------100%-----------------] 1000 of 1000 complete in 5.2 sec



.. parsed-literal::

    <pymc.MCMC.MCMC at 0x7ff30157b150>



.. code:: ipython3

    hddm_model_no_a_2.gen_stats()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>mean</th>
          <th>std</th>
          <th>2.5q</th>
          <th>25q</th>
          <th>50q</th>
          <th>75q</th>
          <th>97.5q</th>
          <th>mc err</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>v</th>
          <td>1.171748</td>
          <td>0.118515</td>
          <td>0.935492</td>
          <td>1.094611</td>
          <td>1.170236</td>
          <td>1.251087</td>
          <td>1.425975</td>
          <td>0.006572</td>
        </tr>
        <tr>
          <th>t</th>
          <td>0.501641</td>
          <td>0.002593</td>
          <td>0.496291</td>
          <td>0.499915</td>
          <td>0.50178</td>
          <td>0.503453</td>
          <td>0.506346</td>
          <td>0.000121</td>
        </tr>
        <tr>
          <th>z</th>
          <td>0.486188</td>
          <td>0.016581</td>
          <td>0.450828</td>
          <td>0.476066</td>
          <td>0.486156</td>
          <td>0.497147</td>
          <td>0.518931</td>
          <td>0.000963</td>
        </tr>
      </tbody>
    </table>
    </div>



As we see, in this case ``v`` is estimated appropriately again.

Let’s compare DICs
''''''''''''''''''

.. code:: ipython3

    print('Standard: ', hddm_model.dic)
    print('No a with HDDM default: ', hddm_model_no_a.dic)
    print('No a with a set to ground truth: ', hddm_model_no_a_2.dic)


.. parsed-literal::

    Standard:  -7.05123814817064
    No a with HDDM default:  562.273161208081
    No a with a set to ground truth:  -9.028954442474097


HDDMnn()
~~~~~~~~

Let’s repeat this with another model via the ``HDDMnn()`` class. We will
pick the ``HDDM``-supplied ``angle`` model.

Simulate Data
^^^^^^^^^^^^^

.. code:: ipython3

    model = 'angle'
    theta = [1., 1.5, .5, .5, 0.2] # v, a, z, t, theta
    data_angle = simulator(theta = theta,
                           model = 'angle',
                           n_samples = 500)
    data_angle = hddm_preprocess(data_angle,
                                 keep_negative_responses = True)

Model and Sample
^^^^^^^^^^^^^^^^

.. code:: ipython3

    model_angle = hddm.HDDMnn(data_angle,
                              model = 'angle',
                              include = ['v', 'a', 't', 'z', 'theta'])


.. parsed-literal::

    Using default priors: Uninformative
    Supplied model_config specifies params_std_upper for  z as  None.
    Changed to 10


.. code:: ipython3

    model_angle.sample(1000, burn = 500)


.. parsed-literal::

     [-----------------100%-----------------] 1000 of 1000 complete in 52.0 sec



.. parsed-literal::

    <pymc.MCMC.MCMC at 0x7ff301575390>



.. code:: ipython3

    model_angle.gen_stats()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>mean</th>
          <th>std</th>
          <th>2.5q</th>
          <th>25q</th>
          <th>50q</th>
          <th>75q</th>
          <th>97.5q</th>
          <th>mc err</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>v</th>
          <td>1.020563</td>
          <td>0.083431</td>
          <td>0.852936</td>
          <td>0.964464</td>
          <td>1.020123</td>
          <td>1.074769</td>
          <td>1.178448</td>
          <td>0.006605</td>
        </tr>
        <tr>
          <th>a</th>
          <td>1.584719</td>
          <td>0.098083</td>
          <td>1.419463</td>
          <td>1.511524</td>
          <td>1.576579</td>
          <td>1.651892</td>
          <td>1.787859</td>
          <td>0.009077</td>
        </tr>
        <tr>
          <th>z</th>
          <td>0.526568</td>
          <td>0.025441</td>
          <td>0.475032</td>
          <td>0.510146</td>
          <td>0.527315</td>
          <td>0.543314</td>
          <td>0.577972</td>
          <td>0.002245</td>
        </tr>
        <tr>
          <th>t</th>
          <td>0.494598</td>
          <td>0.037365</td>
          <td>0.420589</td>
          <td>0.470531</td>
          <td>0.495294</td>
          <td>0.521575</td>
          <td>0.561596</td>
          <td>0.003361</td>
        </tr>
        <tr>
          <th>theta</th>
          <td>0.270865</td>
          <td>0.050656</td>
          <td>0.177934</td>
          <td>0.236454</td>
          <td>0.269616</td>
          <td>0.304024</td>
          <td>0.377811</td>
          <td>0.004407</td>
        </tr>
      </tbody>
    </table>
    </div>



Again we will now leave out one parameter (let’s pick ``theta`` this
time). As we can see from the printed ``model_config`` below, the
default that will be chosen for this parameter is to set it to ``0`` in
this case.

.. code:: ipython3

    hddm.model_config.model_config




.. parsed-literal::

    {'ddm_hddm_base': {'doc': 'Model used internally for simulation purposes. Do NOT use with the LAN extension.',
      'params': ['v', 'a', 'z', 't'],
      'params_trans': [0, 0, 1, 0],
      'params_std_upper': [1.5, 1.0, None, 1.0],
      'param_bounds': [[-5.0, 0.1, 0.05, 0], [5.0, 5.0, 0.95, 3.0]],
      'boundary': <function ssms.basic_simulators.boundary_functions.constant(t=0)>,
      'params_default': [0.0, 2.0, 0.5, 0],
      'hddm_include': ['v', 'a', 't', 'z'],
      'choices': [0, 1],
      'slice_widths': {'v': 1.5,
       'v_std': 1,
       'a': 1,
       'a_std': 1,
       'z': 0.1,
       'z_trans': 0.2,
       't': 0.01,
       't_std': 0.15}},
     'full_ddm_hddm_base': {'doc': 'Model used internally for simulation purposes. Do NOT use with the LAN extension.',
      'params': ['v', 'a', 'z', 't', 'sz', 'sv', 'st'],
      'params_trans': [0, 0, 1, 0, 0, 0, 0],
      'params_std_upper': [1.5, 1.0, None, 1.0, 0.1, 0.5, 0.1],
      'param_bounds': [[-5.0, 0.1, 0.3, 0.25, 0, 0, 0],
       [5.0, 5.0, 0.7, 2.25, 0.25, 4.0, 0.25]],
      'boundary': <function ssms.basic_simulators.boundary_functions.constant(t=0)>,
      'params_default': [0.0, 1.0, 0.5, 0.25, 0, 0, 0],
      'hddm_include': ['v', 'a', 't', 'z', 'st', 'sv', 'sz'],
      'choices': [0, 1],
      'slice_widths': {'v': 1.5,
       'v_std': 1,
       'a': 1,
       'a_std': 1,
       'z': 0.1,
       'z_trans': 0.2,
       't': 0.01,
       't_std': 0.15,
       'sz': 1.1,
       'st': 0.1,
       'sv': 0.5}},
     'ddm': {'doc': 'Basic DDM. Meant for use with the LAN extension. \nNote that the boundaries here are coded as -a, and a in line with all other models meant for the LAN extension. \nTo compare model fits between standard HDDM and HDDMnn when using the DDM model, multiply the boundary (a) parameter by 2. \nWe recommend using standard HDDM if you are interested in the basic DDM, but you might want to use this for testing.',
      'params': ['v', 'a', 'z', 't'],
      'params_trans': [0, 0, 1, 0],
      'params_std_upper': [1.5, 1.0, None, 1.0],
      'param_bounds': [[-3.0, 0.3, 0.1, 0.001], [3.0, 2.5, 0.9, 2.0]],
      'boundary': <function ssms.basic_simulators.boundary_functions.constant(t=0)>,
      'params_default': [0.0, 1.0, 0.5, 0.001],
      'hddm_include': ['v', 'a', 't', 'z'],
      'choices': [-1, 1],
      'slice_widths': {'v': 1.5,
       'v_std': 1,
       'a': 1,
       'a_std': 1,
       'z': 0.1,
       'z_trans': 0.2,
       't': 0.01,
       't_std': 0.15}},
     'angle': {'doc': 'Model formulation is described in the documentation under LAN Extension.\nMeant for use with the extension.',
      'params': ['v', 'a', 'z', 't', 'theta'],
      'params_trans': [0, 0, 1, 0, 0],
      'params_std_upper': [1.5, 1.0, None, 1.0, 1.0],
      'param_bounds': [[-3.0, 0.3, 0.1, 0.001, -0.1], [3.0, 3.0, 0.9, 2.0, 1.3]],
      'boundary': <function ssms.basic_simulators.boundary_functions.angle(t=1, theta=1)>,
      'params_default': [0.0, 1.0, 0.5, 0.001, 0.1],
      'hddm_include': ['v', 'a', 't', 'z', 'theta'],
      'choices': [-1, 1],
      'slice_widths': {'v': 1.5,
       'v_std': 1,
       'a': 1,
       'a_std': 1,
       'z': 0.1,
       'z_trans': 0.2,
       't': 0.01,
       't_std': 0.15,
       'theta': 0.1,
       'theta_std': 0.2}},
     'weibull': {'doc': 'Model formulation is described in the documentation under LAN Extension.\nMeant for use with the extension.',
      'params': ['v', 'a', 'z', 't', 'alpha', 'beta'],
      'params_trans': [0, 0, 1, 0, 0, 0],
      'params_std_upper': [1.5, 1.0, None, 1.0, 2.0, 2.0],
      'param_bounds': [[-2.5, 0.3, 0.2, 0.001, 0.31, 0.31],
       [2.5, 2.5, 0.8, 2.0, 4.99, 6.99]],
      'boundary': <function ssms.basic_simulators.boundary_functions.weibull_cdf(t=1, alpha=1, beta=1)>,
      'params_default': [0.0, 1.0, 0.5, 0.001, 3.0, 3.0],
      'hddm_include': ['v', 'a', 't', 'z', 'alpha', 'beta'],
      'choices': [-1, 1],
      'slice_widths': {'v': 1.5,
       'v_std': 1,
       'a': 1,
       'a_std': 1,
       'z': 0.1,
       'z_trans': 0.2,
       't': 0.01,
       't_std': 0.15,
       'alpha': 1.0,
       'alpha_std': 0.5,
       'beta': 1.0,
       'beta_std': 0.5}},
     'levy': {'doc': 'Model formulation is described in the documentation under LAN Extension.\nMeant for use with the extension.',
      'params': ['v', 'a', 'z', 'alpha', 't'],
      'params_trans': [0, 0, 1, 0, 0],
      'params_std_upper': [1.5, 1.0, None, 1.0, 1.0],
      'param_bounds': [[-3.0, 0.3, 0.1, 1.0, 0.001], [3.0, 2.0, 0.9, 2.0, 2]],
      'boundary': <function ssms.basic_simulators.boundary_functions.constant(t=0)>,
      'params_default': [0.0, 1.0, 0.5, 1.5, 0.001],
      'hddm_include': ['v', 'a', 't', 'z', 'alpha'],
      'choices': [-1, 1],
      'slice_widths': {'v': 1.5,
       'v_std': 1,
       'a': 1,
       'a_std': 1,
       'z': 0.1,
       'z_trans': 0.2,
       't': 0.01,
       't_std': 0.15,
       'alpha': 1.0,
       'alpha_std': 0.5}},
     'full_ddm': {'doc': 'Currently unavailable, for LANs after switch to pytorch. \nComing soon... Please use standard HDDM if you want to fit this model to your data.',
      'params': ['v', 'a', 'z', 't', 'sz', 'sv', 'st'],
      'params_trans': [0, 0, 1, 0, 0, 0, 0],
      'params_std_upper': [1.5, 1.0, None, 1.0, 0.1, 0.5, 0.1],
      'param_bounds': [[-3.0, 0.3, 0.3, 0.25, 0.001, 0.001, 0.001],
       [3.0, 2.5, 0.7, 2.25, 0.2, 2.0, 0.25]],
      'boundary': <function ssms.basic_simulators.boundary_functions.constant(t=0)>,
      'params_default': [0.0, 1.0, 0.5, 0.25, 0.001, 0.001, 0.001],
      'hddm_include': ['v', 'a', 't', 'z', 'st', 'sv', 'sz'],
      'choices': [-1, 1],
      'slice_widths': {'v': 1.5,
       'v_std': 1,
       'a': 1,
       'a_std': 1,
       'z': 0.1,
       'z_trans': 0.2,
       't': 0.01,
       't_std': 0.15,
       'sz': 1.1,
       'st': 0.1,
       'sv': 0.5}},
     'ornstein': {'doc': 'Model formulation is described in the documentation under LAN Extension.Meant for use with the extension.',
      'params': ['v', 'a', 'z', 'g', 't'],
      'params_trans': [0, 0, 1, 0, 0],
      'params_std_upper': [1.5, 1.0, None, 1.0, 1.0],
      'param_bounds': [[-2.0, 0.3, 0.2, -1.0, 0.001], [2.0, 2.0, 0.8, 1.0, 2]],
      'boundary': <function ssms.basic_simulators.boundary_functions.constant(t=0)>,
      'params_default': [0.0, 1.0, 0.5, 0.0, 0.001],
      'hddm_include': ['v', 'a', 't', 'z', 'g'],
      'choices': [-1, 1],
      'slice_widths': {'v': 1.5,
       'v_std': 0.1,
       'a': 1,
       'a_std': 0.1,
       'z': 0.1,
       'z_trans': 0.2,
       't': 0.01,
       't_std': 0.15,
       'g': 0.1,
       'g_trans': 0.2,
       'g_std': 0.1}},
     'ddm_sdv': {'doc': 'Currently unavailable, for LANs after switch to pytorch. Coming soon...Please use standard HDDM if you want to fit this model to your data.',
      'params': ['v', 'a', 'z', 't', 'sv'],
      'params_trans': [0, 0, 1, 0, 0],
      'params_std_upper': [1.5, 1.0, None, 1.0, 1.0],
      'param_bounds': [[-3.0, 0.3, 0.1, 0.001, 0.001], [3.0, 2.5, 0.9, 2.0, 2.5]],
      'boundary': <function ssms.basic_simulators.boundary_functions.constant(t=0)>,
      'params_default': [0.0, 1.0, 0.5, 0.001, 0.001],
      'hddm_include': ['v', 'a', 't', 'z', 'sv'],
      'choices': [-1, 1],
      'slice_widths': {'v': 1.5,
       'v_std': 1,
       'a': 1,
       'a_std': 1,
       'z': 0.1,
       'z_trans': 0.2,
       't': 0.01,
       't_std': 0.15,
       'sv': 0.5}},
     'gamma_drift': {'doc': 'Meant for use with the LAN extension',
      'params': ['v', 'a', 'z', 't', 'shape', 'scale', 'c'],
      'params_trans': [0, 0, 1, 0, 0, 0, 0],
      'params_std_upper': [1.5, 1.0, None, 1.0, 2.0, 2.0, 1.5],
      'param_bounds': [[-3.0, 0.3, 0.1, 0.001, 2.0, 0.01, -3.0],
       [3.0, 3.0, 0.9, 2.0, 10.0, 1.0, 3.0]],
      'boundary': <function ssms.basic_simulators.boundary_functions.constant(t=0)>,
      'params_default': [0.0, 1.0, 0.5, 0.25, 5.0, 0.5, 1.0],
      'hddm_include': ['v', 'a', 't', 'z', 'shape', 'scale', 'c'],
      'choices': [-1, 1],
      'slice_widths': {'v': 1.5,
       'v_std': 1,
       'a': 1,
       'a_std': 1,
       'z': 0.1,
       'z_trans': 0.2,
       't': 0.01,
       't_std': 0.15,
       'shape': 1,
       'shape_std': 1,
       'scale': 1,
       'scale_std': 1,
       'c': 1,
       'c_std': 1}},
     'gamma_drift_angle': {'doc': 'Meant for use with the LAN extension',
      'params': ['v', 'a', 'z', 't', 'theta', 'shape', 'scale', 'c'],
      'params_trans': [0, 0, 1, 0, 0, 0, 0, 0],
      'params_std_upper': [1.5, 1.0, None, 1.0, 1.0, 2.0, 2.0, 1.5],
      'param_bounds': [[-3.0, 0.3, 0.1, 0.001, -0.1, 2.0, 0.01, -3.0],
       [3.0, 3.0, 0.9, 2.0, 1.3, 10.0, 1.0, 3.0]],
      'boundary': <function ssms.basic_simulators.boundary_functions.angle(t=1, theta=1)>,
      'params_default': [0.0, 1.0, 0.5, 0.25, 0.0, 5.0, 0.5, 1.0],
      'hddm_include': ['v', 'a', 't', 'z', 'shape', 'scale', 'c', 'theta'],
      'choices': [-1, 1],
      'slice_widths': {'v': 1.5,
       'v_std': 1,
       'a': 1,
       'a_std': 1,
       'z': 0.1,
       'z_trans': 0.2,
       't': 0.01,
       't_std': 0.15,
       'theta': 0.1,
       'theta_std': 0.2,
       'shape': 1,
       'shape_std': 1,
       'scale': 1,
       'scale_std': 1,
       'c': 1,
       'c_std': 1}},
     'ds_conflict_drift': {'doc': 'Meant for use with LAN extension.',
      'params': ['a',
       'z',
       't',
       'tinit',
       'dinit',
       'tslope',
       'dslope',
       'tfixedp',
       'tcoh',
       'dcoh'],
      'param_bounds': [[0.3, 0.1, 0.001, 0, 0, 0.01, 0.01, 0, -1.0, -1.0],
       [3.0, 0.9, 2.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0]],
      'params_trans': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
      'params_std_upper': [1.0, None, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.0, 1.0],
      'params_default': [0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 1.0],
      'hddm_include': ['a',
       'z',
       't',
       'tinit',
       'dinit',
       'tslope',
       'dslope',
       'tfixedp',
       'tcoh',
       'dcoh'],
      'boundary': <function ssms.basic_simulators.boundary_functions.constant(t=0)>,
      'choices': [-1, 1],
      'slice_widhts': {'a': 1,
       'a_std': 1,
       'z': 0.1,
       'z_trans': 0.2,
       'z_std': 1,
       'z_trans_std': 1,
       't': 0.01,
       't_std': 1,
       'tinit': 0.1,
       'tinit_std': 1,
       'dinit': 0.1,
       'dinit_std': 1,
       'tslope': 1,
       'tslope_std': 1,
       'dslope': 1,
       'dslope_std': 1,
       'tfixedp': 1,
       'tfixedp_std': 1,
       'tcoh': 1,
       'tcoh_std': 1,
       'dcoh': 1,
       'dcohe_std': 1}},
     'ds_conflict_drift_angle': {'doc': 'Meant for use with LAN extension.',
      'params': ['a',
       'z',
       't',
       'tinit',
       'dinit',
       'tslope',
       'dslope',
       'tfixedp',
       'tcoh',
       'dcoh',
       'angle'],
      'param_bounds': [[0.3, 0.1, 0.001, 0, 0, 0.01, 0.01, 0, -1.0, -1.0, -0.1],
       [3.0, 0.9, 2.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.3]],
      'params_trans': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      'params_std_upper': [1.0, None, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0],
      'params_default': [0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 1.0, 0.0],
      'hddm_include': ['a',
       'z',
       't',
       'tinit',
       'dinit',
       'tslope',
       'dslope',
       'tfixedp',
       'tcoh',
       'dcoh',
       'theta'],
      'boundary': <function ssms.basic_simulators.boundary_functions.angle(t=1, theta=1)>,
      'choices': [-1, 1],
      'slice_widhts': {'a': 1,
       'a_std': 1,
       'z': 0.1,
       'z_trans': 0.2,
       'z_std': 1,
       'z_trans_std': 1,
       't': 0.01,
       't_std': 1,
       'tinit': 0.1,
       'tinit_std': 1,
       'dinit': 0.1,
       'dinit_std': 1,
       'tslope': 1,
       'tslope_std': 1,
       'dslope': 1,
       'dslope_std': 1,
       'tfixedp': 1,
       'tfixedp_std': 1,
       'tcoh': 1,
       'tcoh_std': 1,
       'dcoh': 1,
       'dcoh_std': 1,
       'theta': 0.1,
       'theta_std': 0.2}},
     'ddm_par2': {'doc': 'Currently undocumented, in testing phase.',
      'params': ['vh', 'vl1', 'vl2', 'a', 'zh', 'zl1', 'zl2', 't'],
      'params_trans': [0, 0, 0, 0, 1, 1, 1, 0],
      'params_std_upper': [1.5, 1.5, 1.5, 1.0, None, None, None, 1.0],
      'param_bounds': [[-2.5, -2.5, -2.5, 0.3, 0.2, 0.2, 0.2, 0.0],
       [2.5, 2.5, 2.5, 2.0, 0.8, 0.8, 0.8, 2.0]],
      'boundary': <function ssms.basic_simulators.boundary_functions.constant(t=0)>,
      'params_default': [0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 1.0],
      'hddm_include': ['vh', 'vl1', 'vl2', 'a', 'zh', 'zl1', 'zl2', 't'],
      'choices': [0, 1, 2, 3],
      'slice_widths': {'vh': 1.5,
       'vh_std': 0.5,
       'vl1': 1.5,
       'vl1_std': 0.5,
       'vl2': 1.5,
       'vl2_std': 0.5,
       'a': 1,
       'a_std': 1,
       'zh': 0.1,
       'zh_trans': 0.2,
       'zl1': 0.1,
       'zl1_trans': 0.2,
       'zl2': 0.1,
       'zl2_trans': 0.2,
       't': 0.01,
       't_std': 0.15}},
     'ddm_par2_no_bias': {'doc': 'Currently undocumented, in testing phase.',
      'params': ['vh', 'vl1', 'vl2', 'a', 't'],
      'param_bounds': [[-2.5, -2.5, -2.5, 0.3, 0.0], [2.5, 2.5, 2.5, 2.0, 2.0]],
      'params_trans': [0, 0, 0, 0, 0],
      'params_std_upper': [1.5, 1.5, 1.5, 1.0, 1.0],
      'boundary': <function ssms.basic_simulators.boundary_functions.constant(t=0)>,
      'params_default': [0.0, 0.0, 0.0, 1.0, 1.0],
      'hddm_include': ['vh', 'vl1', 'vl2', 'a', 't'],
      'choices': [0, 1, 2, 3],
      'slice_widths': {'vh': 1.5,
       'vh_std': 0.5,
       'vl1': 1.5,
       'vl1_std': 0.5,
       'vl2': 1.5,
       'vl2_std': 0.5,
       'a': 1,
       'a_std': 1,
       't': 0.01,
       't_std': 0.15}},
     'ddm_par2_angle_no_bias': {'doc': 'Currently undocumented, in testing phase.',
      'params': ['vh', 'vl1', 'vl2', 'a', 't', 'theta'],
      'param_bounds': [[-2.5, -2.5, -2.5, 0.3, 0.0, -0.1],
       [2.5, 2.5, 2.5, 2.0, 2.0, 1.0]],
      'params_trans': [0, 0, 0, 0, 0, 0],
      'params_std_upper': [1.5, 1.5, 1.5, 1.0, 1.0, 1.0],
      'boundary': <function ssms.basic_simulators.boundary_functions.angle(t=1, theta=1)>,
      'boundary_multiplicative': False,
      'params_default': [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
      'hddm_include': ['vh', 'vl1', 'vl2', 'a', 't', 'theta'],
      'choices': [0, 1, 2, 3],
      'slice_widths': {'vh': 1.5,
       'vh_std': 0.5,
       'vl1': 1.5,
       'vl1_std': 0.5,
       'vl2': 1.5,
       'vl2_std': 0.5,
       'a': 1,
       'a_std': 1,
       't': 0.01,
       't_std': 0.15,
       'theta': 0.1,
       'theta_std': 0.2}},
     'ddm_par2_weibull_no_bias': {'doc': 'Currently undocumented, in testing phase.',
      'params': ['vh', 'vl1', 'vl2', 'a', 't', 'alpha', 'beta'],
      'param_bounds': [[-2.5, -2.5, -2.5, 0.3, 0.0, 0.31, 0.31],
       [2.5, 2.5, 2.5, 2.0, 2.0, 4.99, 6.99]],
      'params_trans': [0, 0, 0, 0, 0, 0, 0],
      'params_std_upper': [1.5, 1.5, 1.5, 1.0, 1.0, 1.5, 1.5],
      'boundary': <function ssms.basic_simulators.boundary_functions.weibull_cdf(t=1, alpha=1, beta=1)>,
      'boundary_multiplicative': True,
      'params_default': [0.0, 0.0, 0.0, 1.0, 1.0, 2.5, 3.5],
      'hddm_include': ['vh', 'vl1', 'vl2', 'a', 't', 'alpha', 'beta'],
      'choices': [0, 1, 2, 3],
      'slice_widths': {'vh': 1.5,
       'vh_std': 0.5,
       'vl1': 1.5,
       'vl1_std': 0.5,
       'vl2': 1.5,
       'vl2_std': 0.5,
       'a': 1,
       'a_std': 1,
       't': 0.01,
       't_std': 0.15,
       'theta': 0.1,
       'theta_std': 0.2,
       'alpha': 1.0,
       'alpha_std': 0.5,
       'beta': 1.0,
       'beta_std': 0.5}},
     'ddm_seq2': {'doc': 'Currently undocumented, in testing phase.',
      'params': ['vh', 'vl1', 'vl2', 'a', 'zh', 'zl1', 'zl2', 't'],
      'params_trans': [0, 0, 0, 0, 1, 1, 1, 0],
      'params_std_upper': [1.5, 1.5, 1.5, 1.0, None, None, None, 1.0],
      'param_bounds': [[-2.5, -2.5, -2.5, 0.3, 0.2, 0.2, 0.2, 0.0],
       [2.5, 2.5, 2.5, 2.0, 0.8, 0.8, 0.8, 2.0]],
      'boundary': <function ssms.basic_simulators.boundary_functions.constant(t=0)>,
      'params_default': [0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 1.0],
      'hddm_include': ['vh', 'vl1', 'vl2', 'a', 'zh', 'zl1', 'zl2', 't'],
      'choices': [0, 1, 2, 3],
      'slice_widths': {'vh': 1.5,
       'vh_std': 0.5,
       'vl1': 1.5,
       'vl1_std': 0.5,
       'vl2': 1.5,
       'vl2_std': 0.5,
       'a': 1,
       'a_std': 1,
       'zh': 0.1,
       'zh_trans': 0.2,
       'zl1': 0.1,
       'zl1_trans': 0.2,
       'zl2': 0.1,
       'zl2_trans': 0.2,
       't': 0.01,
       't_std': 0.15}},
     'ddm_seq2_no_bias': {'doc': 'Currently undocumented, in testing phase.',
      'params': ['vh', 'vl1', 'vl2', 'a', 't'],
      'param_bounds': [[-2.5, -2.5, -2.5, 0.3, 0.0], [2.5, 2.5, 2.5, 2.0, 2.0]],
      'params_trans': [0, 0, 0, 0, 0],
      'params_std_upper': [1.5, 1.5, 1.5, 1.0, 1.0],
      'boundary': <function ssms.basic_simulators.boundary_functions.constant(t=0)>,
      'params_default': [0.0, 0.0, 0.0, 1.0, 1.0],
      'hddm_include': ['vh', 'vl1', 'vl2', 'a', 't'],
      'choices': [0, 1, 2, 3],
      'slice_widths': {'vh': 1.5,
       'vh_std': 0.5,
       'vl1': 1.5,
       'vl1_std': 0.5,
       'vl2': 1.5,
       'vl2_std': 0.5,
       'a': 1,
       'a_std': 1,
       't': 0.01,
       't_std': 0.15}},
     'ddm_seq2_angle_no_bias': {'doc': 'Currently undocumented, in testing phase.',
      'params': ['vh', 'vl1', 'vl2', 'a', 't', 'theta'],
      'param_bounds': [[-2.5, -2.5, -2.5, 0.3, 0.0, -0.1],
       [2.5, 2.5, 2.5, 2.0, 2.0, 1.0]],
      'params_trans': [0, 0, 0, 0, 0, 0],
      'params_std_upper': [1.5, 1.5, 1.5, 1.0, 1.0, 1.0],
      'boundary': <function ssms.basic_simulators.boundary_functions.angle(t=1, theta=1)>,
      'boundary_multiplicative': False,
      'params_default': [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
      'hddm_include': ['vh', 'vl1', 'vl2', 'a', 't', 'theta'],
      'choices': [0, 1, 2, 3],
      'slice_widths': {'vh': 1.5,
       'vh_std': 0.5,
       'vl1': 1.5,
       'vl1_std': 0.5,
       'vl2': 1.5,
       'vl2_std': 0.5,
       'a': 1,
       'a_std': 1,
       't': 0.01,
       't_std': 0.15,
       'theta': 0.1,
       'theta_std': 0.2}},
     'ddm_seq2_weibull_no_bias': {'doc': 'Currently undocumented, in testing phase.',
      'params': ['vh', 'vl1', 'vl2', 'a', 't', 'alpha', 'beta'],
      'param_bounds': [[-2.5, -2.5, -2.5, 0.3, 0.0, 0.31, 0.31],
       [2.5, 2.5, 2.5, 2.0, 2.0, 4.99, 6.99]],
      'params_trans': [0, 0, 0, 0, 0, 0, 0],
      'params_std_upper': [1.5, 1.5, 1.5, 1.0, 1.0, 1.5, 1.5],
      'boundary': <function ssms.basic_simulators.boundary_functions.weibull_cdf(t=1, alpha=1, beta=1)>,
      'boundary_multiplicative': True,
      'params_default': [0.0, 0.0, 0.0, 1.0, 1.0, 2.5, 3.5],
      'hddm_include': ['vh', 'vl1', 'vl2', 'a', 't', 'alpha', 'beta'],
      'choices': [0, 1, 2, 3],
      'slice_widths': {'vh': 1.5,
       'vh_std': 0.5,
       'vl1': 1.5,
       'vl1_std': 0.5,
       'vl2': 1.5,
       'vl2_std': 0.5,
       'a': 1,
       'a_std': 1,
       't': 0.01,
       't_std': 0.15,
       'alpha': 1.0,
       'alpha_std': 0.5,
       'beta': 1.0,
       'beta_std': 0.5}},
     'ddm_mic2_adj': {'doc': 'Currently undocumented, in testing phase.',
      'params': ['vh', 'vl1', 'vl2', 'a', 'zh', 'zl1', 'zl2', 'd', 't'],
      'params_trans': [0, 0, 0, 0, 1, 1, 1, 1, 0],
      'params_std_upper': [1.5, 1.5, 1.5, 1.0, None, None, None, None, 1.0],
      'param_bounds': [[-2.5, -2.5, -2.5, 0.3, 0.2, 0.2, 0.2, 0.0, 0.0],
       [2.5, 2.5, 2.5, 2.0, 0.8, 0.8, 0.8, 1.0, 2.0]],
      'boundary': <function ssms.basic_simulators.boundary_functions.constant(t=0)>,
      'params_default': [0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5],
      'hddm_include': ['vh', 'vl1', 'vl2', 'a', 'zh', 'zl1', 'zl2', 'd', 't'],
      'choices': [0, 1, 2, 3],
      'slice_widths': {'vh': 1.5,
       'vh_std': 0.5,
       'vl1': 1.5,
       'vl1_std': 0.5,
       'vl2': 1.5,
       'vl2_std': 0.5,
       'a': 1,
       'a_std': 1,
       'zh': 0.1,
       'zh_trans': 0.2,
       'zl1': 0.1,
       'zl1_trans': 0.2,
       'zl2': 0.1,
       'zl2_trans': 0.2,
       'd': 0.1,
       'd_trans': 0.2,
       't': 0.01,
       't_std': 0.15}},
     'ddm_mic2_adj_no_bias': {'doc': 'Currently undocumented, in testing phase.',
      'params': ['vh', 'vl1', 'vl2', 'a', 'd', 't'],
      'param_bounds': [[-2.5, -2.5, -2.5, 0.3, 0.0, 0.0],
       [2.5, 2.5, 2.5, 2.0, 1.0, 2.0]],
      'params_trans': [0, 0, 0, 0, 0, 0],
      'params_std_upper': [1.5, 1.5, 1.5, 1.0, 1.0, 1.0],
      'boundary': <function ssms.basic_simulators.boundary_functions.constant(t=0)>,
      'params_default': [0.0, 0.0, 0.0, 1.0, 0.5, 1.0],
      'hddm_include': ['vh', 'vl1', 'vl2', 'a', 'd', 't'],
      'choices': [0, 1, 2, 3],
      'slice_widths': {'vh': 1.5,
       'vh_std': 0.5,
       'vl1': 1.5,
       'vl1_std': 0.5,
       'vl2': 1.5,
       'vl2_std': 0.5,
       'a': 1,
       'a_std': 1,
       'd': 0.1,
       'd_trans': 0.2,
       't': 0.01,
       't_std': 0.15}},
     'ddm_mic2_adj_angle_no_bias': {'doc': 'Currently undocumented, in testing phase.',
      'params': ['vh', 'vl1', 'vl2', 'a', 'd', 't', 'theta'],
      'param_bounds': [[-2.5, -2.5, -2.5, 0.3, 0.0, 0.0, -0.1],
       [2.5, 2.5, 2.5, 2.0, 1.0, 2.0, 1.0]],
      'params_trans': [0, 0, 0, 0, 0, 0, 0],
      'params_std_upper': [1.5, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0],
      'boundary': <function ssms.basic_simulators.boundary_functions.angle(t=1, theta=1)>,
      'boundary_multiplicative': False,
      'params_default': [0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0],
      'hddm_include': ['vh', 'vl1', 'vl2', 'a', 'd', 't', 'theta'],
      'choices': [0, 1, 2, 3],
      'slice_widths': {'vh': 1.5,
       'vh_std': 0.5,
       'vl1': 1.5,
       'vl1_std': 0.5,
       'vl2': 1.5,
       'vl2_std': 0.5,
       'a': 1,
       'a_std': 1,
       'd': 0.1,
       'd_trans': 0.2,
       't': 0.01,
       't_std': 0.15,
       'theta': 0.1,
       'theta_std': 0.2}},
     'ddm_mic2_adj_weibull_no_bias': {'doc': 'Currently undocumented, in testing phase.',
      'params': ['vh', 'vl1', 'vl2', 'a', 'd', 't', 'alpha', 'beta'],
      'param_bounds': [[-2.5, -2.5, -2.5, 0.3, 0.0, 0.0, 0.31, 0.31],
       [2.5, 2.5, 2.5, 2.0, 1.0, 2.0, 4.99, 6.99]],
      'params_trans': [0, 0, 0, 0, 0, 0, 0, 0],
      'params_std_upper': [1.5, 1.5, 1.5, 1.0, 1.0, 1.0, 1.5, 1.5],
      'boundary': <function ssms.basic_simulators.boundary_functions.weibull_cdf(t=1, alpha=1, beta=1)>,
      'boundary_multiplicative': True,
      'params_default': [0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 2.5, 3.5],
      'hddm_include': ['vh', 'vl1', 'vl2', 'a', 'd', 't', 'alpha', 'beta'],
      'choices': [0, 1, 2, 3],
      'slice_widths': {'vh': 1.5,
       'vh_std': 0.5,
       'vl1': 1.5,
       'vl1_std': 0.5,
       'vl2': 1.5,
       'vl2_std': 0.5,
       'a': 1,
       'a_std': 1,
       'd': 0.1,
       'd_trans': 0.2,
       't': 0.01,
       't_std': 0.15,
       'alpha': 1.0,
       'alpha_std': 0.5,
       'beta': 1.0,
       'beta_std': 0.5}},
     'tradeoff_no_bias': {'doc': 'Currently undocumented, in testing phase.',
      'params': ['vh', 'vl1', 'vl2', 'a', 'd', 't'],
      'params_trans': [0, 0, 0, 0, 0, 0],
      'param_bounds': [[-4.0, -4.0, -4.0, 0.3, 0.0, 0.0],
       [4.0, 4.0, 4.0, 2.5, 1.0, 2.0]],
      'params_std_upper': [1.5, 1.5, 1.5, 1.0, 1.0, 1.0],
      'boundary': <function ssms.basic_simulators.boundary_functions.constant(t=0)>,
      'n_params': 6,
      'default_params': [0.0, 0.0, 0.0, 1.0, 0.5, 1.0],
      'hddm_include': ['vh', 'vl1', 'vl2', 'a', 'd', 't'],
      'choices': [0, 1, 2, 3],
      'slice_widths': {'vh': 1.5,
       'vh_std': 0.5,
       'vl1': 1.5,
       'vl1_std': 0.5,
       'vl2': 1.5,
       'vl2_std': 0.5,
       'a': 1,
       'a_std': 0.1,
       'd': 0.1,
       'd_trans': 0.2,
       't': 0.01,
       't_std': 0.15}},
     'tradeoff_angle_no_bias': {'doc': 'Currently undocumented, in testing phase.',
      'params': ['vh', 'vl1', 'vl2', 'a', 'd', 't', 'theta'],
      'params_trans': [0, 0, 0, 0, 0, 0, 0],
      'param_bounds': [[-4.0, -4.0, -4.0, 0.3, 0.0, 0.0, -0.1],
       [4.0, 4.0, 4.0, 2.5, 1.0, 2.0, 1.0]],
      'params_std_upper': [1.5, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0],
      'boundary': <function ssms.basic_simulators.boundary_functions.angle(t=1, theta=1)>,
      'boundary_multiplicative': False,
      'n_params': 7,
      'default_params': [0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0],
      'hddm_include': ['vh', 'vl1', 'vl2', 'a', 'd', 't', 'theta'],
      'choices': [0, 1, 2, 3],
      'slice_widths': {'vh': 1.5,
       'vh_std': 0.5,
       'vl1': 1.5,
       'vl1_std': 0.5,
       'vl2': 1.5,
       'vl2_std': 0.5,
       'a': 1,
       'a_std': 0.1,
       'd': 0.1,
       'd_trans': 0.2,
       't': 0.01,
       't_std': 0.15,
       'theta': 0.1,
       'theta_std': 0.2}},
     'tradeoff_weibull_no_bias': {'doc': 'Currently undocumented, in testing phase.',
      'params': ['vh', 'vl1', 'vl2', 'a', 'd', 't', 'alpha', 'beta'],
      'params_trans': [0, 0, 0, 0, 0, 0, 0, 0],
      'param_bounds': [[-4.0, -4.0, -4.0, 0.3, 0.0, 0.0, 0.31, 0.31],
       [4.0, 4.0, 4.0, 2.5, 1.0, 2.0, 4.99, 6.99]],
      'params_std_upper': [1.5, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0],
      'boundary': <function ssms.basic_simulators.boundary_functions.weibull_cdf(t=1, alpha=1, beta=1)>,
      'boundary_multiplicative': True,
      'n_params': 8,
      'default_params': [0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 2.5, 3.5],
      'hddm_include': ['vh', 'vl1', 'vl2', 'a', 'd', 't', 'alpha', 'beta'],
      'choices': [0, 1, 2, 3],
      'slice_widths': {'vh': 1.5,
       'vh_std': 0.5,
       'vl1': 1.5,
       'vl1_std': 0.5,
       'vl2': 1.5,
       'vl2_std': 0.5,
       'a': 1,
       'a_std': 0.1,
       'd': 0.1,
       'd_trans': 0.2,
       't': 0.01,
       't_std': 0.15,
       'alpha': 1.0,
       'alpha_std': 0.5,
       'beta': 1.0,
       'beta_std': 0.5}},
     'tradeoff_conflict_gamma_no_bias': {'doc': 'Currently undocumented, in testing phase.',
      'params': ['vh',
       'vl1',
       'vl2',
       'd',
       't',
       'a',
       'theta',
       'scale',
       'alphagamma',
       'scalegamma'],
      'params_trans': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      'param_bounds': [[-4.0, -4.0, -4.0, 0.0, 0.0, 0.3, 0.0, 0.0, 1.1, 0.5],
       [4.0, 4.0, 4.0, 1.0, 2.0, 2.5, 0.5, 5.0, 5.0, 5.0]],
      'params_std_upper': [1.5, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      'boundary': <function ssms.basic_simulators.boundary_functions.conflict_gamma_bound(t=array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ,
            1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2. ,  2.1,
            2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  3. ,  3.1,  3.2,
            3.3,  3.4,  3.5,  3.6,  3.7,  3.8,  3.9,  4. ,  4.1,  4.2,  4.3,
            4.4,  4.5,  4.6,  4.7,  4.8,  4.9,  5. ,  5.1,  5.2,  5.3,  5.4,
            5.5,  5.6,  5.7,  5.8,  5.9,  6. ,  6.1,  6.2,  6.3,  6.4,  6.5,
            6.6,  6.7,  6.8,  6.9,  7. ,  7.1,  7.2,  7.3,  7.4,  7.5,  7.6,
            7.7,  7.8,  7.9,  8. ,  8.1,  8.2,  8.3,  8.4,  8.5,  8.6,  8.7,
            8.8,  8.9,  9. ,  9.1,  9.2,  9.3,  9.4,  9.5,  9.6,  9.7,  9.8,
            9.9, 10. , 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9,
           11. , 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12. ,
           12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13. , 13.1,
           13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 14. , 14.1, 14.2,
           14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15. , 15.1, 15.2, 15.3,
           15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 16. , 16.1, 16.2, 16.3, 16.4,
           16.5, 16.6, 16.7, 16.8, 16.9, 17. , 17.1, 17.2, 17.3, 17.4, 17.5,
           17.6, 17.7, 17.8, 17.9, 18. , 18.1, 18.2, 18.3, 18.4, 18.5, 18.6,
           18.7, 18.8, 18.9, 19. , 19.1, 19.2, 19.3, 19.4, 19.5, 19.6, 19.7,
           19.8, 19.9]), a=0.5, theta=0.5, scale=1, alpha_gamma=1.01, scale_gamma=0.3)>,
      'boundary_multiplicative': False,
      'n_params': 10,
      'default_params': [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
      'hddm_include': ['vh',
       'vl1',
       'vl2',
       'a',
       'd',
       't',
       'theta',
       'scale',
       'alphagamma',
       'scalegamma'],
      'choices': [0, 1, 2, 3],
      'slice_widths': {'vh': 1.5,
       'vh_std': 0.5,
       'vl1': 1.5,
       'vl1_std': 0.5,
       'vl2': 1.5,
       'vl2_std': 0.5,
       'a': 1,
       'a_std': 0.1,
       'd': 0.1,
       'd_trans': 0.2,
       't': 0.01,
       't_std': 0.15,
       'theta': 0.1,
       'theta_std': 0.2,
       'scale': 0.1,
       'scale_std': 0.2,
       'alphagamma': 0.1,
       'alphagamma_std': 0.2,
       'scalegamma': 0.1,
       'scalegamma_std': 0.2}},
     'race_no_bias_3': {'doc': 'To be used with the LAN extension. Note, we suggest to fix z instead here, since it is essentially \nredundant with the boundary separation parameter a. Future versions will drop z altogether.',
      'params': ['v0', 'v1', 'v2', 'a', 'z', 't'],
      'params_trans': [0, 0, 0, 0, 1, 0],
      'params_std_upper': [1.5, 1.5, 1.5, 1.0, None, 1.0],
      'param_bounds': [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
       [2.5, 2.5, 2.5, 3.0, 0.9, 2.0]],
      'boundary': <function ssms.basic_simulators.boundary_functions.constant(t=0)>,
      'params_default': [0.0, 0.0, 0.0, 2.0, 0.5, 0.001],
      'hddm_include': ['v0', 'v1', 'v2', 'a', 'z', 't'],
      'choices': [0, 1, 2],
      'slice_widths': {'v0': 1.5,
       'v0_std': 0.5,
       'v1': 1.5,
       'v1_std': 0.5,
       'v2': 1.5,
       'v2_std': 0.5,
       'a': 1,
       'a_std': 1,
       'z': 0.1,
       'z_trans': 0.2,
       't': 0.01,
       't_std': 0.15}},
     'race_no_bias_angle_3': {'doc': 'To be used with the LAN extension. Note, we suggest to fix z instead here, since it is essentially \nredundant with the boundary separation parameter a. Future versions will drop z altogether.',
      'params': ['v0', 'v1', 'v2', 'a', 'z', 't', 'theta'],
      'param_bounds': [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -0.1],
       [2.5, 2.5, 2.5, 3.0, 0.9, 2.0, 1.45]],
      'params_trans': [0, 0, 0, 0, 1, 0, 0],
      'params_std_upper': [1.5, 1.5, 1.5, 1.0, None, 1.0, 1.0],
      'boundary': <function ssms.basic_simulators.boundary_functions.angle(t=1, theta=1)>,
      'params_default': [0.0, 0.0, 0.0, 2.0, 0.5, 0.001, 0.0],
      'hddm_include': ['v0', 'v1', 'v2', 'a', 'z', 't', 'theta'],
      'choices': [0, 1, 2],
      'slice_widths': {'v0': 1.5,
       'v0_std': 0.5,
       'v1': 1.5,
       'v1_std': 0.5,
       'v2': 1.5,
       'v2_std': 0.5,
       'a': 1,
       'a_std': 1,
       'z': 0.1,
       'z_trans': 0.2,
       't': 0.01,
       't_std': 0.15,
       'theta': 0.1,
       'theta_std': 0.2}},
     'race_no_bias_4': {'doc': 'To be used with the LAN extension. Note, we suggest to fix z instead here, since it is essentially \nredundant with the boundary separation parameter a. Future versions will drop z altogether.',
      'params': ['v0', 'v1', 'v2', 'v3', 'a', 'z', 't'],
      'param_bounds': [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
       [2.5, 2.5, 2.5, 2.5, 3.0, 0.9, 2.0]],
      'params_std_upper': [1.5, 1.5, 1.5, 1.5, 1.0, None, 1.0],
      'params_trans': [0, 0, 0, 0, 0, 1, 0],
      'boundary': <function ssms.basic_simulators.boundary_functions.constant(t=0)>,
      'params_default': [0.0, 0.0, 0.0, 0.0, 2.0, 0.5, 0.001],
      'hddm_include': ['v0', 'v1', 'v2', 'v3', 'a', 'z', 't'],
      'choices': [0, 1, 2, 3],
      'slice_widths': {'v0': 1.5,
       'v0_std': 0.5,
       'v1': 1.5,
       'v1_std': 0.5,
       'v2': 1.5,
       'v2_std': 0.5,
       'v3': 1.5,
       'v3_std': 0.5,
       'a': 1,
       'a_std': 1,
       'z': 0.1,
       'z_trans': 0.2,
       't': 0.01,
       't_std': 0.15}},
     'race_no_bias_angle_4': {'doc': 'To be used with the LAN extension. Note, we suggest to fix z instead here, since it is essentially \nredundant with the boundary separation parameter a. Future versions will drop z altogether.',
      'params': ['v0', 'v1', 'v2', 'v3', 'a', 'z', 't', 'theta'],
      'param_bounds': [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -0.1],
       [2.5, 2.5, 2.5, 2.5, 3.0, 0.9, 2.0, 1.45]],
      'params_trans': [0, 0, 0, 0, 0, 1, 0, 0],
      'params_std_upper': [1.5, 1.5, 1.5, 1.5, 1.0, None, 1.0, 1.0],
      'boundary': <function ssms.basic_simulators.boundary_functions.angle(t=1, theta=1)>,
      'params_default': [0.0, 0.0, 0.0, 0.0, 2.0, 0.5, 0.001, 0.0],
      'hddm_include': ['v0', 'v1', 'v2', 'v3', 'a', 'z', 't', 'theta'],
      'choices': [0, 1, 2, 3],
      'slice_widths': {'v0': 1.5,
       'v0_std': 0.5,
       'v1': 1.5,
       'v1_std': 0.5,
       'v2': 1.5,
       'v2_std': 0.5,
       'v3': 1.5,
       'v3_std': 0.5,
       'a': 1,
       'a_std': 1,
       'z': 0.1,
       'z_trans': 0.2,
       't': 0.01,
       't_std': 0.15,
       'theta': 0.1,
       'theta_std': 0.2}},
     'lca_no_bias_3': {'doc': 'To be used with the LAN extension. Note, we suggest to fix z instead here, since it is essentially \nredundant with the boundary separation parameter a. Future versions will drop z altogether.',
      'params': ['v0', 'v1', 'v2', 'a', 'z', 'g', 'b', 't'],
      'param_bounds': [[0.0, 0.0, 0.0, 1.0, 0.0, -1.0, -1.0, 0.0],
       [2.5, 2.5, 2.5, 3.0, 0.9, 1.0, 1.0, 2.0]],
      'params_trans': [0, 0, 0, 0, 1, 0, 0, 0],
      'params_std_upper': [1.5, 1.5, 1.5, 1.0, None, 1.0, 1.0, 1.0],
      'boundary': <function ssms.basic_simulators.boundary_functions.constant(t=0)>,
      'params_default': [0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 0.001],
      'hddm_include': ['v0', 'v1', 'v2', 'a', 'z', 'g', 'b', 't'],
      'choices': [0, 1, 2],
      'slice_widths': {'v0': 1.5,
       'v0_std': 0.5,
       'v1': 1.5,
       'v1_std': 0.5,
       'v2': 1.5,
       'v2_std': 0.5,
       'a': 1,
       'a_std': 1,
       'z': 0.1,
       'z_trans': 0.2,
       't': 0.01,
       't_std': 0.15,
       'g': 0.1,
       'g_std': 0.2,
       'b': 0.1,
       'b_std': 0.2}},
     'lca_no_bias_angle_3': {'doc': 'To be used with the LAN extension. Note, we suggest to fix z instead here, since it is essentially \nredundant with the boundary separation parameter a. Future versions will drop z altogether.',
      'params': ['v0', 'v1', 'v2', 'a', 'z', 'g', 'b', 't', 'theta'],
      'param_bounds': [[0.0, 0.0, 0.0, 1.0, 0.0, -1.0, -1.0, 0.0, -1.0],
       [2.5, 2.5, 2.5, 3.0, 0.9, 1.0, 1.0, 2.0, 1.45]],
      'params_trans': [0, 0, 0, 0, 1, 0, 0, 0, 0],
      'params_std_upper': [1.5, 1.5, 1.5, 1.0, None, 1.5, 1.0, 1.0, 1.0],
      'boundary': <function ssms.basic_simulators.boundary_functions.angle(t=1, theta=1)>,
      'params_default': [0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 0.001, 0.0],
      'hddm_include': ['v0', 'v1', 'v2', 'a', 'z', 'g', 'b', 't', 'theta'],
      'choices': [0, 1, 2],
      'slice_widths': {'v0': 1.5,
       'v0_std': 0.5,
       'v1': 1.5,
       'v1_std': 0.5,
       'v2': 1.5,
       'v2_std': 0.5,
       'a': 1,
       'a_std': 1,
       'z': 0.1,
       'z_trans': 0.2,
       't': 0.01,
       't_std': 0.15,
       'g': 0.1,
       'g_std': 0.2,
       'b': 0.1,
       'b_std': 0.2,
       'theta': 0.1,
       'theta_std': 0.2}},
     'lca_no_bias_4': {'doc': 'To be used with the LAN extension. Note, we suggest to fix z instead here, since it is essentially \nredundant with the boundary separation parameter a. Future versions will drop z altogether.',
      'params': ['v0', 'v1', 'v2', 'v3', 'a', 'z', 'g', 'b', 't'],
      'param_bounds': [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, -1.0, 0.0],
       [2.5, 2.5, 2.5, 2.5, 3.0, 0.9, 1.0, 1.0, 2.0]],
      'params_trans': [0, 0, 0, 0, 0, 1, 0, 0, 0],
      'params_std_upper': [1.5, 1.5, 1.5, 1.5, 1.0, None, 1.5, 1.0, 1.0],
      'boundary': <function ssms.basic_simulators.boundary_functions.constant(t=0)>,
      'params_default': [0.0, 0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 0.001],
      'hddm_include': ['v0', 'v1', 'v2', 'v3', 'a', 'z', 'g', 'b', 't'],
      'choices': [0, 1, 2, 3],
      'slice_widths': {'v0': 1.5,
       'v0_std': 0.5,
       'v1': 1.5,
       'v1_std': 0.5,
       'v2': 1.5,
       'v2_std': 0.5,
       'v3': 1.5,
       'v3_std': 0.5,
       'a': 1,
       'a_std': 1,
       'z': 0.1,
       'z_trans': 0.2,
       't': 0.01,
       't_std': 0.15,
       'g': 0.1,
       'g_std': 0.2,
       'b': 0.1,
       'b_std': 0.2}},
     'lca_no_bias_angle_4': {'doc': 'To be used with the LAN extension. Note, we suggest to fix z instead here, since it is essentially \nredundant with the boundary separation parameter a. Future versions will drop z altogether.',
      'params': ['v0', 'v1', 'v2', 'v3', 'a', 'z', 'g', 'b', 't', 'theta'],
      'param_bounds': [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, -1.0, 0.0, -0.1],
       [2.5, 2.5, 2.5, 2.5, 3.0, 0.9, 1.0, 1.0, 2.0, 1.45]],
      'params_trans': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
      'params_std_upper': [1.5, 1.5, 1.5, 1.5, 1.0, None, 1.5, 1.0, 1.0, 1.0],
      'boundary': <function ssms.basic_simulators.boundary_functions.angle(t=1, theta=1)>,
      'params_default': [0.0, 0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 0.001, 0.0],
      'hddm_include': ['v0', 'v1', 'v2', 'v3', 'a', 'z', 'g', 'b', 't', 'theta'],
      'choices': [0, 1, 2, 3],
      'slice_widths': {'v0': 1.5,
       'v0_std': 0.5,
       'v1': 1.5,
       'v1_std': 0.5,
       'v2': 1.5,
       'v2_std': 0.5,
       'v3': 1.5,
       'v3_std': 0.5,
       'a': 1,
       'a_std': 1,
       'z': 0.1,
       'z_trans': 0.2,
       't': 0.01,
       't_std': 0.15,
       'g': 0.1,
       'g_std': 0.2,
       'b': 0.1,
       'b_std': 0.2,
       'theta': 0.1,
       'theta_std': 0.2}},
     'full_ddm2': {'doc': 'Currently unavailable, for LANs after switch to pytorch. \nComing soon... Please use standard HDDM if you want to fit this model to your data.',
      'params': ['v', 'a', 'z', 't', 'sz', 'sv', 'st'],
      'params_trans': [0, 0, 1, 0, 0, 0, 0],
      'params_std_upper': [1.5, 1.0, None, 1.0, 0.1, 0.5, 0.1],
      'param_bounds': [[-3.0, 0.3, 0.3, 0.25, 0.001, 0.001, 0.001],
       [3.0, 2.5, 0.7, 2.25, 0.2, 2.0, 0.25]],
      'boundary': <function ssms.basic_simulators.boundary_functions.constant(t=0)>,
      'params_default': [0.0, 1.0, 0.5, 0.25, 0.001, 0.001, 0.001],
      'hddm_include': ['v', 'a', 't', 'z', 'st', 'sv', 'sz'],
      'choices': [-1, 1],
      'slice_widths': {'v': 1.5,
       'v_std': 1,
       'a': 1,
       'a_std': 1,
       'z': 0.1,
       'z_trans': 0.2,
       't': 0.01,
       't_std': 0.15,
       'sz': 1.1,
       'st': 0.1,
       'sv': 0.5}}}



.. code:: ipython3

    model_angle_no_theta = hddm.HDDMnn(data_angle,
                                       model = 'angle',
                                       include = ['v', 'a', 't', 'z'])


.. parsed-literal::

    Using default priors: Uninformative
    Supplied model_config specifies params_std_upper for  z as  None.
    Changed to 10


.. code:: ipython3

    model_angle_no_theta.sample(1000, burn = 500)


.. parsed-literal::

     [-----------------100%-----------------] 1000 of 1000 complete in 47.4 sec



.. parsed-literal::

    <pymc.MCMC.MCMC at 0x7ff30160bed0>



.. code:: ipython3

    model_angle_no_theta.gen_stats()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>mean</th>
          <th>std</th>
          <th>2.5q</th>
          <th>25q</th>
          <th>50q</th>
          <th>75q</th>
          <th>97.5q</th>
          <th>mc err</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>v</th>
          <td>1.124278</td>
          <td>0.09761</td>
          <td>0.928012</td>
          <td>1.052883</td>
          <td>1.128692</td>
          <td>1.191886</td>
          <td>1.310813</td>
          <td>0.008418</td>
        </tr>
        <tr>
          <th>a</th>
          <td>1.363875</td>
          <td>0.053492</td>
          <td>1.259736</td>
          <td>1.327325</td>
          <td>1.364278</td>
          <td>1.396537</td>
          <td>1.468327</td>
          <td>0.004211</td>
        </tr>
        <tr>
          <th>z</th>
          <td>0.49972</td>
          <td>0.030672</td>
          <td>0.436555</td>
          <td>0.477589</td>
          <td>0.498887</td>
          <td>0.522161</td>
          <td>0.558534</td>
          <td>0.002693</td>
        </tr>
        <tr>
          <th>t</th>
          <td>0.536048</td>
          <td>0.033127</td>
          <td>0.464361</td>
          <td>0.515404</td>
          <td>0.53657</td>
          <td>0.558055</td>
          <td>0.595229</td>
          <td>0.002959</td>
        </tr>
      </tbody>
    </table>
    </div>



Again we observe how the parameter estimates are affected by the *wrong
choice of ``theta``. The model tries to compensate for the parallel
bounds (no collapse), implied by the ``theta`` default, by decreasing
``a`` and slightly increasing ``v``. Let’s now try again, but this time
we set ``theta`` fixed to the actual*\ ground truth*.

.. code:: ipython3

    # copy out the model_config dictionary for the angle model
    my_model_config_angle = deepcopy(hddm.model_config.model_config['angle'])
    # set theta default to the ground truth defined above
    my_model_config_angle['params_default'][4] = 0.2
    
    model_angle_no_theta_2 = hddm.HDDMnn(data_angle,
                                         model = 'angle',
                                         include = ['v', 'a', 't', 'z'],
                                         model_config = my_model_config_angle)


.. parsed-literal::

    Using default priors: Uninformative
    Supplied model_config specifies params_std_upper for  z as  None.
    Changed to 10


.. code:: ipython3

    model_angle_no_theta_2.sample(1000, burn = 500)


.. parsed-literal::

     [-----------------100%-----------------] 1000 of 1000 complete in 53.4 sec



.. parsed-literal::

    <pymc.MCMC.MCMC at 0x7ff301652c90>



.. code:: ipython3

    model_angle_no_theta_2.gen_stats()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>mean</th>
          <th>std</th>
          <th>2.5q</th>
          <th>25q</th>
          <th>50q</th>
          <th>75q</th>
          <th>97.5q</th>
          <th>mc err</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>v</th>
          <td>1.020949</td>
          <td>0.089617</td>
          <td>0.838443</td>
          <td>0.959232</td>
          <td>1.019338</td>
          <td>1.087935</td>
          <td>1.188067</td>
          <td>0.007229</td>
        </tr>
        <tr>
          <th>a</th>
          <td>1.46397</td>
          <td>0.049143</td>
          <td>1.363087</td>
          <td>1.429878</td>
          <td>1.465694</td>
          <td>1.49972</td>
          <td>1.554566</td>
          <td>0.003623</td>
        </tr>
        <tr>
          <th>z</th>
          <td>0.527111</td>
          <td>0.027281</td>
          <td>0.471521</td>
          <td>0.508124</td>
          <td>0.527366</td>
          <td>0.547015</td>
          <td>0.581392</td>
          <td>0.002391</td>
        </tr>
        <tr>
          <th>t</th>
          <td>0.528561</td>
          <td>0.029786</td>
          <td>0.470851</td>
          <td>0.50755</td>
          <td>0.527586</td>
          <td>0.547936</td>
          <td>0.588767</td>
          <td>0.002522</td>
        </tr>
      </tbody>
    </table>
    </div>



As we see, fixing ``theta`` to the actual ground truth, corrects the
parameter estimates of the remaining parameters to be much more accurate
again.

Let’s compare DICs
''''''''''''''''''

.. code:: ipython3

    print('Standard: ', model_angle.dic)
    print('theta set to model_config default: ', model_angle_no_theta.dic)
    print('theta set to ground truth: ', model_angle_no_theta_2.dic)


.. parsed-literal::

    Standard:  1059.453694824219
    theta set to model_config default:  1066.945202636719
    theta set to ground truth:  1058.248090332031


We observe in this case, that fixing ``theta`` to ``0`` instead of
``0.2``, didn’t do too much damage as far as the DICs are concerned.
Nevertheless, the *explicitly wrong* model performs worst as per this
metric.

END
'''

Hopefully this was helpful.
