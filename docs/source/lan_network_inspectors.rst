.. index:: LANs
.. _chap_network_inspectors:

Network Inspectors
------------------

The ``network_inspectors()`` module allows you to inspect the LANs
directly. We will be grateful if you report any strange behavior you
might find.

.. code:: ipython3

    # MODULE IMPORTS ----
    import numpy as np
    import hddm

Direct access to batch predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use the ``hddm.network_inspectors.get_torch_mlp()`` function to
access network predictions.

.. code:: ipython3

    # Specify model
    model = 'angle'
    lan_angle = hddm.network_inspectors.get_torch_mlp(model = model)

.. code:: ipython3

    # Make some random parameter set
    parameter_df = hddm.simulators.make_parameter_vectors_nn(model = model,
                                                             param_dict = None,
                                                             n_parameter_vectors = 1)
    
    parameter_matrix = np.tile(np.squeeze(parameter_df.values), (200, 1))
    
    # Initialize network input
    network_input = np.zeros((parameter_matrix.shape[0], parameter_matrix.shape[1] + 2)) # Note the + 2 on the right --> we append the parameter vectors with reaction times (+1 columns) and choices (+1 columns)
    
    # Add reaction times
    network_input[:, -2] = np.linspace(0, 3, parameter_matrix.shape[0])
    
    # Add choices
    network_input[:, -1] = np.repeat(np.random.choice([-1, 1]), parameter_matrix.shape[0])
    
    # Note: The networks expects float32 inputs
    network_input = network_input.astype(np.float32)
    
    # Show example output
    print('Some network outputs')
    print(lan_angle(network_input)[:10]) # printing the first 10 outputs
    print('Shape')
    print(lan_angle(network_input).shape) # original shape of output


.. parsed-literal::

    Some network outputs
    [[-6.5302672 ]
     [ 0.5264376 ]
     [ 0.41089576]
     [-0.5228093 ]
     [-1.0521748 ]
     [-1.5529907 ]
     [-2.0735157 ]
     [-2.6183674 ]
     [-3.2071757 ]
     [-3.8784742 ]]
    Shape
    (200, 1)


Plotting Utilities
~~~~~~~~~~~~~~~~~~

HDDM provides two plotting function to investigate the network outputs
directly. The ``kde_vs_lan_likelihoods()`` plot and the
``lan_manifold()`` plot.

``kde_vs_lan_likelihoods()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    # Make some parameters
    parameter_df = hddm.simulators.make_parameter_vectors_nn(model = model,
                                                             param_dict = None,
                                                             n_parameter_vectors = 10)

.. code:: ipython3

    parameter_df




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
          <th>v</th>
          <th>a</th>
          <th>z</th>
          <th>t</th>
          <th>theta</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>-0.365440</td>
          <td>0.810500</td>
          <td>0.600011</td>
          <td>1.799593</td>
          <td>0.659474</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.416247</td>
          <td>1.178634</td>
          <td>0.754132</td>
          <td>1.730125</td>
          <td>1.405258</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-2.297998</td>
          <td>0.509990</td>
          <td>0.253447</td>
          <td>1.985974</td>
          <td>0.974159</td>
        </tr>
        <tr>
          <th>3</th>
          <td>-1.233046</td>
          <td>0.912007</td>
          <td>0.449142</td>
          <td>0.523020</td>
          <td>0.606210</td>
        </tr>
        <tr>
          <th>4</th>
          <td>-0.257579</td>
          <td>1.377592</td>
          <td>0.558203</td>
          <td>0.349555</td>
          <td>1.085357</td>
        </tr>
        <tr>
          <th>5</th>
          <td>-2.359525</td>
          <td>0.828887</td>
          <td>0.393902</td>
          <td>1.459226</td>
          <td>0.277007</td>
        </tr>
        <tr>
          <th>6</th>
          <td>-1.324190</td>
          <td>0.486667</td>
          <td>0.235727</td>
          <td>1.052526</td>
          <td>1.188342</td>
        </tr>
        <tr>
          <th>7</th>
          <td>2.530732</td>
          <td>0.943623</td>
          <td>0.777100</td>
          <td>1.410220</td>
          <td>1.400012</td>
        </tr>
        <tr>
          <th>8</th>
          <td>-2.448473</td>
          <td>1.042435</td>
          <td>0.690794</td>
          <td>1.097246</td>
          <td>0.326023</td>
        </tr>
        <tr>
          <th>9</th>
          <td>-2.102524</td>
          <td>0.466584</td>
          <td>0.611397</td>
          <td>0.191337</td>
          <td>0.566287</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    hddm.network_inspectors.kde_vs_lan_likelihoods(parameter_df = parameter_df, 
                                                   model = model,
                                                   cols = 3,
                                                   n_samples = 2000,
                                                   n_reps = 2,
                                                   show = True)


.. parsed-literal::

    1 of 10
    2 of 10
    3 of 10
    4 of 10
    5 of 10
    6 of 10
    7 of 10
    8 of 10
    9 of 10
    10 of 10



.. image:: lan_network_inspectors_files/lan_network_inspectors_12_1.png


``lan_manifold()``
^^^^^^^^^^^^^^^^^^

Lastly, you can use the ``lan_manifold()`` plot to investigate the LAN
likelihoods over a range of parameters.

The idea is to use a base parameter vector and vary one of the
parameters in a prespecificed range.

This plot can be informative if you would like to understand better how
a parameter affects model behavior.

.. code:: ipython3

    # Now plotting
    hddm.network_inspectors.lan_manifold(parameter_df = parameter_df,
                                         vary_dict = {'v': np.linspace(-2, 2, 20)},
                                         model = model,
                                         n_rt_steps = 300,
                                         fig_scale = 1.0,
                                         max_rt = 5,
                                         save = True,
                                         show = True)


.. parsed-literal::

    Using only the first row of the supplied parameter array !



.. image:: lan_network_inspectors_files/lan_network_inspectors_15_1.png

