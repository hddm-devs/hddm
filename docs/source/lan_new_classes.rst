New Classes
-----------

The **LAN extension (HDDM >= 0.9.0)**, provides three new classes which
are LAN-enabled versions of the respective classes in base HDDM. These
new classes are,

-  The ``HDDMnn()`` class
-  The ``HDDMnnStimCoding()`` class
-  The ``HDDMnnRegressor()`` class

The usage mirrors what you are used to from standard ``HDDM``
equivalents.

What changes is that you now use the ``model`` argument to specify one
of the models you find listed in the ``hddm.model_config.model_config``
dictionary (you can also provide a custom model, for which you should
look into the respective section in this documentation).

Moreover, you have to be a little more careful when specifying the
``include`` argument, since the ability to use new models comes with new
parameters. To help get started here, the
``hddm.model_config.model_config`` dictionary provides you a
``hddm_include`` key for *every* model-specific sub-dictionary. This
let’s you fit all parameters of a given model. To keep some parameters
fixed, remove them respectively from the resulting list.

Short example
~~~~~~~~~~~~~

.. code:: ipython3

    import hddm

.. code:: ipython3

    model = 'angle'
    cavanagh_data = hddm.load_csv(hddm.__path__[0] + '/examples/cavanagh_theta_nn.csv')
    model_ = hddm.HDDMnn(cavanagh_data,
                         model = model,
                         include = hddm.model_config.model_config[model]['hddm_include'],
                         is_group_model = False)


.. parsed-literal::

    Using default priors: Uninformative


.. code:: ipython3

    model_.sample(1000, burn = 200)


.. parsed-literal::

     [-----------------100%-----------------] 1000 of 1000 complete in 260.3 sec



.. parsed-literal::

    <pymc.MCMC.MCMC at 0x13e449650>



.. code:: ipython3

    model_.get_traces()




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
          <th>z_trans</th>
          <th>t</th>
          <th>theta</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.370402</td>
          <td>1.325747</td>
          <td>0.023242</td>
          <td>0.284196</td>
          <td>0.253870</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.338917</td>
          <td>1.328545</td>
          <td>0.062895</td>
          <td>0.283047</td>
          <td>0.248485</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.386179</td>
          <td>1.321476</td>
          <td>0.054727</td>
          <td>0.285712</td>
          <td>0.250671</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.387484</td>
          <td>1.323711</td>
          <td>-0.019109</td>
          <td>0.274198</td>
          <td>0.253445</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.370557</td>
          <td>1.323342</td>
          <td>0.015675</td>
          <td>0.277691</td>
          <td>0.255681</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>795</th>
          <td>0.325748</td>
          <td>1.331846</td>
          <td>0.113685</td>
          <td>0.270311</td>
          <td>0.252461</td>
        </tr>
        <tr>
          <th>796</th>
          <td>0.337564</td>
          <td>1.315446</td>
          <td>0.111898</td>
          <td>0.286141</td>
          <td>0.252236</td>
        </tr>
        <tr>
          <th>797</th>
          <td>0.387142</td>
          <td>1.309284</td>
          <td>0.036839</td>
          <td>0.286663</td>
          <td>0.238878</td>
        </tr>
        <tr>
          <th>798</th>
          <td>0.388073</td>
          <td>1.313791</td>
          <td>-0.013604</td>
          <td>0.271768</td>
          <td>0.235831</td>
        </tr>
        <tr>
          <th>799</th>
          <td>0.397477</td>
          <td>1.314008</td>
          <td>-0.007186</td>
          <td>0.276948</td>
          <td>0.242729</td>
        </tr>
      </tbody>
    </table>
    <p>800 rows × 5 columns</p>
    </div>



.. code:: ipython3

    model_.gen_stats()




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
          <td>0.369154</td>
          <td>0.0207375</td>
          <td>0.329893</td>
          <td>0.355813</td>
          <td>0.369495</td>
          <td>0.382592</td>
          <td>0.409568</td>
          <td>0.00111918</td>
        </tr>
        <tr>
          <th>a</th>
          <td>1.31224</td>
          <td>0.0212032</td>
          <td>1.26826</td>
          <td>1.29879</td>
          <td>1.31332</td>
          <td>1.32755</td>
          <td>1.3514</td>
          <td>0.00180178</td>
        </tr>
        <tr>
          <th>z</th>
          <td>0.504951</td>
          <td>0.00604908</td>
          <td>0.493251</td>
          <td>0.500775</td>
          <td>0.504934</td>
          <td>0.509041</td>
          <td>0.517023</td>
          <td>0.000311986</td>
        </tr>
        <tr>
          <th>t</th>
          <td>0.283719</td>
          <td>0.00943542</td>
          <td>0.265774</td>
          <td>0.277639</td>
          <td>0.283707</td>
          <td>0.290191</td>
          <td>0.302331</td>
          <td>0.00070058</td>
        </tr>
        <tr>
          <th>theta</th>
          <td>0.242432</td>
          <td>0.0127552</td>
          <td>0.216824</td>
          <td>0.234284</td>
          <td>0.242875</td>
          <td>0.251645</td>
          <td>0.265587</td>
          <td>0.00103379</td>
        </tr>
      </tbody>
    </table>
    </div>



