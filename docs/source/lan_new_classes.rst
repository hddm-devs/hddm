.. index:: LANs
.. _chap_new_classes:

New Classes
--------------------------

The **LAN extension (HDDM >= 0.9.0)**, provides three new classes which
are LAN-enabled versions of the respective classes in base HDDM. These
new classes are,

-  The ``HDDMnn()`` class
-  The ``HDDMnnStimCoding()`` class
-  The ``HDDMnnRegressor()`` class

The usage mirrors what you are used to from standard HDDM equivalents.

What changes is that you now use the ``model`` argument to specify one
of the models you find listed in the ``hddm.model_config.model_config``
dictionary (you can also provide a custom model, for which you should
look into the respective section in this documentation).

Moreover, you have to be a little more careful when specifying the
``include`` argument, since the ability to use new models comes with new
parameters. To help get started here, the
``hddm.model_config.model_config`` dictionary provides you a
``hddm_include`` key *for every model-specific sub-dictionary*. This
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

    Setting priors uninformative (LANs only work with uninformative priors for now)
    Includes supplied:  ['z', 'theta']


.. code:: ipython3

    model_.sample(1000, burn = 200)


.. parsed-literal::

     [-----------------100%-----------------] 1001 of 1000 complete in 22.0 sec



.. parsed-literal::

    <pymc.MCMC.MCMC at 0x7f044c3a4550>



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
          <td>0.362657</td>
          <td>1.279809</td>
          <td>0.100552</td>
          <td>0.293367</td>
          <td>0.219623</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.347255</td>
          <td>1.265401</td>
          <td>0.065679</td>
          <td>0.303313</td>
          <td>0.217390</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.353600</td>
          <td>1.260157</td>
          <td>0.074713</td>
          <td>0.297913</td>
          <td>0.213457</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.327564</td>
          <td>1.261381</td>
          <td>0.043989</td>
          <td>0.309488</td>
          <td>0.216046</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.363990</td>
          <td>1.274243</td>
          <td>0.030869</td>
          <td>0.295772</td>
          <td>0.221900</td>
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
          <td>0.369220</td>
          <td>1.322772</td>
          <td>0.013902</td>
          <td>0.287593</td>
          <td>0.242824</td>
        </tr>
        <tr>
          <th>796</th>
          <td>0.383198</td>
          <td>1.300787</td>
          <td>0.018496</td>
          <td>0.281645</td>
          <td>0.230118</td>
        </tr>
        <tr>
          <th>797</th>
          <td>0.352363</td>
          <td>1.298063</td>
          <td>-0.004444</td>
          <td>0.283756</td>
          <td>0.235838</td>
        </tr>
        <tr>
          <th>798</th>
          <td>0.355785</td>
          <td>1.303034</td>
          <td>0.032976</td>
          <td>0.285204</td>
          <td>0.232276</td>
        </tr>
        <tr>
          <th>799</th>
          <td>0.368346</td>
          <td>1.295958</td>
          <td>0.038444</td>
          <td>0.286766</td>
          <td>0.230839</td>
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
          <td>0.367355</td>
          <td>0.022102</td>
          <td>0.325831</td>
          <td>0.352295</td>
          <td>0.367635</td>
          <td>0.383658</td>
          <td>0.409726</td>
          <td>0.001155</td>
        </tr>
        <tr>
          <th>a</th>
          <td>1.310485</td>
          <td>0.023723</td>
          <td>1.264522</td>
          <td>1.295559</td>
          <td>1.309213</td>
          <td>1.325134</td>
          <td>1.361408</td>
          <td>0.002074</td>
        </tr>
        <tr>
          <th>z</th>
          <td>0.505486</td>
          <td>0.005988</td>
          <td>0.493992</td>
          <td>0.501693</td>
          <td>0.505369</td>
          <td>0.509463</td>
          <td>0.517051</td>
          <td>0.00034</td>
        </tr>
        <tr>
          <th>t</th>
          <td>0.285347</td>
          <td>0.009728</td>
          <td>0.266361</td>
          <td>0.279472</td>
          <td>0.28532</td>
          <td>0.291927</td>
          <td>0.304614</td>
          <td>0.00077</td>
        </tr>
        <tr>
          <th>theta</th>
          <td>0.242268</td>
          <td>0.014679</td>
          <td>0.214883</td>
          <td>0.233054</td>
          <td>0.241219</td>
          <td>0.252035</td>
          <td>0.273616</td>
          <td>0.001201</td>
        </tr>
      </tbody>
    </table>
    </div>



