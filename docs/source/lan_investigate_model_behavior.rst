Investigate Model Behavior
--------------------------

.. code:: ipython3

    # MODULE IMPORTS ----
    import hddm

From version HDDM >= 0.9.0, you have access to multiple **new sequential
sampling models**. You can simulate from these models, perform parameter
estimation and moreover you have some extended plotting capabilities
which can be useful to visualize model fits, or simply to investigate
the behavior of models across parameter settings.

Metadata
~~~~~~~~

Lets take a look at the new ``hddm.model_config.model_config``
dictionary, which allows you to investigate metadata for all the new
(and old) models which are available through the HDDM-LAN extension.

.. code:: ipython3

    # List all available models
    list(hddm.model_config.model_config.keys())[:10]




.. parsed-literal::

    ['ddm_hddm_base',
     'full_ddm_hddm_base',
     'ddm',
     'angle',
     'weibull',
     'levy',
     'full_ddm',
     'ornstein',
     'ddm_sdv',
     'ddm_par2']



.. code:: ipython3

    # Take an example to list data available for a given model
    model_tmp = 'ornstein'
    hddm.model_config.model_config[model_tmp]




.. parsed-literal::

    {'doc': 'Model formulation is described in the documentation under LAN Extension.Meant for use with the extension.',
     'params': ['v', 'a', 'z', 'g', 't'],
     'params_trans': [0, 0, 1, 0, 0],
     'params_std_upper': [1.5, 1.0, None, 1.0, 1.0],
     'param_bounds': [[-2.0, 0.3, 0.2, -1.0, 0.001], [2.0, 2.0, 0.8, 1.0, 2]],
     'boundary': <function hddm.simulators.boundary_functions.constant(t=0)>,
     'params_default': [0.0, 1.0, 0.5, 0.0, 0.001],
     'hddm_include': ['z', 'g'],
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
      'g_std': 0.1}}



You have access to the following data (we focus on the parts important
for the user):

-  ``params``, the names of paramaters for a given model (order matters)
-  ``params_trans`` whether HDDM should internally transform a parameter
   to an unconstrained domain
-  ``param_bounds`` the range of parameter values that the respective
   LAN was trained on (order as in ``params``)
-  ``boundary`` the boundary function, which corresponds to the model
   (access the available boundary functions through the
   ``hddm.simulators.boundary_functions`` module.
-  ``params_default``, defaults settings for the parameters of the model
-  ``hddm_include``, list to supply to hddm to include all model
   parameters (you may want to drop some)
-  ``slide_widths``, slice sampler settings parameter by parameter
   (changing these can improve / deteriorate sampler behavior)

You can change these settings as you see fit.

Simulate
~~~~~~~~

The new ``simulator_h_c()`` function lets you generate complex datasets
using the models available under ``hddm.model_config.model_config``. The
function is especially useful for parameter recovery studies. It can
generate fully synthetic data, or you can supply an empirial dataset and
it’s structure can be used to generate simulation based replicas. Find
more information using the ``help()`` function. Here we give a simple
example.

.. code:: ipython3

    # test regressors only False
    # add p_outliers to the generator !
    model = 'angle'
    n_subjects = 1
    n_samples_by_subject = 500
    
    data, full_parameter_dict = hddm.simulators.hddm_dataset_generators.simulator_h_c(n_subjects = n_subjects,
                                                                                      n_samples_by_subject = n_samples_by_subject,
                                                                                      model = model,
                                                                                      p_outlier = 0.00,
                                                                                      conditions = None, 
                                                                                      depends_on = None, 
                                                                                      regression_models = None,
                                                                                      regression_covariates = None,
                                                                                      group_only_regressors = False,
                                                                                      group_only = None,
                                                                                      fixed_at_default = None)

.. code:: ipython3

    # A look at the data generated
    data




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
          <th>rt</th>
          <th>response</th>
          <th>subj_idx</th>
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
          <td>0.770753</td>
          <td>1.0</td>
          <td>0</td>
          <td>0.380409</td>
          <td>0.839763</td>
          <td>0.479834</td>
          <td>0.515753</td>
          <td>0.959683</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.812753</td>
          <td>1.0</td>
          <td>0</td>
          <td>0.380409</td>
          <td>0.839763</td>
          <td>0.479834</td>
          <td>0.515753</td>
          <td>0.959683</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.707753</td>
          <td>0.0</td>
          <td>0</td>
          <td>0.380409</td>
          <td>0.839763</td>
          <td>0.479834</td>
          <td>0.515753</td>
          <td>0.959683</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.616753</td>
          <td>1.0</td>
          <td>0</td>
          <td>0.380409</td>
          <td>0.839763</td>
          <td>0.479834</td>
          <td>0.515753</td>
          <td>0.959683</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.746753</td>
          <td>1.0</td>
          <td>0</td>
          <td>0.380409</td>
          <td>0.839763</td>
          <td>0.479834</td>
          <td>0.515753</td>
          <td>0.959683</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>95</th>
          <td>0.696753</td>
          <td>0.0</td>
          <td>0</td>
          <td>0.380409</td>
          <td>0.839763</td>
          <td>0.479834</td>
          <td>0.515753</td>
          <td>0.959683</td>
        </tr>
        <tr>
          <th>96</th>
          <td>0.778753</td>
          <td>1.0</td>
          <td>0</td>
          <td>0.380409</td>
          <td>0.839763</td>
          <td>0.479834</td>
          <td>0.515753</td>
          <td>0.959683</td>
        </tr>
        <tr>
          <th>97</th>
          <td>0.708753</td>
          <td>1.0</td>
          <td>0</td>
          <td>0.380409</td>
          <td>0.839763</td>
          <td>0.479834</td>
          <td>0.515753</td>
          <td>0.959683</td>
        </tr>
        <tr>
          <th>98</th>
          <td>0.848752</td>
          <td>0.0</td>
          <td>0</td>
          <td>0.380409</td>
          <td>0.839763</td>
          <td>0.479834</td>
          <td>0.515753</td>
          <td>0.959683</td>
        </tr>
        <tr>
          <th>99</th>
          <td>0.817753</td>
          <td>1.0</td>
          <td>0</td>
          <td>0.380409</td>
          <td>0.839763</td>
          <td>0.479834</td>
          <td>0.515753</td>
          <td>0.959683</td>
        </tr>
      </tbody>
    </table>
    <p>100 rows × 8 columns</p>
    </div>



The ``full_parameter_dic`` returned plays well with ``HDDM`` and some
plots that give you the option to provide ground truth parameters. In
our case the output is simple. More complicated datasets, will make this
much more interesting.

.. code:: ipython3

    full_parameter_dict




.. parsed-literal::

    {'z': 0.4798341238342858,
     'theta': 0.9596826258981702,
     'v': 0.3804088421388936,
     't': 0.51575296339607,
     'a': 0.8397625801445228}


