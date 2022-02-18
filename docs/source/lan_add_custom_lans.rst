Use your Likelihood Function
----------------------------

If you have your own LAN, or any class with a ``predict_on_batch()``
method which you can call to get back the log-likelihood of a dataset,
you can pass it as an argument to the ``HDDMnn``, ``HDDMnnRegressor`` or
``HDDMnnStimCoding`` classes.

In this document we provide you with a simple complete example.

Such a new model will be called ``custom``, for our purposes. Let’s say
we trained a LAN for it.

We need two components to use our ``custom`` LAN.

1. A config dictionary for the ``custom`` model.
2. A pretrained LAN with a ``predict_on_batch`` method.

Construct config dictionary
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import hddm

Access the config dictionary for all models included in ``HDDM`` by
calling the ``hddm.model_config.model_config`` dictionary. Learn more
about this object in **lan tutorial concerning new models**.

We now generate a custom ``model_config`` for our new model. Let’s start
with the minimal ``dictionary`` that you need to befine.

.. code:: ipython3

    my_model_config = {}
    
    # parameter names associated to your model 
    my_model_config["params"] =  ["v", "a", "z", "t"]
    
    # the parameter boundaries you used for training your LAN
    my_model_config["param_bounds"] = [[-3.0, 0.3, 0.1, 1e-3], 
                                       [3.0, 2.5, 0.9, 2.0]]
    
    # add a boundary function
    my_model_config["boundary"] = hddm.simulators.boundary_functions.constant
    
    # suggestion for which parameters to include
    # via the include statement of an HDDM model 
    # (usually you want all of the parameters from above)
    my_model_config["hddm_include"] = ["z"]
    
    # choice labels (what your simulator spits out)
    my_model_config["choices"] = [-1, 1]

Now we can add extra options, useful for example to improve performance
of the sampler.

.. code:: ipython3

    # specifies parameters which the sampler should tansform 
    my_model_config['params_trans'] = [0, 0, 1, 0] 
    
    # adds sampler settings for each parameter
    my_model_config['slice_widths'] = {"v": 1.5, "v_std": 1,  
                                       "a": 1, "a_std": 1, 
                                       "z": 0.1, "z_trans": 0.2, 
                                       "t": 0.01, "t_std": 0.15}
    
    # set sampler starting points manually for each parameter
    my_model_config['params_default'] = [0.0, 1.0, 0.5, 1e-3] 
    
    # set a (reasonable) upper limit of group level standard deviations,
    # this can help with sampler stability 
    my_model_config['params_std_upper'] = [1.5, 1.0, None, 1.0] 

Load Network
~~~~~~~~~~~~

To make the example complete, here is a code snippet to load in a
``Keras`` model. We are simply using the ``ddm`` network here.

.. code:: ipython3

    custom_network = hddm.torch.mlp_inference_class.load_torch_mlp(model = 'ddm')  
    # any class with a valid predict on batch function works here

**NOTE:**

Above, for simplicity, we load a network that is already available in
HDDM. You would call you own code instead.

.. code:: ipython3

    from hddm.simulators.hddm_dataset_generators import simulator_h_c
    
    # Simulate some data:
    model = 'ddm'
    n_subjects = 1
    n_samples_by_subject = 500
    
    data, full_parameter_dict = simulator_h_c(n_subjects = n_subjects,
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

Initialize HDDM Model
~~~~~~~~~~~~~~~~~~~~~

Now you are ready to load a HDDM model with your ``custom`` LAN.

.. code:: ipython3

    # Define the HDDM model
    hddmnn_model = hddm.HDDMnn(data = data,
                               informative = False,
                               include = my_model_config['hddm_include'], # Note: This include statement is an example, you may pick any other subset of the parameters of your model here
                               model = 'custom',
                               model_config = my_model_config,
                               network = custom_network)

You are now ready to get samples from your model.

.. code:: ipython3

    hddmnn_model.sample(1000, burn = 100)


.. parsed-literal::

     [-----------------100%-----------------] 1000 of 1000 complete in 36.9 sec



.. parsed-literal::

    <pymc.MCMC.MCMC at 0x149f01150>



**WARNING**

Not all the functionality of the HDDM package will work seamlessly with
such custom likelihoods. You will be able to generate some, but not all
plots.

The utility lies in using HDDM as a vehicle to sample from user defined
approximate likelihoods. Most of the packages utility functions have a
higher degree of specificity to models that have been fully incorporated
into the package.

A tutorial concerning full integration into HDDM will be made available
in the near future.
