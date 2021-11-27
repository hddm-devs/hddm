.. index:: LANs
.. _chap_custom_lans:

Use your Custom LAN
===================

If you have your own LAN, or really, a class with a
``predict_on_batch()`` method which you can call to get back the
log-likelihood of a dataset, you can pass it as an argument to the
``HDDMnn``, ``HDDMnnRegressor`` or ``HDDMnnStimCoding`` classes.

In this document we provide you with a simple complete example.

Such a new model will be called ``custom``, for our purposes. Let’s say
we trained a LAN for it.

We need two components to use our ``custom`` LAN.

1. A config dictionary for the ``custom`` model.
2. A pretrained LAN with a ``predict_on_batch`` method.

Construct the config dictionary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import hddm

Access the config dictionary for all models included in ``HDDM`` by
calling the ``hddm.model_config.model_config`` dictionary. Learn more
about this object in **lan tutorial concerning new models**.

.. code:: ipython3

    hddm.model_config.model_config["custom"] =  {
            "params": ["v", "a", "z", "t"], # parameter names associated to your model 
            "params_trans": [0, 0, 1, 0], # for each parameter do you want the sampler to transform it into an unconstrained space? (invlogit <--> logistic)
            "params_std_upper": [1.5, 1.0, None, 1.0], # for group models, what is the maximal standard deviation to consider for the prior on the parameter
            "param_bounds": [[-3.0, 0.3, 0.1, 1e-3], [3.0, 2.5, 0.9, 2.0]], # the parameter boundaries you used for training your LAN
            "boundary": hddm.simulators.bf.constant, # add a boundary function (if relevant to your model) (optional)
            "n_params": 4, # number of parameters of your model
            "default_params": [0.0, 1.0, 0.5, 1e-3], # defaults for each parameter 
            "hddm_include": ["z"], # suggestion for which parameters to include via the include statement of an HDDM model (usually you want all of the parameters from above)
            "n_choices": 2, # number of choice options of the model
            "choices": [-1, 1], # choice labels (what your simulator spits out)
            "slice_widths": {"v": 1.5, "v_std": 1,  
                             "a": 1, "a_std": 1, 
                             "z": 0.1, "z_trans": 0.2, 
                             "t": 0.01, "t_std": 0.15}, # hyperparameters for the slice-sampler used for posterior sampling, take these as an orientation, can be helpful to optimize speed (optional)
        }

Load the network
~~~~~~~~~~~~~~~~

To make the example complete, here is a code snippet to load in a
``Keras`` model. We are simply using the ``ddm`` network here.

.. code:: ipython3

    custom_network = hddm.torch.mlp_inference_class.load_torch_mlp(model = 'ddm')  # or any class with a valid predict on batch function

.. code:: ipython3

    # Simulate some data:
    model = 'ddm'
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

Initialize the Hddm Model
~~~~~~~~~~~~~~~~~~~~~~~~~

Now you are ready to load a HDDM model with your ``custom`` LAN.

.. code:: ipython3

    # Define the HDDM model
    hddmnn_model = hddm.HDDMnn(data = data,
                               informative = False,
                               include = hddm.model_config.model_config['custom']['hddm_include'], # Note: This include statement is an example, you may pick any other subset of the parameters of your model here
                               model = 'custom',
                               network_type = 'torch_mlp',
                               network = custom_network)


.. parsed-literal::

    Setting priors uninformative (LANs only work with uninformative priors for now)
    Includes supplied:  ['z']


You are now ready to get samples from your model.

.. code:: ipython3

    hddmnn_model.sample(1000, burn = 100)


.. parsed-literal::

     [-----------------100%-----------------] 1000 of 1000 complete in 23.1 sec



.. parsed-literal::

    <pymc.MCMC.MCMC at 0x7fc56d733a50>



Warning
~~~~~~~

Not all the functionality of the HDDM package will work seamlessly with
such custom likelihoods. You will be able to generate some, but not all
plots.

The utility lies in using HDDM as a vehicle to sample from user defined
approximate likelihoods. Most of the packages utility functions have a
higher degree of specificity to models that have been fully incorporated
into the package.

Look at the tutorial ``add_new_models_to_hddm_tutorial.ipynb`` for a
higher degree of integration.
