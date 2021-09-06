Train a Universal Background Model
==================================

Universal Background (Gaussian mixture) Models (UBM) are trained via EM algorithm using
the Mixture class from **SIDEKIT**.

UBM are trained using acoustic features that can be extracted on-line or loaded and post-processed from existing HDF5 feature files.
We acknowledge that UBM training might not be the most efficient as post processing of the acoustic features is performed on-line
(computation of the derivatives, concatenation of the different types of features, normalization) and that iterating over the data
might be time consuming. However, given the performance of parallel computing and the fact that a large quantity of
data is not necessary to train good quality models, we chose to use this approach which greatly reduces the feature storage on disk.

1. Training using EM split
--------------------------

Training of a GMM-UBM with diagonal covariance is straightforward:
   - create a Mixture
   - perform EM training

The two instructions are::

   ubm = sidekit.Mixture()

   ubm.EM_split(features_server=fs,
                feature_list=ubm_list,
                distrib_nb=1024,
                iterations=(1, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8), 
                num_thread=10,
                save_partial=False,
                ceil_cov=10, 
                floor_cov=1e-2
                )

In the above example, a GMM is trained with 1024 distributions.
Note that to perform the EM training you need to provide the following parameters:

   - a FeaturesServer that will be used to load data from disk or to process on the fly
   - a list of feature files to process (following the FeaturesServer requirements)
   - the final expected number of distribution as a power of 2
   - a tuple of iteration numbers where the i_th component is the number of iteration to run for the i_th size of model
   - a number of parallel process to run

You might also save the model after each iteration by setting `save_partial` to True.
You can constrain the covariance of the distribution by providing a ceiling and flooring values.

The training process is as follow:
   - initialize one Gaussian distribution given all the training data
   - Iterate n_i iterations of EM with the current size of model (fixed number of distributions)
   - split all distributions in two according to their variance in order to double the number of distributions of the GMM
   - save the resulting model

As an example, given the above `iterations` parameter, if training a model with 1024 distributions SIDEKIT
will perform:

   - 1 iteration of EM with 1 distribution
   - 2 iterations of EM with 2 distributions
   - 2 iterations of EM with 4 distributions
   - 4 iterations of EM with 8 distributions
   - 4 iterations of EM with 16 distributions
   - 4 iterations of EM with 32 distributions
   - 4 iterations of EM with 64 distributions
   - 8 iterations of EM with 128 distributions
   - 8 iterations of EM with 256 distributions
   - 8 iterations of EM with 512 distributions
   - 8 iterations of EM with 1024 distributions


2. Training using simple EM with fixed number of distributions
--------------------------------------------------------------

It is also possible to train a GMM with EM algorithm by directly setting the
number of distributions. In this case, the distributions will be initialized by taking the mean and covariance of
random subsets of the training data.
The code is as follow::
   
   ubm = sidekit.Mixture()

   ubm.EM_uniform(cep, 
                  distrib_nb, 
                  iteration_min=3, 
                  iteration_max=10,
                  llk_gain=0.01, 
                  do_init=True
                  )

Be careful that the arguments of this method are a bit different from the previous one.
Indeed, this method doesn't use any FeaturesServer but takes a ndarray containing all features
as rows.


3. Training using EM split on several nodes
-------------------------------------------

**SIDEKIT** allows parallel training of GMM using several nodes (machines) via the Message Passing Interface (MPI).

First, make sure MPI is installed on each node you intend to use.

Then enable the use of MPI by setting your environment variable to something like: ``SIDEKIT="mpi=true"``.

You're now ready to train your GMM by running:

.. code:: python

   ubm = sidekit.Mixture()
   sidekit.sidekit_mpi.EM_split(ubm=ubm,
                                features_server=fs,
                                feature_list=ubm_list,
                                distrib_nb=2048,
                                output_filename="ubm_tandem_mpi",
                                iterations=(1, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8),
                                llk_gain=0.01,
                                save_partial=True,
                                ceil_cov=10,
                                floor_cov=1e-2,
                                num_thread=30)

Where:
   - ``fs`` is a ``FeaturesServer`` object used to load the acoustic features
   - ``ubm_list`` is a list of shows (sessions) to process

Parameter ``num_thread`` is related to Multiprocessing that is used to load the features at first on Node 0.
Note that Multiprocessing is not used later in the process.

Refer to the :ref:`MPI`. page to see how to launch your computation on several nodes.

4 Full covariance UBM
---------------------

In order to train full covariance GMMs you can first train a GMM with diagonal covariance
using one of the two above methods then perform a number of EM iterations
to estimate the full covariance matrices.
This can be implementated as follows::
   
   ubm = sidekit.Mixture()

   ubm.EM_split(features_server, 
                feature_list, 
                distrib_nb,
                iterations=(1, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8), 
                num_thread=10,
                llk_gain=0.01, 
                save_partial=False,
                ceil_cov=10, 
                floor_cov=1e-2
                )

   ubm.EM_convert_full(features_server, 
                       featureList, 
                       distrib_nb,
                       iterations=2, 
                       num_thread=10
                       )

The method `EM_convert_full` can be applied on a previously trained diagonal Mixture.
We use here a FeaturesServer to access the acoustic frames and a list of shows (sessions).
The only two other parameters are the number of EM iterations to run and the number of thread in
case you want to paralllize the process.

