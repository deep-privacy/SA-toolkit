Acoustic parametrization
========================

This part of the documentation details the different tools for acoustic parameters extraction, storage and usage.
In **SIDEKIT**, low level interface for acoustic parametrization is implemented in the ``frontend``
module.
Two high level classes allow a fast and simple extraction of acoustic parameters:

   - ``FeaturesExtractor``
   - ``FeaturesServer``

Before introducing those objects, we give a brief description of the HDF5 format that is used to store and exchange
acoustic features. The HDF5 format is the prefered serialization format in **SIDEKIT**.

.. toctree::
   :maxdepth: 2

   hdf5
   featuresextractor
   featuresserver
