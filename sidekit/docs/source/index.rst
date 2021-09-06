.. SIDEKIT documentation master file, created by
   sphinx-quickstart on Mon Oct 27 10:12:02 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |logo| image:: logo_lium.png


Welcome to SIDEKIT 1.3.1 documentation!
=======================================

| **SIDEKIT** is an open source package for Speaker and Language recognition.
| The aim of **SIDEKIT** is to provide an educational and efficient toolkit for speaker/language recognition
| including the whole chain of treatment that goes from the audio data to the analysis of the system performance.


:Authors: 
    Anthony Larcher \&
    Kong Aik Lee \&
    Sylvain Meignier

:Version: 1.3.1 of 2019/01/22

.. seealso::

   News for **SIDEKIT** 1.3.1:

      - new ``sidekit_mpi`` module that allows parallel computing on several nodes (cluster)
        MPI implementations are provided for GMM EM algorithm, TotalVariability matrix EM estimation
        and i-vector extraction
        see `MPI <https://pythonhosted.org/mpi4py/>`_ for more information about MPI
      - new ``FactorAnalyser`` class that simplifies the interface
         Note that FA estimation and i-vector extraction is still available in ``StatServer`` but deprecated
      - i-vector scoring with scaling factor
      - uncertainty propagation is available in PLDA scoring


What's here?
============

.. toctree::
   :maxdepth: 1
   :name: mastertoc

   overview/index.rst
   install/index.rst
   api/envvar.rst
   api/index.rst
   tutorial/index.rst
   addon/index.rst


Citation
--------

When using **SIDEKIT** for research, please cite:

| Anthony Larcher, Kong Aik Lee and Sylvain Meignier, 
| **An extensible speaker identification SIDEKIT in Python**,
| in International Conference on Audio Speech and Signal Processing (ICASSP), 2016

Documentation
-------------

This documentation is available in PDF format :download:`here <../build/latex/sidekit.pdf>`


Contacts and info
-----------------

.. toctree::
   :maxdepth: 3
   :titlesonly:

   contact.rst


Sponsors
========

|logo|

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

