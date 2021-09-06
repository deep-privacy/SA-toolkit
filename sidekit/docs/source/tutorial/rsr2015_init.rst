Prepare to run experiments on RSR2015
=====================================

| Before running experiments on the RSR2015 database
| you need to run the script :mod:`rsr2015_init.py`
| in order to prepare the lists of file and indexes required.
|
| To work, this script only requires you to modify the path
| where the RSR2015 database is stored
|
| We assume here that the sphere files from the RSR2015 have been
| decompressed in the original directory and that the architecture
| of the directories follows the original architecture provided by the I2R.
|
| i.e.:
|
| rsr2015_root_directory
|          - key
|          - sph
|              - male
|              - female

Enter the path where the **RSR2015** database is stored
and run `python rsr2015_init.py`.
This script generates the following files:

      - ``task/3sess-pwd_eval_m_back.h5``
      - ``task/3sess-pwd_eval_m_key.h5``
      - ``task/3sess-pwd_eval_m_key.h5``
      - ``task/3sess-pwd_eval_m_nap.h5``
      - ``task/3sess-pwd_eval_m_ndx.h5``
      - ``task/3sesspwd_eval_m_trn.h5``
      - ``task/ubm_list.txt``

Below is a description of this script.


First, loads the required PYTHON packages.

.. code-block:: python

    import numpy as np
    import sidekit
    import os
    import sys
    import re
    import random
    import pandas as pd

Before running this script, don't forget to enter the path of the directory where the RSR2015 database is stored::

    rsr2015Path = '/Users/larcher/LIUM/RSR2015/RSR2015_V1/'

The rest of the script generates the files defining the enrollment data, the trials to process and the key for scoring
from the original files provided with the **RSR2015**.

The :mod:`IdMap` object is created from the *3sesspwd_eval_m.trn* file and save to disk:

.. code-block:: python

    rsrEnroll = pd.read_csv(rsr2015Path + '/key/part1/trn/3sesspwd_eval_m.trn', delimiter='[,\s*]', header=None, engine='python')

    # remove the extension of the files (.sph)
    for i in range(1,4):
        rsrEnroll[i] = rsrEnroll[i].str.replace('.sph$', '')
        rsrEnroll[i] = rsrEnroll[i].str.replace('^male/', '')

    # Create the list of models and enrollment sessions
    models = []
    segments = []
    for idx, mod in enumerate(rsrEnroll[0]):
        models.extend([mod, mod, mod])
        segments.extend([rsrEnroll[1][idx], rsrEnroll[2][idx], rsrEnroll[3][idx]])

    # Create and fill the IdMap with the enrollment definition
    enroll_idmap = sidekit.IdMap()
    enroll_idmap.leftids = np.asarray(models)
    enroll_idmap.rightids = np.asarray(segments)
    enroll_idmap.start = np.empty(enroll_idmap.rightids.shape, '|O')
    enroll_idmap.stop = np.empty(enroll_idmap.rightids.shape, '|O')
    enroll_idmap.validate()
    enroll_idmap.write('task/3sesspwd_eval_m_trn.h5')


The file *3sess-pwd_eval_m.ndx* is read and we extract information to process **target** trials
as well as **nontarget** trials that correspond to the case of an impostor pronouncing the correct sentence.
The :mod:`Key` object is stored in HDF5 format:

.. code-block:: python

    rsrKey = pd.read_csv(rsr2015Path + '/key/part1/ndx/3sess-pwd_eval_m.ndx', delimiter='[,\s*]', header=None, engine='python')
    rsrKey[1] = rsrKey[1].str.replace('.sph$', '')

    models = []
    testsegs = []
    trials = []

    for idx,model in enumerate(list(rsrKey[0])):
        if (rsrKey[2][idx] == 'Y'):
            models.append(rsrKey[0][idx])
            testsegs.append(rsrKey[1][idx])
            trials.append('target')
        elif (rsrKey[4][idx] == 'Y'):
            models.append(rsrKey[0][idx])
            testsegs.append(rsrKey[1][idx])
            trials.append('nontarget')

    key = sidekit.Key(models=np.array(models), testsegs=np.array(testsegs), trials=np.array(trials))

    key.write('task/3sess-pwd_eval_m_key.h5')

The index file that defines the trials to process is derived from the :mod:`Key` object and stored to disk
in HDF5 format:

.. code-block:: python

    ndx = key.to_ndx()
    ndx.write('task/3sess-pwd_eval_m_ndx.h5')

The following block creates a list of files that will be used to train
a Universal Background Model. This list is stored in ASCII format.
All the 30 sentences from the PART I of the **RSR2015** database 
from the 50 male speakers of the background set are used to train the
UBM:

.. code-block:: python

    ubmList = []
    p = re.compile('(.*)((m0[0-4][0-9])|(m050))(.*)((0[0-2][0-9])|(030))(\.sph$)')
    for dir_, _, files in os.walk(rsr2015Path):
        for fileName in files:
            if p.search(fileName):
                relDir = os.path.relpath(dir_, rsr2015Path + "/sph/male")
                relFile = os.path.join(relDir, fileName)
                ubmList.append(os.path.splitext(relFile)[0])
    with open('task/ubm_list.txt','w') as of:
        of.write("\n".join(ubmList))

The next section creates the list of files used to train the Nuisance Projection Attribute
matrix that can be used for SVM-GMM tutorial:

.. code-block:: python

    napSegments = ubmList[::7]
    napSpeakers = [seg.split('/')[0] for seg in napSegments]
    nap_idmap = sidekit.IdMap()
    nap_idmap.leftids = np.array(napSpeakers)
    nap_idmap.rightids = np.array(napSegments)
    nap_idmap.start = np.empty(nap_idmap.rightids.shape, '|O')
    nap_idmap.stop = np.empty(nap_idmap.rightids.shape, '|O')
    nap_idmap.validate()
    nap_idmap.write('task/3sess-pwd_eval_m_nap.h5')

Generate now the list of models that will be used 
as blacklist to train the Support Vector Machines:

.. code-block:: python

    backSegments = random.sample(ubmList, 200)
    backSpeakers = [seg.split('/')[0] for seg in backSegments]
    back_idmap = sidekit.IdMap()
    back_idmap.leftids = np.array(backSpeakers)
    back_idmap.rightids = np.array(backSegments)
    back_idmap.start = np.empty(back_idmap.rightids.shape, '|O')
    back_idmap.stop = np.empty(back_idmap.rightids.shape, '|O')
    back_idmap.validate()
    back_idmap.write('task/3sess-pwd_eval_m_back.h5')

Eventually creates the :mod:`IdMap` to compute statistics of the test segments
for the tutorial on SVMs:

.. code-block:: python

    test_idmap = sidekit.IdMap()
    test_idmap.leftids = ndx.segset
    test_idmap.rightids = ndx.segset
    test_idmap.start = np.empty(test_idmap.rightids.shape, '|O')
    test_idmap.stop = np.empty(test_idmap.rightids.shape, '|O')
    test_idmap.validate()
    test_idmap.write('task/3sess-pwd_eval_m_test.h5')

