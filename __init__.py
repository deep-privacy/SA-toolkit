# -*- coding: utf-8 -*-
#
# This file is part of SIDEKIT.
#
# SIDEKIT is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#
# SIDEKIT is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#    
# SIDEKIT is free software: you can redistribute it and/or modify
# it under the terms of the GNU LLesser General Public License as 
# published by the Free Software Foundation, either version 3 of the License, 
# or (at your option) any later version.
#
# SIDEKIT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with SIDEKIT.  If not, see <http://www.gnu.org/licenses/>.

"""
Copyright 2014-2021 Anthony Larcher and Sylvain Meignier
"""

from ctypes import *
from ctypes.util import find_library
import importlib
import logging
import numpy
import os
import sys


# Read environment variable if it exists
SIDEKIT_CONFIG={"libsvm":False,
                "mpi":False,
                "cuda":True
                }

if 'SIDEKIT' in os.environ:
    for cfg in os.environ['SIDEKIT'].split(","):
        k, val = cfg.split("=")
        if k == "libsvm":
           if val == "false":
                SIDEKIT_CONFIG["libsvm"] = False 
        elif k == "mpi":
            if val == "true":
               SIDEKIT_CONFIG["mpi"] = True
        if k == "cuda":
            if val == "true":
                SIDEKIT_CONFIG["cuda"] = True


PARALLEL_MODULE = 'multiprocessing'  # can be , threading, multiprocessing MPI is planned in the future
PARAM_TYPE = numpy.float32
STAT_TYPE = numpy.float64  # can be numpy.float32 to speed up the computation but can lead to numerical issuess

# Import bosaris-like classes
from .bosaris import IdMap
from .bosaris import Ndx
from .bosaris import Key
from .bosaris import Scores
from .bosaris import DetPlot
from .bosaris import effective_prior
from .bosaris import logit_effective_prior
from .bosaris import fast_minDCF

# Import classes
from .features_extractor import FeaturesExtractor
from .features_server import FeaturesServer
from .mixture import Mixture, vad_energy
from .statserver import StatServer
from .factor_analyser import FactorAnalyser

from .frontend.io import write_pcm
from .frontend.io import read_pcm
from .frontend.io import pcmu2lin
from .frontend.io import read_sph
from .frontend.io import write_label
from .frontend.io import read_label
from .frontend.io import read_spro4
from .frontend.io import read_audio
from .frontend.io import write_spro4
from .frontend.io import read_htk
from .frontend.io import write_htk

from .frontend.vad import vad_energy
from .frontend.vad import vad_snr
from .frontend.vad import label_fusion
from .frontend.vad import speech_enhancement

from .frontend.normfeat import cms
from .frontend.normfeat import cmvn
from .frontend.normfeat import stg
from .frontend.normfeat import rasta_filt

from .frontend.features import compute_delta
from .frontend.features import framing
from .frontend.features import pre_emphasis
from .frontend.features import trfbank
from .frontend.features import mel_filter_bank
from .frontend.features import mfcc
from .frontend.features import pca_dct
from .frontend.features import shifted_delta_cepstral

from .iv_scoring import cosine_scoring
from .iv_scoring import mahalanobis_scoring
from .iv_scoring import two_covariance_scoring
from .iv_scoring import PLDA_scoring
from .iv_scoring import fast_PLDA_scoring

from .gmm_scoring import gmm_scoring

from .jfa_scoring import jfa_scoring

from .score_normalization import znorm
from .score_normalization import tnorm
from .score_normalization import ztnorm
from .score_normalization import snorm

from .sidekit_io import write_norm_hdf5
from .sidekit_io import write_matrix_hdf5


from .sv_utils import clean_stat_server

libsvm_loaded = False
if SIDEKIT_CONFIG["libsvm"]:
    try:
        dirname = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libsvm')
        if sys.platform == 'win32':
            libsvm = CDLL(os.path.join(dirname, r'libsvm.dll'))
            libsvm_loaded = True
        else:
            libsvm = CDLL(os.path.join(dirname, 'libsvm.so.2'))
            libsvm_loaded = True
    except:
        # For unix the prefix 'lib' is not considered.
        if find_library('svm'):
            libsvm = CDLL(find_library('svm'))
            libsvm_loaded = True
        elif find_library('libsvm'):
            libsvm = CDLL(find_library('libsvm'))
            libsvm_loaded = True
        else:
            libsvm_loaded = False
            logging.warning('WARNNG: libsvm is not installed, please refer to the' +
                            ' documentation if you intend to use SVM classifiers')

if libsvm_loaded:
    from sidekit.libsvm import *
    from sidekit.svm_scoring import *
    from sidekit.svm_training import *


CUDA = False
if SIDEKIT_CONFIG["cuda"]:
    CUDA = True


if CUDA:
    from .nnet import Xtractor
    from .nnet import xtrain
    from .nnet import extract_embeddings
    from .nnet import ResBlock
    from .nnet import SincNet

else:
    print("Don't import Torch")

if SIDEKIT_CONFIG["mpi"]:
    found_mpi4py = importlib.find_loader('mpi4py') is not None
    if found_mpi4py:
        from .sidekit_mpi import EM_split
        from .sidekit_mpi import total_variability
        from .sidekit_mpi import extract_ivector
        print("Import MPI")
        

__author__ = "Anthony Larcher and Sylvain Meignier"
__copyright__ = "Copyright 2014-2021 Anthony Larcher and Sylvain Meignier"
__license__ = "LGPL"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'
__version__="1.4"

