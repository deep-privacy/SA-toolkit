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

:mod:`frontend` provides methods to process an audio signal in order to extract
useful parameters for speaker verification.
"""

from .io import write_pcm
from .io import read_pcm
from .io import pcmu2lin
from .io import read_sph
from .io import write_label
from .io import read_label
from .io import read_spro4
from .io import read_audio
from .io import write_spro4
from .io import read_htk
from .io import write_htk
from .io import read_hdf5_segment
from .io import write_hdf5
from .io import read_hdf5

from .vad import vad_snr
from .vad import label_fusion
from .vad import speech_enhancement


from .normfeat import cms
from .normfeat import cmvn
from .normfeat import stg
from .normfeat import rasta_filt
from .normfeat import cep_sliding_norm


from .features import compute_delta
from .features import framing
from .features import pre_emphasis
from .features import trfbank
from .features import mel_filter_bank
from .features import mfcc
from .features import pca_dct
from .features import shifted_delta_cepstral


__author__ = "Anthony Larcher and Sylvain Meignier"
__copyright__ = "Copyright 2014-2021 Anthony Larcher and Sylvain Meignier"
__license__ = "LGPL"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'
