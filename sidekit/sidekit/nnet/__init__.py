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

:mod:`nnet` provides methods to manage Neural Networks using PyTorch
"""


from .xsets import IdMapSetPerSpeaker
from .xsets import SideSet
from .xsets import SideSampler
from .xvector import Xtractor
from .xvector import xtrain
from .xvector import extract_embeddings
from .pooling import MeanStdPooling
from .pooling import AttentivePooling
from .pooling import GruPooling
from .res_net import ResBlock
from .res_net import PreResNet34
from .res_net import PreFastResNet34
from .res_net import PreHalfResNet34
from .sincnet import SincNet
from .preprocessor import RawPreprocessor
from .preprocessor import MfccFrontEnd 
from .preprocessor import MelSpecFrontEnd


__author__ = "Anthony Larcher and Sylvain Meignier"
__copyright__ = "Copyright 2014-2021 Anthony Larcher and Sylvain Meignier"
__license__ = "LGPL"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'
