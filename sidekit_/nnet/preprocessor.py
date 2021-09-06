# coding: utf-8 -*-
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
Copyright 2014-2021 Anthony Larcher, Yevhenii Prokopalo
"""


import logging
import numpy
import os
import torch
import torchaudio

from .augmentation import PreEmphasis
from .sincnet import SincConv1d
from .res_net import LayerNorm


os.environ['MKL_THREADING_LAYER'] = 'GNU'

__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2015-2021 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reS'


logging.basicConfig(format='%(asctime)s %(message)s')


# Make PyTorch Deterministic
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
numpy.random.seed(0)


class MfccFrontEnd(torch.nn.Module):
    """
    Module that extract MFCC coefficients
    """
    def __init__(self,
                 pre_emphasis=0.97,
                 sample_rate=16000,
                 n_fft=2048,
                 f_min=133.333,
                 f_max=6855.4976,
                 win_length=1024,
                 window_fn=torch.hann_window,
                 hop_length=512,
                 power=2.0,
                 n_mels=100,
                 n_mfcc=80):

        super(MfccFrontEnd, self).__init__()

        self.pre_emphasis = pre_emphasis
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max
        self.win_length = win_length
        self.window_fn=window_fn
        self.hop_length = hop_length
        self.power=power
        self.window_fn = window_fn
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

        self.PreEmphasis = PreEmphasis(self.pre_emphasis)

        self.melkwargs = {"n_fft":self.n_fft,
                          "f_min":self.f_min,
                          "f_max":self.f_max,
                          "win_length":self.win_length,
                          "window_fn":self.window_fn,
                          "hop_length":self.hop_length,
                          "power":self.power,
                          "n_mels":self.n_mels}

        self.MFCC = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            dct_type=2,
            log_mels=True,
            melkwargs=self.melkwargs)

        self.CMVN = torch.nn.InstanceNorm1d(self.n_mfcc)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                mfcc = self.PreEmphasis(x)
                mfcc = self.MFCC(mfcc)
                mfcc = self.CMVN(mfcc)
        return mfcc


class MelSpecFrontEnd(torch.nn.Module):
    """
    Module that compute Mel spetrogramm on an audio signal
    """
    def __init__(self,
                 pre_emphasis=0.97,
                 sample_rate=16000,
                 n_fft=1024,
                 f_min=90,
                 f_max=7600,
                 win_length=400,
                 window_fn=torch.hann_window,
                 hop_length=160,
                 power=2.0,
                 n_mels=80):

        super(MelSpecFrontEnd, self).__init__()

        self.pre_emphasis = pre_emphasis
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max
        self.win_length = win_length
        self.window_fn=window_fn
        self.hop_length = hop_length
        self.power=power
        self.window_fn = window_fn
        self.n_mels = n_mels

        self.PreEmphasis = PreEmphasis(self.pre_emphasis)

        self.melkwargs = {"n_fft":self.n_fft,
                          "f_min":self.f_min,
                          "f_max":self.f_max,
                          "win_length":self.win_length,
                          "window_fn":self.window_fn,
                          "hop_length":self.hop_length,
                          "power":self.power,
                          "n_mels":self.n_mels}

        self.MelSpec = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate,
                                                            n_fft=self.melkwargs['n_fft'],
                                                            f_min=self.melkwargs['f_min'],
                                                            f_max=self.melkwargs['f_max'],
                                                            win_length=self.melkwargs['win_length'],
                                                            hop_length=self.melkwargs['hop_length'],
                                                            window_fn=self.melkwargs['window_fn'],
                                                            power=self.melkwargs['power'],
                                                            n_mels=self.melkwargs['n_mels'])

        self.CMVN = torch.nn.InstanceNorm1d(self.n_mels)
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=5)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)

    def forward(self, x, is_eval=False):
        """

        :param x:
        :param is_eval:
        :return:
        """
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                out = self.PreEmphasis(x)
                out = self.MelSpec(out)+1e-6
                out = torch.log(out)
                out = self.CMVN(out)
                if not is_eval:
                    out = self.freq_masking(out)
                    out = self.time_masking(out)
        return out


class RawPreprocessor(torch.nn.Module):
    """
    Pre-process the raw audio signal by using a SincNet architecture
    [ADD REF]
    """
    def __init__(self, nb_samp, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50, sample_rate=16000):
        """

        :param nb_samp:
        :param in_channels:
        :param filts:
        :param first_conv:
        """
        super(RawPreprocessor, self).__init__()
        self.ln = LayerNorm(nb_samp)
        self.first_conv = SincConv1d(in_channels = in_channels,
                                     out_channels = out_channels,
                                     kernel_size = kernel_size,
                                     sample_rate = sample_rate,
                                     stride=stride,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=bias,
                                     groups=groups,
                                     min_low_hz=min_low_hz,
                                     min_band_hz=min_band_hz
                                     )
        self.first_bn = torch.nn.BatchNorm1d(num_features = out_channels)
        self.lrelu = torch.nn.LeakyReLU()
        self.lrelu_keras = torch.nn.LeakyReLU(negative_slope = 0.3)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        out = self.ln(x)
        out = out.view(nb_samp, 1, len_seq)
        out = torch.nn.functional.max_pool1d(torch.abs(self.first_conv(out)), 3)
        out = self.first_bn(out)
        out = self.lrelu_keras(out)

        return out

