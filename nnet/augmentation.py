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
Copyright 2014-2021 Anthony Larcher
"""

import collections
import numpy
import random
import torch
import torchaudio

from scipy import signal


__author__ = "Anthony Larcher and Sylvain Meignier"
__copyright__ = "Copyright 2014-2021 Anthony Larcher and Sylvain Meignier"
__license__ = "LGPL"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


Noise = collections.namedtuple('Noise', 'type file_id duration')


class PreEmphasis(torch.nn.Module):
    """
    Apply pre-emphasis filtering
    """

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input_signal: torch.tensor) -> torch.tensor:
        """
        Forward pass of the pre-emphasis filtering

        :param input_signal: the input signal
        :return: the filtered signal
        """
        assert len(input_signal.size()) == 2, 'The number of dimensions of input tensor must be 2!'
        # reflect padding to match lengths of in/out
        input_signal = input_signal.unsqueeze(1)
        input_signal = torch.nn.functional.pad(input_signal, (1, 0), 'reflect')
        return torch.nn.functional.conv1d(input_signal, self.flipped_filter).squeeze(1)


class FrequencyMask(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, max_size, feature_size):
        self.max_size = max_size
        self.feature_size = feature_size

    def __call__(self, sample):
        data = sample[0]
        if sample[2]:
            size = numpy.random.randint(1, self.max_size)
            f0 = numpy.random.randint(0, self.feature_size - self.max_size)
            data[f0:f0+size, :] = 10.
        return data, sample[1], sample[2], sample[3], sample[4], sample[5]


class TemporalMask(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, max_size):
        self.max_size = max_size

    def __call__(self, sample):
        data = sample[0]
        if sample[3]:
            size = numpy.random.randint(1, self.max_size)
            t0 = numpy.random.randint(0, sample[0].shape[1] - self.max_size)
            data[:, t0:t0+size] = 10.
        return data, sample[1], sample[2], sample[3], sample[4], sample[5]


def normalize(wav):
    """
    Center and reduce a waveform

    :param wav: the input waveform
    :return: the normalized waveform
    """
    return wav / (numpy.sqrt(numpy.mean(wav ** 2)) + 1e-8)


def crop(input_signal, duration):
    """
    Select a chunk from an audio segment

    :param input_signal: signal to select a chunk from
    :param duration: duration of the chunk to select
    :return:
    """
    start = random.randint(0, input_signal.shape[0] - duration)
    chunk = input_signal[start: start + duration]
    return chunk


def data_augmentation(speech,
                      sample_rate,
                      transform_dict,
                      transform_number,
                      noise_df=None,
                      rir_df=None,
                      babble_noise=True):
    """
    Perform data augmentation on an input signal.
    Each speech chunk is augmented by using 'transform_number' transformations that are picked up randomly from a
    dictionary of possible transformations.

    :param speech: the input signal to be augmented
    :param sample_rate: sampling rate of the input signal to augment
    :param transform_dict: the dictionary of possibles augmentations to apply
    :param transform_number: the number of transformations to apply on each chunk
    :param rir_df: a pandas dataframe object including the list of RIR signals to chose from; default is None
    :param noise_df: a pandas dataframe object including the list of NOISE signals to chose from; default is None
    :param babble_noise: boolean that enable the use of babble noise, True by default (typically turned to False when
    the task includes overlapping speech detection).

    :return: augmented signal

    tranformation
        pipeline: add_noise,add_reverb
        add_noise:
            noise_db_csv: filename.csv
            snr: 5,6,7,8,9,10,11,12,13,14,15
        add_reverb:
            rir_db_csv: filename.csv
        codec: true
        phone_filtering: true
    """
    # Select the data augmentation randomly
    aug_idx = random.sample(range(len(transform_dict.keys())), k=transform_number)
    augmentations = numpy.array(list(transform_dict.keys()))[aug_idx]

    if "stretch" in augmentations:
        strech = torchaudio.functional.TimeStretch()
        rate = random.uniform(0.8,1.2)
        speech = strech(speech, rate)

    if "add_reverb" in augmentations:
        rir_nfo = rir_df.iloc[random.randrange(rir_df.shape[0])].file_id
        rir_fn = transform_dict["add_reverb"]["data_path"] + rir_nfo  # TODO harmonize with noise
        rir, rir_fs = torchaudio.load(rir_fn)
        assert rir_fs == sample_rate
        #rir = rir[rir_nfo[1], :] #keep selected channel
        speech = torch.tensor(signal.convolve(speech, rir, mode='full')[:, :speech.shape[1]])

    if "add_noise" in augmentations:
        # Pick a noise type
        noise = torch.zeros_like(speech)
        if not babble_noise:
            noise_idx = random.randrange(1, 3)
        else:
            noise_idx = random.randrange(0, 4)

        # speech
        if noise_idx == 0:
            # Pick a SNR level
            # TODO make SNRs configurable by noise type
            snr_db = random.randint(13, 20)
            pick_count = random.randint(3, 7)
            index_list = random.sample(range(noise_df.loc['speech'].shape[0]), k=pick_count)
            for idx in index_list:
                noise_row = noise_df.loc['speech'].iloc[idx]
                noise += load_noise_seg(noise_row, speech.shape, sample_rate, transform_dict["add_noise"]["data_path"])
            noise /= pick_count
        # music
        elif noise_idx == 1:
            snr_db = random.randint(5, 15)
            noise_row = noise_df.loc['music'].iloc[random.randrange(noise_df.loc['music'].shape[0])]
            noise += load_noise_seg(noise_row, speech.shape, sample_rate, transform_dict["add_noise"]["data_path"])
        # noise
        elif noise_idx == 2:
            snr_db = random.randint(0, 15)
            noise_row = noise_df.loc['noise'].iloc[random.randrange(noise_df.loc['noise'].shape[0])]
            noise += load_noise_seg(noise_row, speech.shape, sample_rate, transform_dict["add_noise"]["data_path"])
        # babble noise with different volume
        elif noise_idx == 3:
            snr_db = random.randint(13,20)
            pick_count = random.randint(5,10) # Randomly select 5 to 10 speakers
            index_list = random.choices(range(noise_df.loc['speech'].shape[0]), k=pick_count)

            noise = torch.zeros(1,speech.shape[1])
            for idx in index_list:
                noise_row = noise_df.loc['speech'].iloc[idx]
                noise_ = load_noise_seg(noise_row, speech.shape, sample_rate, transform_dict["add_noise"]["data_path"])
                transform = torchaudio.transforms.Vol(gain=random.randint(5,15),gain_type='db') # Randomly select volume level (5-15d)
                noise += transform(noise_)
            noise /= pick_count

        speech_power = speech.norm(p=2)
        noise_power = noise.norm(p=2)
        snr = 10 ** (snr_db / 20)
        scale = snr * noise_power / speech_power
        speech = (scale * speech + noise) / 2

    if "phone_filtering" in augmentations:
        final_shape = speech.shape[1]
        speech, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
            speech,
            sample_rate,
            effects=[
                ["lowpass", "4000"],
                ["compand", "0.02,0.05", "-60,-60,-30,-10,-20,-8,-5,-8,-2,-8", "-8", "-7", "0.05"],
                ["rate", "16000"],
            ])
        speech = speech[:, :final_shape]

    if "filtering" in augmentations:
        effects = [
            ["bandpass","2000","3500"],
            ["bandstop","200","500"]]
        speech, sample_rate = torchaudio.sox_eefects.apply_effects_tensor(
            speech,
            sample_rate,
            effects=[effects[random.randint(0, 1)]],
        )

    if "codec" in augmentations:
        final_shape = speech.shape[1]
        configs = [
            ({"format": "wav", "encoding": 'ULAW', "bits_per_sample": 8}, "8 bit mu-law"),
            ({"format": "gsm"}, "GSM-FR"),
            ({"format": "mp3", "compression": -9}, "MP3"),
            ({"format": "vorbis", "compression": -1}, "Vorbis")
        ]
        param, title = random.choice(configs)
        speech = torchaudio.functional.apply_codec(speech, sample_rate, **param)
        speech = speech[:, :final_shape]

    return speech


def load_noise_seg(noise_row, speech_shape, sample_rate, data_path):
    """
    Pick a noise signal to add while performing data augmentation

    :param noise_row: a row from a Pandas dataframe object
    :param speech_shape: shape of the speech signal to be augmented
    :param sample_rate: sampling rate of the speech signal to be augmented
    :param data_path: directory where to load the noise file from
    :return:
    """
    noise_start = noise_row['start']
    noise_duration = noise_row['duration']
    noise_file_id = noise_row['file_id']

    if noise_duration * sample_rate > speech_shape[1]:
        # It is recommended to split noise files (especially speech noise type) in shorter subfiles
        # When frame_offset is too high, loading the segment can take much longer
        frame_offset = random.randrange(noise_start * sample_rate,
                                        int((noise_start + noise_duration) * sample_rate - speech_shape[1]))
    else:
        frame_offset = noise_start * sample_rate

    noise_fn = data_path + "/" + noise_file_id + ".wav"
    if noise_duration * sample_rate > speech_shape[1]:
        noise_seg, noise_sr = torchaudio.load(noise_fn, frame_offset=int(frame_offset), num_frames=int(speech_shape[1]))
    else:
        noise_seg, noise_sr = torchaudio.load(noise_fn,
                                              frame_offset=int(frame_offset),
                                              num_frames=int(noise_duration * sample_rate))
    assert noise_sr == sample_rate

    if noise_seg.shape[1] < speech_shape[1]:
        noise_seg = torch.tensor(numpy.resize(noise_seg.numpy(), speech_shape))
    return noise_seg
