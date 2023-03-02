# This file is part of SIDEKIT.
# https://git-lium.univ-lemans.fr/speaker/sidekit


import collections
import numpy
import random
import torch
import torchaudio

from scipy import signal


def fuse_speech_noise(speech, noise, snr_db):
    speech_power = speech.norm(p=2)
    if speech_power == 0:
        speech += 10e-3 * torch.randn(speech.shape)
        speech_power = speech.norm(p=2)
    noise_power = noise.norm(p=2)
    snr = 10 ** (snr_db / 20)
    if speech_power == 0:
        speech += 10e-3 * torch.randn(speech.shape)
    scale = snr * noise_power / speech_power
    return (scale * speech + noise) / 2


def data_augmentation(speech,
                      transform_dict,
                      sample_rate=16000,
                      noise_df=None,
                      rir_df=None):
    """
    Perform data augmentation on an input signal.
    Each speech chunk is augmented by using transform_dict["aug_number"] transformations that are picked up randomly from a
    dictionary of possible transformations.

    :param speech: the input signal to be augmented
    :param sample_rate: sampling rate of the input signal to augment
    :param transform_dict: the dictionary of possibles augmentations to apply
    :param rir_df: a pandas dataframe object including the list of RIR signals to chose from; default is None
    :param noise_df: a pandas dataframe object including the list of NOISE signals to chose from; default is None
    :param babble_noise: boolean that enable the use of babble noise, True by default (typically turned to False when
    the task includes overlapping speech detection).

    Config example:

    augmentation = {
      "pipeline": ["add_reverb", "add_noise", "phone_filtering", "codec"],
      "aug_number": 1,
      "add_noise": {
            "babble_noise": "true",
            "noise_db_csv": "/lium/raid01_b/pchampi/lab/sidekit/egs/voxceleb/list/musan.csv",
            "data_path": "/"
      },
      "add_reverb": {
            "rir_db_csv": "/lium/raid01_b/pchampi/lab/sidekit/egs/voxceleb/list/reverb.csv",
            "data_path": "/"
      },
      "sanity_check_path" : "/tmp/sanity_test",
      "sanity_check_samples" : 2
    }

    :return: augmented signal
    """
    # Select the data augmentation randomly
    aug_idx = random.sample(range(len(transform_dict["pipeline"])), k=transform_dict["aug_number"])
    augmentations = numpy.array(transform_dict["pipeline"])[aug_idx]

    if speech.dim() == 1:
        speech = speech.unsqueeze(0)

    if "none" in augmentations:
        pass

    if "add_reverb" in augmentations:
        rir_nfo = rir_df.iloc[random.randrange(rir_df.shape[0])].file_id
        rir_fn = transform_dict["add_reverb"]["data_path"] + rir_nfo  # TODO harmonize with noise
        rir, rir_fs = torchaudio.load(rir_fn)
        assert rir_fs == sample_rate
        # rir = rir[rir_nfo[1], :] #keep selected channel
        speech = torch.tensor(signal.convolve(speech, rir, mode='full')[:, :speech.shape[1]])

    if "add_noise" in augmentations:
        # Pick a noise type
        noise = torch.zeros_like(speech)
        if "babble_noise" in transform_dict["add_noise"] and transform_dict["add_noise"]["babble_noise"].lower() == "true":
            noise_idx = random.randrange(0, 4)
        else:
            noise_idx = random.randrange(1, 3)

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

        fuse_speech_noise(speech, noise, snr_db)

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

    if "codec" in augmentations:
        final_shape = speech.shape[1]
        configs = [
            ({"format": "wav", "encoding": 'ULAW', "bits_per_sample": 8}, "8 bit mu-law"),
            ({"format": "wav", "encoding": 'ALAW', "bits_per_sample": 8}, "8 bit a-law"),
            #({"format": "mp3", "compression": -9}, "MP3"),
            #({"format": "vorbis", "compression": -1}, "Vorbis")
        ]
        param, title = random.choice(configs)
        speech = torchaudio.functional.apply_codec(speech, sample_rate, **param)
        speech = speech[:, :final_shape]

    return speech, augmentations


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

