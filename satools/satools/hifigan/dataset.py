import os
import random

import librosa
import numpy as np
import torch
import torchaudio
from librosa.filters import mel as librosa_mel_fn

from . import f0


class WavList(torch.utils.data.Dataset):
    def __init__(self, wavs_paths, wavs_idx, load_func=None):
        if isinstance(wavs_paths, str):
            self.wavs_path = wavs_paths.split(",")
            self.wavs_idx = wavs_idx.split(",")
        else:
            self.wavs_path = wavs_paths
            self.wavs_idx = wavs_idx

        assert len(self.wavs_path) == len(self.wavs_idx)
        if load_func == None:

            def _load(filename):
                return torchaudio.load(filename)

            self.load = _load
        else:
            self.load = load_func

    def __len__(self):
        return len(self.wavs_path)

    def __getitem__(self, idx):
        filename = self.wavs_path[idx]
        waveform, sr = self.load(filename)
        return (waveform, self.wavs_idx[idx], filename)


def collate_fn_padd():
    def _func_pad(batch):
        filenames = [b[1] for b in batch]
        batch = [b[0] for b in batch]
        """
        Padds batch of variable length and compute f0
        """
        ## get sequence lengths
        lengths = torch.tensor([t.shape[1] for t in batch])
        ## padd
        batch = [torch.Tensor(t).permute(1, 0) for t in batch]
        feats = torch.nn.utils.rnn.pad_sequence(
            batch, batch_first=True, padding_value=0
        )
        feats = torch.squeeze(feats.permute(0, 2, 1))

        if len(feats.shape) == 1:
            feats = torch.unsqueeze(feats, 0)

        acc_y = []
        for i, b in enumerate(batch):
            # normalize feats for hifigan grount truth
            _feats_norm = b.squeeze().numpy()
            _feats_norm = _feats_norm * 2 ** 15
            _feats_norm = _feats_norm / f0.MAX_WAV_VALUE
            _feats_norm = librosa.util.normalize(_feats_norm) * 0.95
            _feats_norm = torch.FloatTensor(_feats_norm).unsqueeze(0)
            acc_y.append(_feats_norm.permute(1, 0))

        ypad = torch.nn.utils.rnn.pad_sequence(acc_y, batch_first=True, padding_value=0)
        ys = ypad.permute(0, 2, 1)

        print("feat.shape:", feat.shape)
        return feats, lengths, filenames, ys

    return _func_pad


def sample_interval(seqs, seq_len, max_len=None):
    """
    Randomly samples an interval of length `seq_len` from a set of sequences `seqs`,
    ensuring that the interval is of the same length for all sequences.
    Takes into account list of sequences with different sampling rates.

    The function first determines the maximum sequence length in the list seqs
    and calculates the "hop size" (the number of time steps per sample) for
    each sequence based on its length. It then finds the least common multiple
    (LCM) of these hop sizes, which ensures that the sampled intervals will be
    of the same length for all sequences, regardless of their original sampling
    rates.

    Args:
        seqs (list of torch.Tensor): A list of sequences, each represented as a torch.Tensor.
        seq_len (int): The length of the interval to sample from each sequence.
        max_len (int): If not None, the maximum length of the sequence to sample from.
                       Defaults to None.

    Returns:
        Tuple (new_seqs, new_iterval): A tuple containing two lists:
            - new_seqs: A list of torch.Tensors, each containing a sampled interval of length `seq_len`.
            - new_iterval: A list of tuples, each representing the start and end indices of the sampled interval.

    """
    N = max([v.shape[-1] for v in seqs])

    hops = [N // v.shape[-1] for v in seqs]
    lcm = np.lcm.reduce(hops)

    # Randomly pickup with the batch_max_steps length of the part
    interval_start = 0
    interval_end = N // lcm - seq_len // lcm
    if max_len != None:
        interval_end = (max_len // lcm) - seq_len // lcm
    if max_len < seq_len:
        start_step = 0
        for i, v in enumerate(seqs):
            seqs[i] = torch.nn.functional.pad(
                v, (0, seq_len - v.shape[-1]), mode="constant", value=0
            )
    else:
        start_step = random.randint(interval_start, interval_end)

    new_seqs = []
    new_iterval = []
    for i, v in enumerate(seqs):
        start = start_step * (lcm // hops[i])
        end = (start_step + seq_len // lcm) * (lcm // hops[i])
        new_seqs += [v[..., start:end]]
        new_iterval += [(start, end)]

    return new_seqs, new_iterval


mel_basis = {}
hann_window = {}


def mel_spectrogram(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    if torch.min(y) < -1.0:
        logging.error("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        logging.error("max value is ", torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output
