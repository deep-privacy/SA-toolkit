import torch
import torchaudio

from librosa.filters import mel as librosa_mel_fn
import librosa

import numpy as np
import random

from . import f0


import matplotlib

matplotlib.use("Agg")
import matplotlib.pylab as plt


class WavList(torch.utils.data.Dataset):
    def __init__(self, wavs_paths, load_func=None):
        if isinstance(wavs_paths, str):
            self.wavs_path = wavs_paths.split(",")
        else:
            self.wavs_path = wavs_paths
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
        return (waveform, filename)


def collate_fn_padd(f0_stats, get_func=None, get_f0=True):
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

        f0s = []
        acc_y = []
        for i, b in enumerate(batch):
            if get_f0:
                f0s.append(
                    f0.get_f0(
                        b.permute(0, 1),
                        f0_stats=f0_stats,
                        cache_with_filename=filenames[i],
                    )
                    .squeeze(dim=1)
                    .permute(1, 0),
                )

            # normalize feats for hifigan grount truth
            _feats_norm = b.squeeze().numpy()
            _feats_norm = _feats_norm * 2 ** 15
            _feats_norm = _feats_norm / f0.MAX_WAV_VALUE
            _feats_norm = librosa.util.normalize(_feats_norm) * 0.95
            _feats_norm = torch.FloatTensor(_feats_norm).unsqueeze(0)
            acc_y.append(_feats_norm.permute(1, 0))

        if get_f0:
            f0spad = torch.nn.utils.rnn.pad_sequence(
                f0s, batch_first=True, padding_value=0
            )
            f0s = f0spad.permute(0, 2, 1)

        ypad = torch.nn.utils.rnn.pad_sequence(acc_y, batch_first=True, padding_value=0)
        ys = ypad.permute(0, 2, 1)

        if get_func != None:
            return get_func(feats, lengths, filenames, f0s, ys)
        return feats, lengths, filenames, f0s, ys

    return _func_pad


def sample_interval(seqs, seq_len, max_len=None):
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
            seqs[i] = torch.nn.functional.pad(v, (0, seq_len-v.shape[-1]), mode='constant', value=0)
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


def plot_spectrogram(audio):
    spectrogram = mel_spectrogram(
        y=audio,
        n_fft=1024,
        num_mels=80,
        sampling_rate=16000,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=8000,
    )
    spectrogram = spectrogram.squeeze(0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig
