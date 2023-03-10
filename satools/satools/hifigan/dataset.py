import os
import random
import inspect
import json

import librosa
import numpy as np
import torch
import logging
import torchaudio
from librosa.filters import mel as librosa_mel_fn

from .. import utils

class WavInfo(object):
    """
    WavInfo objects hole information about each example
    Can either be used for a single wav:
        self.name = name
        self.filename = name
        self.wav = wav
    Or for multiple wavs batch during training
        self.name = List[name]
        self.filename = List[name]
        self.wav = Tensor[wav]
        self.ys = Tensor[ys]
        self.lengths = List[lengths]
    """

    def __init__(self, wav, name, filename, ys=None, lengths=None, other={}):
        self.name = name
        self.filename = filename
        self.wav = wav

        # for batch
        self.ys = ys
        self.lengths = lengths

        # pre-precessed feature (f0 asrbn..)
        self.other = other

    def __repr__(self):
        if self.ys == None:
            return f"(name={self.name}, wav={self.wav.shape}, filename={self.filename})"
        return f"(Batch names={self.name}, wavs={self.wav.shape}, filenames={self.filename}, lengths={self.length}, other={self.other})"

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
        return WavInfo(waveform, self.wavs_idx[idx], filename)


def collate_fn_padd(model, opts):

    cache_functions = opts.cache_functions
    if not isinstance(opts.cache_functions, dict):
        cache_functions = json.loads(opts.cache_functions)
        print("JSON::", opts.cache_functions, flush=True)


    def _func_pad(batch_in):
        names = [b.name for b in batch_in]
        filenames = [b.filename for b in batch_in]
        feat_batch = [b.wav for b in batch_in]
        ## get sequence lengths
        lengths = torch.tensor([t.shape[1] for t in feat_batch])
        ## padd
        feat_batch = [torch.Tensor(t).permute(1, 0) for t in feat_batch]
        feats = torch.nn.utils.rnn.pad_sequence(
            feat_batch, batch_first=True, padding_value=0
        )
        feats = torch.squeeze(feats.permute(0, 2, 1))

        if len(feats.shape) == 1:
            feats = torch.unsqueeze(feats, 0)

        acc_y = []
        for i, b in enumerate(feat_batch):
            # normalize feats for hifigan grount truth
            _feats_norm = b.squeeze().numpy()
            _feats_norm = _feats_norm * 2 ** 15
            _feats_norm = _feats_norm / 32768.0
            _feats_norm = librosa.util.normalize(_feats_norm) * 0.95
            _feats_norm = torch.FloatTensor(_feats_norm).unsqueeze(0)
            acc_y.append(_feats_norm.permute(1, 0))

        ypad = torch.nn.utils.rnn.pad_sequence(acc_y, batch_first=True, padding_value=0)
        ys = ypad.permute(0, 2, 1)

        wavinfo = WavInfo(feats, names, filenames, ys=ys, lengths=lengths)

        info = torch.utils.data.get_worker_info()
        print("WK", info)
        _id = info.id if info != None else "_"
        specifier_format = {"dir": opts.cache_path + "/", "worker": "_"+str(_id)+str(opts.rank)}

        # extract feat from model def
        other = {}
        for b in batch_in:
            try:
                o = extract_features(model, b, ask_compute="cpu",
                                     specifier_format=specifier_format,
                                     cache_funcs=cache_functions)
                other.update(o)
            except Exception:
                logging.error("Error while processing:", str(b))
        wavinfo.other = other

        return wavinfo

    return _func_pad


def register_feature_extractor(compute_device="cpu", scp_cache=None):
    cache = utils.fs.SCPCache(enabled=scp_cache!=None,
                              key=lambda hisself, egs:egs.name,
                              specifier=scp_cache,
                              )

    def wrapper(func):
        func = cache.decorate()(func)
        def model_feat_wrapper(hisself, egs: WavInfo, exec_in_decorator=False, ask_compute="cpu", specifier_format={}):
            cache.update_formatter(specifier_format)
            if not exec_in_decorator:
                return func(hisself, egs)

            if ask_compute == compute_device:
                result = func(hisself, egs)

            if not ask_compute == compute_device:
                return None

            return result
        model_feat_wrapper.cache = cache
        return model_feat_wrapper

    return wrapper


def extract_features(instance:torch.nn.Module, egs:WavInfo, ask_compute="cpu", specifier_format={}, cache_funcs=[]):
    # Get all functions from the class
    functions = inspect.getmembers(instance.__class__, predicate=inspect.isfunction)

    # Filter functions that have the decorator
    decorated_functions = [(f[0], f[1]) for f in functions if f[1].__name__.startswith("model_feat_wrapper")]
    # cache on for everyone by default if cache_funcs not provided
    if len(cache_funcs) == 0:
        cache_funcs = [f[0] for f in decorated_functions]
    ret = {}
    for fname, fdef in decorated_functions:
        print("cacheing?:", cache_funcs, fname, flush=True)
        if fdef.cache.enabled and fname not in cache_funcs: # disable if not in cache_funcs
            print("Disable", fname, flush=True)
            fdef.cache.enabled = False
        feat = getattr(instance, fname)(egs, exec_in_decorator=True, ask_compute=ask_compute, specifier_format=specifier_format)
        if feat != None:
            ret[fname] = feat
    return ret


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
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
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
        return_complex=True,
    )
    spec = torch.view_as_real(spec)

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

    import matplotlib.pylab as plt
    import matplotlib
    matplotlib.use("Agg")

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig
