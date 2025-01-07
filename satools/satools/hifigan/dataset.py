import os
import sys
import random

import librosa
import numpy as np
import torch
import logging
from librosa.filters import mel as librosa_mel_fn
from typing import List, Union, Dict, Optional, Callable, TypeVar, Any

from .. import utils


class Egs(object):
    def __init__(self, wavs:torch.Tensor, names:List[str], filenames:List[str], yss:torch.Tensor=torch.tensor([0]), lengths:torch.Tensor=torch.tensor([0])):
        self.names = names
        self.filenames = filenames
        self.wavs = wavs

        # for batch
        self.yss = yss
        self.lengths = lengths

        # pre-precessed feature (f0 asrbn..)
        self.extractor:Dict[str, torch.Tensor] = {}
        self.sample_done = False

    @torch.jit.unused
    def compute_cuda_extract_feat(self, generator, opts, name, device):
        if not self.sample_done:
            self.extractor.update(
                extract_features(
                    generator, self, opts.cache_functions,
                    {"dir": opts.cache_path + "/", "name": "_"+name, "worker": "_split"+str(opts.rank)}, ask_compute=str(device)
                )
            )

    @torch.jit.unused
    @torch.no_grad()
    def sample(self, segment_size):
        if self.sample_done:
            return self
        self.sample_done = True
        extractor_feat_to_sample = {k:v for k,v in self.extractor.items() if not k.endswith("_no_sample")}
        wavs, yss = [], []
        extractor_feat = {}
        for i, k in enumerate(extractor_feat_to_sample.keys()):
            extractor_feat[k] = []

        for batch_idx in range(len(self.names)):
            new_seqs, iterval_idx = sample_interval(
                [self.wavs[batch_idx], self.yss[batch_idx],
                 *list(map(lambda x:x[batch_idx], extractor_feat_to_sample.values()))],
                seq_len=segment_size,
                max_len=self.lengths[batch_idx].item(),
            )

            wavs.append(new_seqs[0])
            yss.append(new_seqs[1])
            for i, k in enumerate(extractor_feat_to_sample.keys()):
                extractor_feat[k].append(new_seqs[i+2])

        self.wavs = torch.stack(wavs)
        self.yss = torch.stack(yss)
        for i, k in enumerate(extractor_feat_to_sample.keys()):
            # Find the maximum length along the dimension to be padded
            max_length = max(tensor.size(0) for tensor in extractor_feat[k])

            # Get the shape of the remaining dimensions (assuming tensors are 2D or higher)
            remaining_shape = extractor_feat[k][0].size()[1:]

            # Initialize a zero tensor for padding
            padded_tensors = torch.zeros(
                (len(extractor_feat[k]), max_length, *remaining_shape), dtype=extractor_feat[k][0].dtype, device=extractor_feat[k][0].device
            )

            # Fill the padded tensor
            for idx, tensor in enumerate(extractor_feat[k]):
                padded_tensors[idx, :tensor.size(0)] = tensor

            # Store the padded tensor in the extractor
            self.extractor[k] = padded_tensors


        return self

    def __getitem__(self, key:str) -> torch.Tensor:
        if key in self.extractor.keys():
            return self.extractor[key]
        key2 = key + "_no_sample"
        if key2 in self.extractor.keys():
            return self.extractor[key2]
        return self.getattr(key)


    @torch.jit.unused
    def getattr(self, key) -> torch.Tensor:
        return getattr(self, key)

    def to(self, device:str="cpu"):
        self.wavs = self.wavs.to(device)
        self.yss = self.yss.to(device)
        self.lengths = self.lengths.to(device)
        for k, v in self.extractor.items():
            self.extractor[k] = v.to(device)

    def __repr__(self):
        return f"(Batch names={self.names}, wavs={self.wavs.shape}, filenames={self.filenames}, lengths={self.lengths}, extractor={self.extractor})"

    @torch.jit.unused
    def __iter__(self):
        for i in range(len(self.names)):
            yield utils.WavInfo(name=self.names[i], filename=self.filenames[i], wav=self.wavs[i][:self.lengths[i]].unsqueeze(0))


@torch.no_grad()
def model_collate(model, opts, auto_sample=True):
    cache_functions, cache_worker_name, cache_path, rank = setup_extractor(model, opts)
    segment_size = opts.segment_size

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
        yss = ypad.permute(0, 2, 1)

        egs = Egs(feats, names, filenames, yss=yss, lengths=lengths)

        info = torch.utils.data.get_worker_info()
        _id = info.id if info != None else -1
        specifier_format = {"dir": cache_path + "/" , "name": "_"+cache_worker_name, "worker": "_split"+str(rank)+str(_id)}

        egs.extractor = extract_features(model, egs, cache_functions, specifier_format, ask_compute="cpu")
        if len(egs.extractor.keys()) >= len(utils.extract_features_fnames(model)) and auto_sample:
            egs.sample(segment_size)
        return egs

    return _func_pad


def setup_extractor(model, opts):
    os.makedirs(opts.cache_path, exist_ok=True)
    cache_functions = opts.cache_functions
    cache_worker_name = opts.cache_worker_name
    cache_path = opts.cache_path
    rank = opts.rank
    if isinstance(cache_functions, str):
        opts.cache_functions = utils.fix_json(cache_functions)
        cache_functions = opts.cache_functions

    return cache_functions, cache_worker_name, cache_path, rank

@torch.no_grad()
def extract_features(model, egs, cache_functions, specifier_format, ask_compute="cpu"):
    # extract feat from model def
    extractor = []
    for b in egs:
        f = utils.extract_features_from_decorator(model, b, ask_compute=ask_compute,
                                                  specifier_format=specifier_format,
                                                  cache_funcs=cache_functions, key=lambda wavinfo:os.path.basename(wavinfo.filename))
        if f == None: # already loaded from cache if cuda
            return {}
        extractor.append(f)
    extractor_values = pad_collate_sequence_dict(extractor)
    for k, value in extractor_values.items():
        extractor_values[k] = squeeze_dim(value)
    return extractor_values


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
    seq_shape = [v.shape[-1] for v in seqs]
    Nargmax = np.flatnonzero(seq_shape == np.max(seq_shape))
    N = seq_shape[Nargmax[0]]

    # When there is the signal, the lcm is very high (N // v), when no signal (N=max) (N // v) adds rounding wich is good for lcm
    seq_shape_2 = np.delete(seq_shape, Nargmax)
    Nargmax2 = np.argmax(seq_shape_2)
    N2 = seq_shape_2[Nargmax2]

    hops = np.array([N // v for v in seq_shape])
    hops2 = np.array([N2 // v for v in seq_shape_2])
    exclude_mask = np.in1d(np.arange(len(hops)), Nargmax, invert=True)

    filtered = (np.around( hops[exclude_mask] / (hops2*4) ) * (hops2*4))
    hops[exclude_mask] = filtered

    lcm = np.lcm.reduce(hops)

    # Randomly pickup with the batch_max_steps length of the part
    interval_start = 0
    interval_end = N // lcm - seq_len // lcm
    if max_len != None:
        interval_end = (max_len // lcm) - seq_len // lcm

    if max_len != None and max_len < seq_len:
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
    #  if torch.min(y) < -1.0:
        #  logging.error("min value is ", torch.min(y))
    #  if torch.max(y) > 1.0:
        #  logging.error("max value is ", torch.max(y))

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


def pad_collate_sequence_dict(sequence, padding_value=0):
    """
    Pad a sequence of dictionaries of tensors with padding_value.
    and
    Collate a batch of dictionaries of tensors into a single dictionary of tensors.
    """
    # Collect the keys and values for each tensor in the dictionary
    keys = list(sequence[0].keys())
    values = {k: [s[k] for s in sequence] for k in keys}

    # Pad each tensor in the dictionary separately
    padded_values = {}
    for k, v in values.items():

        # Get the shape of each tensor in the table
        shapes = [tensor.shape for tensor in v]

        #  # Find the dimension where the tensors have different shapes
        diff_dim = None
        for dim in range(len(shapes[0])):
            if len(set(shape[dim] for shape in shapes)) > 1:
                diff_dim = dim
                break

        if diff_dim != None:
            #  # Permute the tensors so that the differing dimension is in the second place from the end
            perm = [j for j in range(len(v[0].shape)) if j != diff_dim]
            perm = perm + [diff_dim]
            for i in range(len(v)):
                  v[i] = v[i].permute(*perm)

        if diff_dim is not None:
            # Pad the tensors along the differing dimension
            max_shape = [max(shape[dim] for shape in shapes) for dim in range(len(shapes[0]))]
            padded_v = [torch.nn.functional.pad(tensor, (0, max_shape[diff_dim]-shape[diff_dim]), mode='constant', value=padding_value) for tensor, shape in zip(v, shapes)]
        else:
            padded_v = v

        padded_v = torch.nn.utils.rnn.pad_sequence(padded_v, batch_first=True)

        padded_v = squeeze_dim(padded_v)

        if diff_dim is not None and len(padded_v.shape) == len(shapes[0]):
            padded_v = padded_v.permute(*perm)

        padded_values[k] = padded_v

    # Combine the padded tensors into a dictionary
    padded_sequence = {k: v for k, v in padded_values.items()}

    return padded_sequence


def squeeze_dim(tensor):
    """
    Removes a dimension of size 1 from a tensor at the first position after the first dimension of size greater than 1.

    Args:
        tensor (torch.Tensor): The tensor to squeeze.

    Returns:
        torch.Tensor: The squeezed tensor.
    """
    dim_to_squeeze = None
    for i, dim_size in enumerate(tensor.shape):
        if dim_size == 1 and i != 0:
            dim_to_squeeze = i
            break

    if dim_to_squeeze is not None:
        tensor = tensor.squeeze(dim=dim_to_squeeze)

    return tensor
