import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
import librosa
from filelock import FileLock

import torch
import numpy as np
import kaldiio

import os
import logging

MAX_WAV_VALUE = 32768.0

f0_stats = None
f0_cache = None
f0_cache_lock = None


def get_f0(
    audio,
    rate=16000,
    interp=False,
    f0_stats_file=None,
    cache_with_filename=None,
):
    global f0_cache
    global f0_cache_lock
    if cache_with_filename != None:
        if f0_cache == None and os.path.exists(".f0_cache"):
            logging.debug("Loading .f0_cache")
            f0_cache = dict(kaldiio.load_ark(".f0_cache"))
        if f0_cache_lock == None:
            f0_cache_lock = FileLock(".f0_cache.lock")

        key = os.path.basename(cache_with_filename)
        if f0_cache != None and key in f0_cache:
            return torch.tensor(f0_cache[key]).unsqueeze(0).unsqueeze(0)

    audio = audio.squeeze().numpy()
    audio *= 2 ** 15

    audio = audio / MAX_WAV_VALUE
    audio = librosa.util.normalize(audio) * 0.95

    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0).numpy()

    frame_length = 20.0
    to_pad = int(frame_length / 1000 * rate) // 2

    f0s = []
    for y in audio.astype(np.float64):
        y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0)
        signal = basic.SignalObj(y_pad, rate)
        pitch = pYAAPT.yaapt(
            signal,
            **{
                "frame_length": frame_length,
                "frame_space": 5.0,
                "nccf_thresh1": 0.25,
                "tda_frame_length": 25.0,
            },
        )
        if interp:
            f0s += [pitch.samp_interp[None, None, :]]
        else:
            f0s += [pitch.samp_values[None, None, :]]

    f0 = np.vstack(f0s)

    f0 = torch.tensor(f0.astype(np.float32))

    if f0_stats_file == None:
        return f0

    global f0_stats
    if f0_stats == None:
        f0_stats = torch.load(f0_stats_file)

    mean = f0_stats["f0_mean"]
    std = f0_stats["f0_std"]

    # Always mean normalize
    ii = f0 != 0
    f0[ii] = (f0[ii] - mean) / std

    if cache_with_filename != None:
        with f0_cache_lock:
            kaldiio.save_ark(
                ".f0_cache",
                {os.path.basename(cache_with_filename): f0.squeeze().numpy()},
                append=True,
            )

    return f0
