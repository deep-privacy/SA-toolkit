import logging
import os

import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
import kaldiio
import librosa
import numpy as np
import torch
from filelock import FileLock

logging.getLogger("filelock").setLevel(logging.INFO)

MAX_WAV_VALUE = 32768.0

f0_cache = None
f0_cache_lock = None
log_one = True

yaapt_opts = None
norm_function = None
cache_file = ".f0_cache"


def set_yaapt_opts(opts):
    global yaapt_opts
    yaapt_opts = opts


def set_cache_file(opts):
    global cache_file
    cache_file = opts


def set_norm_func(func):
    global norm_function
    norm_function = func


def m_std_norm(f0, f0_stats, filename):
    if f0_stats == None:
        global log_one
        if log_one:
            log_one = False
            logging.warning("Warning ONLY for F0 mean and std calculation")
        return f0

    mean = f0_stats["f0_mean"]
    std = f0_stats["f0_std"]

    # Always mean normalize
    ii = f0 != 0
    f0[ii] = (f0[ii] - mean) / std
    return f0


def get_f0(
    audio,
    f0_stats,
    rate=16000,
    interp=False,
    cache_with_filename=None,
):

    norm = m_std_norm
    global norm_function
    if norm_function != None:
        norm = norm_function

    global f0_cache
    global f0_cache_lock
    global cache_file
    if cache_with_filename != None:
        if cache_with_filename.endswith("|"):
            cache_with_filename = cache_with_filename.split("/")[-1].split()[0]
        if f0_cache == None:
            if os.path.exists(cache_file):
                logging.debug(f"Loading {cache_file}")
                f0_cache = dict(kaldiio.load_ark(cache_file))
            else:
                f0_cache = {}
        if f0_cache_lock == None:
            f0_cache_lock = FileLock(f"{cache_file}.lock")

        key = cache_with_filename
        if f0_cache != None and key in f0_cache:
            f0 = torch.tensor(f0_cache[key]).unsqueeze(0).unsqueeze(0)
            return norm(f0, f0_stats, cache_with_filename)

    audio = audio.squeeze().numpy()
    audio *= 2 ** 15

    audio = audio / MAX_WAV_VALUE
    audio = librosa.util.normalize(audio) * 0.95

    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0).numpy()

    global yaapt_opts
    if yaapt_opts == None:
        _yaapt_opts = {
            "frame_length": 20.0,
            "frame_space": 5.0,
            "nccf_thresh1": 0.25,
            "tda_frame_length": 25.0,
        }
    else:
        _yaapt_opts = yaapt_opts

    to_pad = int(_yaapt_opts["frame_length"] / 1000 * rate) // 2

    f0s = []
    for y in audio.astype(np.float64):
        y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0)
        try:
            signal = basic.SignalObj(y_pad, rate)
            pitch = pYAAPT.yaapt(
                signal,
                **_yaapt_opts,
            )
        except Exception as e:
            print(
                "Error occured while computing: "
                + str(cache_with_filename)
                + " f0 | "
                + str(audio.shape)
                + "\n",
                flush=True,
            )
            raise e

        if interp:
            f0s += [pitch.samp_interp[None, None, :]]
        else:
            f0s += [pitch.samp_values[None, None, :]]

    f0 = np.vstack(f0s)

    f0 = torch.tensor(f0.astype(np.float32))

    if cache_with_filename != None:
        key = cache_with_filename
        with f0_cache_lock:
            kaldiio.save_ark(
                f"{cache_file}",
                {key: f0.squeeze().numpy()},
                append=True,
            )
            f0_cache[key] = f0.squeeze().numpy()

    return norm(f0, f0_stats, cache_with_filename)
