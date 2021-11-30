import demo

import argparse
import random
import numpy as np
import torch
import torchaudio
import sys
import os
from tqdm import tqdm
from pathlib import Path
import argparse
from multiprocessing import Manager, Pool
from scipy.io import wavfile
from itertools import islice

torch.set_num_threads(1)


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def stream(message):
    sys.stdout.write(f"\r{message}")


def progbar(i, n, size=16):
    done = (i * size) // n
    bar = ""
    for i in range(size):
        bar += "█" if i <= done else "░"
    return bar


def convert(id):
    filename = wavs_path[id]
    waveform, sr = torchaudio.load(filename)
    out, net = pk_model(waveform.to(demo.device))

    f0 = demo.get_f0(waveform, f0_stats_file="datasets/LJSpeech/f0_stats.pth")
    audio = synthesis_model(net.bottleneck_out.permute(0, 2, 1).contiguous(), f0)

    fname_out_name = Path(filename).stem
    output_file = os.path.join(out_dir, fname_out_name + "_gen.wav")
    wavfile.write(output_file, synthesis_sr, audio)


if __name__ == "__main__":

    print("Initializing Inference Process..")

    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--of", type=int, default=10)
    args = parser.parse_args()

    global synthesis_model
    global synthesis_sr
    global pk_model
    global wavs_path
    global out_dir

    dim = "128"
    synt_model = "g_00087000"
    root_data = "/lium/home/pchampi/lab/asr-based-privacy-preserving-separation/pkwrap/egs/librispeech/v1/corpora/LibriSpeech/train-clean-360"
    audio_extension = "flac"
    out_dir = "generated_train-clean-360"

    os.makedirs(out_dir, exist_ok=True)

    print(f"Locating {audio_extension}(s)")
    wavs_path = []
    wav_count = 0
    pbar = tqdm(os.walk(root_data))
    for root, dirs, files in pbar:
        if Path(root).parent == Path(root_data):
            dataset = root.split("/")[-1]
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.splitext(file_path)[1] == f".{audio_extension}":
                wav_count += 1
                pbar.set_description(f"audio file count : {wav_count}")
                wavs_path.append(file_path)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    synthesis_model, synthesis_sr, _ = demo.init_speech_synthesis_model(
        f"checkpoints/lj_vq_tdnnf_asr_vq_{dim}/config.json",
        f"checkpoints/lj_vq_tdnnf_asr_vq_{dim}/{synt_model}",
    )

    pk_model = demo.init_pkwrap_model(
        model=f"local/chain/e2e/tuning/tdnnf_vq.py",
        exp_path=f"exp/chain/e2e_tdnnf_vq_sizeco_{dim}/",
        pkwrap_vq_dim=dim,
    )

    wavs_path = list(split(wavs_path, args.of))[args.part]

    for i in range(len(wavs_path)):
        convert(i)
        bar = progbar(i, len(wavs_path))
        message = f"{bar} {i}/{len(wavs_path)} "
        stream(message)
