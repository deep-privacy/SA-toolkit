import satools.infer_helper as demo
import satools

import argparse
import json
import sys
import os
from multiprocessing import Manager, Pool
from tqdm import tqdm
from pathlib import Path

import random
import numpy as np
from scipy.io import wavfile

import torch
import torchaudio

from multiprocessing import Process

import satools.utils.ioTools as readwrite
from kaldiio import WriteHelper, ReadHelper


def stream(message):
    sys.stdout.write(f"\r{message}")


def progbar(i, n, size=16):
    done = (i * size) // n
    bar = ""
    for i in range(size):
        bar += "█" if i <= done else "░"
    return bar


def convert(sample):
    waveform, lengths, filename, f0, ys = sample

    asr_out, net = forward_asr(
        waveform.to(demo.device).clone(),
    )
    mat = net.bottleneck_out.squeeze(0).cpu().numpy()

    def parallel_write():
        for i, f in enumerate(filename):
            fname_out_name = Path(f).stem
            readwrite.write_raw_mat(mat, os.path.join(out_dir, fname_out_name + ".ppg"))

    p = Process(target=parallel_write, args=())
    p.start()
    return p


if __name__ == "__main__":

    print("Initializing Inference Process..")

    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)  # don't touch
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--in", type=str, dest="_in")
    parser.add_argument("--in-wavscp", type=str, dest="_in_scp", default=None)
    parser.add_argument("--ext", type=str, dest="ext", default="flac")
    parser.add_argument("--out", type=str, dest="_out")
    parser.add_argument("--vq-dim", type=int, dest="vq_dim")
    parser.add_argument("--model-type", type=str, default="tdnnf")
    args = parser.parse_args()

    global forward_asr
    global out_dir

    #  dim = 128
    #  root_data = "/lium/home/pchampi/lab/asr-based-privacy-preserving-separation/pkwrap/egs/librispeech/v1/corpora/LibriSpeech/train-clean-360"
    #  out_dir = "generated_train-clean-360_vq_" + str(dim)

    audio_extension = args.ext
    dim = args.vq_dim
    out_dir = args._out

    os.makedirs(out_dir, exist_ok=True)

    if args._in_scp != None:
        wavs_path = list(satools.utils.kaldi.read_wav_scp(args._in_scp).values())
        wavs_path = list(demo.split(wavs_path, args.of))[args.part]
        torch_dataset = satools.hifigan.dataset.WavList(
            wavs_path, load_func=satools.utils.kaldi.load_wav_from_scp
        )
    else:
        root_data = args._in
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

            if len(wavs_path) > 10:
                break

        wavs_path = list(demo.split(wavs_path, args.of))[args.part]
        torch_dataset = satools.hifigan.dataset.WavList(wavs_path)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    batch_size = 1  # ONLY a batch_size of 1 is supported!
    dataloader = torch.utils.data.DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=satools.hifigan.dataset.collate_fn_padd(f0_stats=None, get_f0=None),
        persistent_workers=True,
    )

    if args.model_type == "tdnnf":
        if dim == -1 or dim == 0:
            forward_asr, pk_model = demo.init_asr_model(
                model=f"local/chain/e2e/tuning/tdnnf.py",
                exp_path=f"exp/chain/e2e_tdnnf/",
            )
        else:
            forward_asr, pk_model = demo.init_asr_model(
                model=f"local/chain/e2e/tuning/tdnnf_vq_bd.py",
                exp_path=f"exp/chain/e2e_tdnnf_vq_{dim}/",
                pkwrap_vq_dim=dim,
            )
    if args.model_type == "wav2vec2":
        if dim == -1 or dim == 0:
            forward_asr, pk_model = demo.init_asr_model(
                model=f"local/chain/e2e/tuning/tdnnf_wav2vec_fairseq_hibitrate.py",
                exp_path=f"exp/chain/e2e_tdnnf_wav2vec_fairseq_hibitrate/",
            )
        else:
            forward_asr, pk_model = demo.init_asr_model(
                model=f"local/chain/e2e/tuning/tdnnf_wav2vec_fairseq_hibitrate_vq.py",
                exp_path=f"exp/chain/e2e_tdnnf_wav2vec_fairseq_hibitrate_vq_{dim}/",
                pkwrap_vq_dim=dim,
            )

    for i, sample in enumerate(dataloader):
        p = convert(sample)
        bar = progbar(i * batch_size, len(wavs_path))
        message = f"{bar} {i*batch_size}/{len(wavs_path)} "
        stream(message)
    bar = progbar(len(wavs_path), len(wavs_path))
    message = f"{bar} {len(wavs_path)}/{len(wavs_path)} "
    stream(message)
    # wait for last p to write
    p.join()
