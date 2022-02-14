import pkwrap.infer_helper as demo
import pkwrap

import argparse
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

    audio = forward_synt(
        audio=waveform.to(demo.device).clone(),
        f0=f0.to(demo.device),
        real_shape=lengths,
    )

    def parallel_write():
        for i, f in enumerate(filename):
            fname_out_name = Path(f).stem
            output_file = os.path.join(out_dir, fname_out_name + "_gen.wav")
            wavfile.write(output_file, synthesis_sr, audio[i])

    p = Process(target=parallel_write, args=())
    p.start()
    return p


if __name__ == "__main__":

    print("Initializing Inference Process..")

    """
    Multi node (3 here) F0 extraction (CPU only and very intensive):
    python3 ./convert.py  --part 0 --of 3 --extract-f0-only
    python3 ./convert.py  --part 1 --of 3 --extract-f0-only
    python3 ./convert.py  --part 2 --of 3 --extract-f0-only

    Convert with high batch-size (you can also do multi node conversion with part/of):
    python3 ./convert.py --num-workers 4 --batch-size 64
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--extract-f0-only", action="store_true")
    parser.add_argument("--in", type=str, dest="_in")
    parser.add_argument("--out", type=str, dest="_out")
    parser.add_argument("--vq-dim", type=int, dest="vq_dim")
    args = parser.parse_args()

    global forward_synt
    global synthesis_sr
    synthesis_sr = 16000
    global out_dir

    f0_stats_file = "exp/f0_stats.pth"

    #  dim = 128
    #  root_data = "/lium/home/pchampi/lab/asr-based-privacy-preserving-separation/pkwrap/egs/librispeech/v1/corpora/LibriSpeech/train-clean-360"
    #  out_dir = "generated_train-clean-360_vq_" + str(dim)

    audio_extension = "flac"
    dim = args.vq_dim
    root_data = args._in
    out_dir = args._out

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

    wavs_path = list(demo.split(wavs_path, args.of))[args.part]

    batch_size = args.batch_size
    dataloader = torch.utils.data.DataLoader(
        pkwrap.hifigan.dataset.WavList(wavs_path),
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pkwrap.hifigan.dataset.collate_fn_padd(f0_stats_file),
        persistent_workers=True,
    )

    if args.extract_f0_only:
        print("Only extracting F0 features")
        for i, sample in enumerate(dataloader):
            bar = progbar(i * batch_size, len(wavs_path))
            message = f"{bar} {i*batch_size}/{len(wavs_path)} "
            stream(message)
        sys.exit(0)

    forward_asr, pk_model = demo.init_asr_model(
        model=f"local/chain/e2e/tuning/tdnnf_vq_bd.py",
        exp_path=f"exp/chain/e2e_tdnnf_vq_{dim}/",
        pkwrap_vq_dim=dim,
    )
    forward_synt, synt_model = demo.init_synt_model(
        model=f"local/tuning/hifi_gan.py",
        exp_path=f"exp/hifigan_vq_{dim}_finetuned",
        asr_bn_model=pk_model,
        model_weight="g_00075000",
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