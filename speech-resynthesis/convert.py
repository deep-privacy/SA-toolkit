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
from torch.utils.data import Dataset, DataLoader

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


class WavList(Dataset):
    def __init__(self, wavs_path):
        self.wavs_path = wavs_path

    def __len__(self):
        return len(self.wavs_path)

    def __getitem__(self, idx):
        filename = self.wavs_path[idx]
        waveform, sr = torchaudio.load(filename)
        return (waveform, filename)


def collate_fn_padd(batch):
    filenames = [b[1] for b in batch]
    batch = [b[0] for b in batch]
    """
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    """
    ## get sequence lengths
    lengths = torch.tensor([t.shape[1] for t in batch])
    ## padd
    batch = [torch.Tensor(t).permute(1, 0) for t in batch]
    feats = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    ## compute mask
    mask = batch != 0
    feats = torch.squeeze(feats.permute(0, 2, 1))

    if len(feats.shape) == 1:
        feats = torch.unsqueeze(feats, 0)

    f0s = []
    for i, b in enumerate(batch):
        f0s.append(
            demo.get_f0(
                b.permute(0, 1),
                f0_stats_file="datasets/LJSpeech/f0_stats.pth",
                _device="cpu",
                cache_with_filename=filenames[i],
            )
            .squeeze(dim=1)
            .permute(1, 0),
        )

    f0spad = torch.nn.utils.rnn.pad_sequence(f0s, batch_first=True, padding_value=0)
    f0s = f0spad.permute(0, 2, 1)

    return feats, lengths, mask, filenames, f0s


def convert(sample):
    waveform, lengths, mask, filename, f0 = sample
    out, net = pk_model(waveform.to(demo.device))

    audio = synthesis_model(
        net.bottleneck_out.permute(0, 2, 1).contiguous(),
        f0.to(demo.device),
        real_shape=lengths,
    )

    for i, f in enumerate(filename):
        fname_out_name = Path(f).stem
        output_file = os.path.join(out_dir, fname_out_name + "_gen.wav")
        wavfile.write(output_file, synthesis_sr, audio[i])


if __name__ == "__main__":

    print("Initializing Inference Process..")

    """
    Multi node (3 here) F0 extraction (CPU only and very intensive):
    python3 ./convert.py  --part 0 --of 3 --extract-f0-only
    python3 ./convert.py  --part 1 --of 3 --extract-f0-only
    python3 ./convert.py  --part 2 --of 3 --extract-f0-only

    Convert (also possible to do multi node conversion):
    python3 ./convert.py --num-workers 4 --batch-size 64
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--extract-f0-only", action="store_true")
    args = parser.parse_args()

    global synthesis_model
    global synthesis_sr
    global pk_model
    global out_dir

    dim = "128"
    synt_model = "g_00087000"
    root_data = "/lium/home/pchampi/lab/asr-based-privacy-preserving-separation/pkwrap/egs/librispeech/v1/corpora/LibriSpeech/train-clean-360"
    audio_extension = "flac"
    out_dir = "generated_train-clean-360_test"

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
        #  if wav_count >= 10:
        #  break

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    wavs_path = list(split(wavs_path, args.of))[args.part]

    batch_size = args.batch_size
    dataloader = DataLoader(
        WavList(wavs_path),
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_padd,
        persistent_workers=True,
    )

    if args.extract_f0_only:
        print("Only extracting F0 features")
        for i, sample in enumerate(dataloader):
            bar = progbar(i * batch_size, len(wavs_path))
            message = f"{bar} {i*batch_size}/{len(wavs_path)} "
            stream(message)
        sys.exit(0)

    synthesis_model, synthesis_sr, _ = demo.init_speech_synthesis_model(
        f"checkpoints/lj_vq_tdnnf_asr_vq_{dim}/config.json",
        f"checkpoints/lj_vq_tdnnf_asr_vq_{dim}/{synt_model}",
    )

    pk_model = demo.init_pkwrap_model(
        model=f"local/chain/e2e/tuning/tdnnf_vq.py",
        exp_path=f"exp/chain/e2e_tdnnf_vq_sizeco_{dim}/",
        pkwrap_vq_dim=dim,
    )

    for i, sample in enumerate(dataloader):
        convert(sample)
        bar = progbar(i * batch_size, len(wavs_path))
        message = f"{bar} {i*batch_size}/{len(wavs_path)} "
        stream(message)
    bar = progbar(len(wavs_path), len(wavs_path))
    message = f"{bar} {len(wavs_path)}/{len(wavs_path)} "
    stream(message)
