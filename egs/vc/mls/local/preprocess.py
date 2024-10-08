import argparse
import os.path
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import resampy
import soundfile as sf
from tqdm import tqdm

import satools


def pad_data(p, out_dir, trim=False, pad=False):
    data, sr = satools.utils.kaldi.load_wav_from_scp(p, out_type="soundfile")
    if sr != 16000:
        data = resampy.resample(data, sr, 16000)
        sr = 16000

    if trim:
        data, _ = librosa.effects.trim(data, 20)

    if pad:
        if data.shape[0] % 1280 != 0:
            data = np.pad(
                data,
                (0, 1280 - data.shape[0] % 1280),
                mode="constant",
                constant_values=0,
            )
        assert data.shape[0] % 1280 == 0

    outpath = Path(os.path.join(out_dir, os.path.splitext(os.path.basename(p))[0] + ".wav"))
    outpath.parent.mkdir(exist_ok=True, parents=True)
    sf.write(outpath, data, sr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--srcdir", type=Path, default="")
    parser.add_argument("--wavscp", type=Path, default="")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--trim", action="store_true")
    parser.add_argument("--pad", action="store_true")
    parser.add_argument("--postfix", type=str, default="wav")
    args = parser.parse_args()

    if args.wavscp:
        # Loading files from wav.scp file
        with open(args.wavscp, 'r') as wav_scp_file:
            files = [" ".join(line.split()[1:]) for line in wav_scp_file]
    elif args.srcdir:
        # Loading files from directory
        files = list(Path(args.srcdir).glob(f"**/*{args.postfix}"))
        files = [p.as_posix() for p in files]
    else:
        raise ValueError("Please provide srcdir or wavscp argument")

    out_dir = Path(args.outdir)

    pad_data_ = partial(pad_data, out_dir=out_dir, trim=args.trim, pad=args.pad)
    with Pool(40) as p:
        rets = list(tqdm(p.imap(pad_data_, files), total=len(files)))


if __name__ == "__main__":
    main()
