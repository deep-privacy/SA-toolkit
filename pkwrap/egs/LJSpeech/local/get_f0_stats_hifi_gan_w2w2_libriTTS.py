import pkwrap
from tqdm import tqdm
import soundfile as sf
from multiprocessing import Pool
from pathlib import Path
import argparse
import numpy
import math
import torch
import os
import json

def calc_stats(file):
    x, sr = sf.read(file)
    f0 = pkwrap.hifigan.f0.get_f0(torch.tensor(x), None, cache_with_filename=str(file))
    f0 = f0[f0 > 1.]
    mu, std, var = f0.mean(), f0.std(), f0.var()

    return {
        "filename": os.path.basename(file),
        "mu_s" : mu.item(),
        "std_s": std.item(),
        "var_s": var.item(),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--srcdir", type=Path, required=True)
    parser.add_argument("--outstats", type=Path, required=True)
    parser.add_argument("--postfix", type=str, default="wav")
    args = parser.parse_args()
    files = list(Path(args.srcdir).glob(f"**/*{args.postfix}"))

    pkwrap.hifigan.f0.set_yaapt_opts({
            "frame_length": 35.0,
            "frame_space": 20.0,
            "nccf_thresh1": 0.25,
            "tda_frame_length": 25.0,
        })


    with Pool(2) as p:
        rets = list(tqdm(p.imap(calc_stats, files), total=len(files)))

    speakers_mu = {}
    speakers_var = {}
    for r in rets:
        spk = r["filename"].split("_")[0]
        if spk not in speakers_mu:
            speakers_mu[spk] = []
        if spk not in speakers_var:
            speakers_var[spk] = []

        speakers_var[spk].append(r["var_s"])
        speakers_mu[spk].append(r["mu_s"])

    stats = {}
    for k in speakers_mu.keys():
        stats[k] = {}
        stats[k]["f0_mean"] = numpy.array(speakers_mu[k]).mean()
        stats[k]["f0_std"] = math.sqrt(numpy.array(speakers_var[k]).mean())

    json_object = json.dumps(stats)

    with open(args.outstats, "w") as outfile:
        outfile.write(json_object)

if __name__ == "__main__":
    main()
