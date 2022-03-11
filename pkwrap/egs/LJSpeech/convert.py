import pkwrap.infer_helper as demo
import pkwrap

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


def stream(message):
    sys.stdout.write(f"\r{message}")


def progbar(i, n, size=16):
    done = (i * size) // n
    bar = ""
    for i in range(size):
        bar += "█" if i <= done else "░"
    return bar


def convert(sample, target=None):
    waveform, lengths, filename, f0, ys = sample

    if target == None:
        audio = forward_synt(
            audio=waveform.to(demo.device).clone(),
            f0=f0.to(demo.device),
            real_shape=lengths,
        )
    else:
        global wav2utt
        _target = []
        for f in filename:
            _target.append(target[wav2utt[f]])
        audio = forward_synt(
            audio=waveform.to(demo.device).clone(),
            f0=f0.to(demo.device),
            real_shape=lengths,
            target=_target,
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
    parser.add_argument("--in-wavscp", type=str, dest="_in_scp", default=None)
    parser.add_argument("--target_id", type=str, default=None)
    parser.add_argument("--ext", type=str, dest="ext", default="flac")
    parser.add_argument("--out", type=str, dest="_out")
    parser.add_argument("--vq-dim", type=int, dest="vq_dim")
    parser.add_argument("--model-type", type=str, default="tdnnf")
    parser.add_argument(
        "--f0-stats",
        type=str,
        dest="f0_stats",
        default="{'f0_mean': 209.04119886766213, 'f0_std': 58.75603900262766}",
    )
    args = parser.parse_args()

    global forward_synt
    global synthesis_sr
    global wav2utt
    synthesis_sr = 16000
    global out_dir

    f0_stats = json.loads(args.f0_stats.replace("'", '"'))

    #  dim = 128
    #  root_data = "/lium/home/pchampi/lab/asr-based-privacy-preserving-separation/pkwrap/egs/librispeech/v1/corpora/LibriSpeech/train-clean-360"
    #  out_dir = "generated_train-clean-360_vq_" + str(dim)

    audio_extension = args.ext
    dim = args.vq_dim
    out_dir = args._out

    os.makedirs(out_dir, exist_ok=True)

    if args.target_id != None:
        spk2target = pkwrap.utils.kaldi.read_wav_scp(args.target_id)

    if args._in_scp != None:
        wavs_scp = pkwrap.utils.kaldi.read_wav_scp(args._in_scp)
        wav2utt = {"".join(v): k for k, v in wavs_scp.items()}
        wavs_path = list(wavs_scp.values())
        wavs_path = list(demo.split(wavs_path, args.of))[args.part]
        torch_dataset = pkwrap.hifigan.dataset.WavList(
            wavs_path, load_func=pkwrap.utils.kaldi.load_wav_from_scp
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

            #  if len(wavs_path) > 10:
            #  break

        # TODO implement wav2utt required by any to many models
        wavs_path = list(demo.split(wavs_path, args.of))[args.part]
        torch_dataset = pkwrap.hifigan.dataset.WavList(wavs_path)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    batch_size = args.batch_size
    dataloader = torch.utils.data.DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pkwrap.hifigan.dataset.collate_fn_padd(f0_stats),
        persistent_workers=True,
    )

    if args.extract_f0_only:
        print("Only extracting F0 features")
        for i, sample in enumerate(dataloader):
            bar = progbar(i * batch_size, len(wavs_path))
            message = f"{bar} {i*batch_size}/{len(wavs_path)} "
            stream(message)
        sys.exit(0)

    if args.model_type == "tdnnf":
        if dim == -1 or dim == 0:
            forward_asr, pk_model = demo.init_asr_model(
                model=f"local/chain/e2e/tuning/tdnnf.py",
                exp_path=f"exp/chain/e2e_tdnnf/",
                load_model=False,
            )
            forward_synt, synt_model = demo.init_synt_model(
                model=f"local/tuning/hifi_gan.py",
                exp_path=f"exp/hifigan",
                asr_bn_model=pk_model,
                model_weight="g_00102000",
            )
        else:
            forward_asr, pk_model = demo.init_asr_model(
                model=f"local/chain/e2e/tuning/tdnnf_vq_bd.py",
                exp_path=f"exp/chain/e2e_tdnnf_vq_{dim}/",
                pkwrap_vq_dim=dim,
                load_model=False,
            )
            forward_synt, synt_model = demo.init_synt_model(
                model=f"local/tuning/hifi_gan.py",
                exp_path=f"exp/hifigan_vq_{dim}_finetuned",
                asr_bn_model=pk_model,
                model_weight="g_00075000",
            )
    if args.model_type == "libritts_tdnnf":
        if dim == -1 or dim == 0:
            forward_asr, pk_model = demo.init_asr_model(
                model=f"local/chain/e2e/tuning/tdnnf.py",
                exp_path=f"exp/chain/e2e_tdnnf/",
                load_model=False,
            )
            forward_synt, synt_model = demo.init_synt_hifigan_w2v2(
                model=f"local/tuning/hifi_gan_tdnnf.py",
                exp_path=f"exp/hifigan_tdnnf",
                asr_bn_model=pk_model,
                model_weight="g_00045000",
            )
        else:
            forward_asr, pk_model = demo.init_asr_model(
                model=f"local/chain/e2e/tuning/tdnnf_vq_bd.py",
                exp_path=f"exp/chain/e2e_tdnnf_vq_{dim}/",
                pkwrap_vq_dim=dim,
                load_model=False,
            )
            forward_synt, synt_model = demo.init_synt_hifigan_w2v2(
                model=f"local/tuning/hifi_gan_tdnnf.py",
                exp_path=f"exp/hifigan_vq_{dim}/",
                asr_bn_model=pk_model,
                model_weight="g_00045000",
            )
    if args.model_type == "wav2vec2":
        if dim == -1 or dim == 0:
            forward_asr, pk_model = demo.init_asr_model(
                model=f"local/chain/e2e/tuning/tdnnf_wav2vec_fairseq_hibitrate.py",
                exp_path=f"exp/chain/e2e_tdnnf_wav2vec_fairseq_hibitrate/",
                load_model=False,
            )
            forward_synt, synt_model = demo.init_synt_hifigan_w2v2(
                model=f"local/tuning/hifi_gan_wav2vec2.py",
                exp_path=f"exp/hifigan_w2w2",
                asr_bn_model=pk_model,
                model_weight="g_00050000",
            )
        else:
            forward_asr, pk_model = demo.init_asr_model(
                model=f"local/chain/e2e/tuning/tdnnf_wav2vec_fairseq_hibitrate_vq.py",
                exp_path=f"exp/chain/e2e_tdnnf_wav2vec_fairseq_hibitrate_vq_{dim}/",
                pkwrap_vq_dim=dim,
                load_model=False,
            )
            forward_synt, synt_model = demo.init_synt_hifigan_w2v2(
                model=f"local/tuning/hifi_gan_wav2vec2.py",
                exp_path=f"exp/hifigan_w2w2_vq_{dim}",
                asr_bn_model=pk_model,
                model_weight="g_00045000",
            )

    if (args.model_type == "wav2vec2" or args.model_type == "libritts_tdnnf") and os.getenv("TARGET_single", default="false") != "true" and args.f0_stats != parser.get_default("f0_stats"):
        # same as in pkwrap/hifigan/f0.py
        def d(a):
            if a.endswith("|"):
                return a.split("/")[-1].split()[0]
            return a

        keys = dict({oldk: d(v) for oldk, v in wavs_scp.items()})
        filename2wav = dict({keys.get(v): v for k, v in wav2utt.items()})

        def _norm(f0, f0_stats, filename):
            spk_id = spk2target[
                    filename2wav[
                        filename]]
            return pkwrap.hifigan.f0.m_std_norm(f0, f0_stats[spk_id], filename)

        pkwrap.hifigan.f0.set_norm_func(_norm)
    else:
        print("Targetting single speaker F0!", args.f0_stats, "(default LJspeech speaker)")

    for i, sample in enumerate(dataloader):
        if args.target_id != None:
            p = convert(sample, target=spk2target)
        else:
            p = convert(sample)
        bar = progbar(i * batch_size, len(wavs_path))
        message = f"{bar} {i*batch_size}/{len(wavs_path)} "
        stream(message)
    bar = progbar(len(wavs_path), len(wavs_path))
    message = f"{bar} {len(wavs_path)}/{len(wavs_path)} "
    stream(message)
    # wait for last p to write
    p.join()
