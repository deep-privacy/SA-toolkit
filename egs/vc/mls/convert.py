import argparse
import json
import os
import random
import sys
from multiprocessing import Process
from pathlib import Path

import numpy as np
import satools.infer_helper as demo
import torch
from scipy.io import wavfile
from tqdm import tqdm

import satools


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
    parser.add_argument("--vq-dim", type=int, dest="vq_dim", default=-1)
    parser.add_argument("--dp-e", type=int, dest="dp_dim", default=0)
    parser.add_argument("--model-type", type=str, default="tdnnf")
    parser.add_argument(
        "--rand-pitch", type=str, default="False", help="Randomize the pitch shape"
    )
    parser.add_argument(
        "--dp-pitch",
        type=str,
        default="0",
        help="Modify the pitch shape with DP models",
    )
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

    if os.path.exists(args.f0_stats):
        # File provided for f0_stats, open it before going through json loading
        with open(args.f0_stats, 'r') as f0_stats_file:
            f0_stats = json.loads(f0_stats_file.read())
    else:
        f0_stats = json.loads(args.f0_stats.replace("'", '"'))


    #  dim = 128
    #  root_data = "/lium/home/pchampi/lab/asr-based-privacy-preserving-separation/satools/egs/librispeech/v1/corpora/LibriSpeech/train-clean-360"
    #  out_dir = "generated_train-clean-360_vq_" + str(dim)

    audio_extension = args.ext
    dim = args.vq_dim
    dp_dim = args.dp_dim
    out_dir = args._out

    os.makedirs(out_dir, exist_ok=True)

    if args.target_id != None:
        spk2target = satools.utils.kaldi.read_wav_scp(args.target_id)

    if args._in_scp != None:
        wavs_scp = satools.utils.kaldi.read_wav_scp(args._in_scp)
        wav2utt = {"".join(v): k for k, v in wavs_scp.items()}
        wavs_path = list(wavs_scp.values())
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

            #  if len(wavs_path) > 10:
            #  break

        # TODO implement wav2utt required by any to many models
        wavs_path = list(demo.split(wavs_path, args.of))[args.part]
        torch_dataset = satools.hifigan.dataset.WavList(wavs_path)

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
        collate_fn=satools.hifigan.dataset.collate_fn_padd(f0_stats),
        persistent_workers=True,
    )

    if args.rand_pitch.lower() == "true" or args.dp_pitch != "0":
        if args.rand_pitch.lower() == "true":
            target_noise_db = 5  # HARD CODED
            #  target_noise_db = 15  # HARD CODED
            print("---> TARGET_NOISE_DB:", target_noise_db)

        print("Apply modification on the F0 shape")

    if args.extract_f0_only:
        print("Only extracting F0 features")
        satools.hifigan.f0.set_yaapt_opts({
            "frame_length": 35.0,
            "frame_space": 20.0,
            "nccf_thresh1": 0.25,
            "tda_frame_length": 25.0,
        })
        for i, sample in enumerate(dataloader):
            bar = progbar(i * batch_size, len(wavs_path))
            message = f"{bar} {i*batch_size}/{len(wavs_path)} "
            stream(message)
        sys.exit(0)

    # Loading model configuration from json config file
    with open("convert_config.json", 'r') as config_file:
        config = json.loads(config_file.read())
    vq = False
    dp = False
    if dim > 0:
        vq = True
    elif dp_dim > 0:
        dp = True

    models_list = config["convert_config"]["models_list"]
    # Select right model according to parameters
    for cur_model in models_list:
        if cur_model["model_name"] == args.model_type \
                and cur_model["vq"] == vq \
                and cur_model["dp"] == dp:
            asr_model_config = cur_model["asr_model"]
            synt_model_config = cur_model["synt_model"]
            break
    else:
        raise ValueError(f"Model type {args.model_type} not found in config file")

    forward_asr, pk_model = demo.init_asr_model(
        model=asr_model_config["model"],
        exp_path=eval(compile(asr_model_config["exp_path"], 'exp_path', 'eval')) if asr_model_config["exp_path"][0] == "f" else asr_model_config["exp_path"],
        load_model=asr_model_config["load_model"],
        egs_path=asr_model_config["egs_path"]
    )
    forward_synt, synt_model = demo.init_synt_hifigan_w2v2(
        model=synt_model_config["model"],
        exp_path=eval(compile(synt_model_config["exp_path"], 'exp_path', 'eval')) if synt_model_config["exp_path"][0] == "f" else synt_model_config["exp_path"],
        asr_bn_model=pk_model,
        model_weight=synt_model_config["model_weight"],
        egs_path=synt_model_config["egs_path"],
        json_stats_file=synt_model_config["json_stats_file"],
    )

    if (
        (args.model_type == "wav2vec2" or args.model_type == "libritts_tdnnf" or args.model_type == "wav2vec2_mls")
        and os.getenv("TARGET_single", default="false") != "true"
        and args.f0_stats != parser.get_default("f0_stats")
    ):
        # same as in satools/hifigan/f0.py
        def d(a):
            if a.endswith("|"):
                return a.split("/")[-1].split()[0]
            return a

        keys = dict({oldk: d(v) for oldk, v in wavs_scp.items()})
        filename2wav = dict({keys.get(v): v for k, v in wav2utt.items()})

        if args.dp_pitch != "0":
            model_data_dir = satools.__path__[0] + "/../../F0_DP"
            sys.path.insert(0, model_data_dir)

            #  This code was kindly provided by the autor of:
            #  @article{differentially_speaker_anon,
            #    title={Differentially Private Speaker Anonymization},
            #    author={Ali Shahin Shamsabadi and Brij Mohan Lal Srivastava and Aurélien Bellet and Nathalie Vauquier and Emmanuel Vincent and Mohamed Maouche and Marc Tommasi and Nicolas Papernot},
            #    year={2022},
            #  }
            #
            # this module was purposly not included in this SA-toolkit
            from f0DP import load_AE, DP_F0

            DP_eps = float(args.dp_pitch)

            conf_filepath = os.path.join(model_data_dir, "parameters_gae.conf")
            gae_path = os.path.join(model_data_dir, f"DP{DP_eps}gae_model.pth")
            dp_device = torch.device("cpu")
            print(f"Loading DP-Pitch AE model from {gae_path}")
            gae_model = load_AE(conf_filepath, dp_device, gae_path, DP_eps)

        def _norm(f0, f0_stats, filename):
            spk_id = spk2target[filename2wav[filename]]
            pitch = f0

            if args.dp_pitch == "0":
                pitch = satools.hifigan.f0.m_std_norm(pitch, f0_stats[spk_id], filename)

            if args.rand_pitch.lower() == "true":
                # Set a target channel noise power to something very noisy
                # Convert to linear Watt units
                target_noise_watts = 10 ** (target_noise_db / 10)
                # Generate noise samples
                mean_noise = 0
                noise_volts = np.random.normal(
                    mean_noise, np.sqrt(target_noise_watts), len(pitch)
                )

                ii = pitch == 0
                pitch = pitch + torch.tensor(noise_volts, dtype=pitch.dtype)
                pitch[ii] = 0

            if args.dp_pitch != "0":
                pitch = pitch.squeeze()

                # Have a record of indexes for puting back zeros later
                idx_zero = np.where(pitch == 0)[0]
                idx_nonzero = np.where(pitch != 0)[0]
                idxs = np.concatenate((idx_zero, idx_nonzero))
                # Getting the log of only nonzero f0
                src_f0_nonzeros = pitch[pitch > 0]
                src_f0_zeros = pitch[pitch == 0]
                src_f0_nonzeros_rec = DP_F0(src_f0_nonzeros, gae_model, dp_device)
                src_f0_nonzeros_rec = (
                    src_f0_nonzeros_rec * pitch[pitch > 0].numpy().std()
                ) + pitch[pitch > 0].numpy().mean()

                # Put zero and nozero back together
                src_f0_all_rec = np.concatenate((src_f0_zeros, src_f0_nonzeros_rec))
                pitch = torch.tensor(
                    [
                        src_f0_all_rec
                        for _, src_f0_all_rec in sorted(zip(idxs, src_f0_all_rec))
                    ]
                )

                pitch = satools.hifigan.f0.m_std_norm(pitch, f0_stats[spk_id], filename)
                pitch = pitch.unsqueeze(dim=0).unsqueeze(dim=0)

            return pitch

        satools.hifigan.f0.set_norm_func(_norm)
    else:
        print(
            "Targetting single speaker F0!", args.f0_stats, "(default LJspeech speaker)"
        )

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
