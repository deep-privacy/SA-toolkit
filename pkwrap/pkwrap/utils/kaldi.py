"""Wrappers to call kaldi's utils/ scripts"""
from ..script_utils import run
import subprocess
import configparser
import torch
import soundfile
import io
import os
import sys


def split_data(data_folder, num_jobs):
    run(
        [
            "utils/split_data.sh",
            data_folder,
            f"{num_jobs}",
        ]
    )
    sdata = "{}/split{}".format(data_folder, num_jobs)
    return sdata


def get_frame_shift(data_folder):
    process = run(["utils/data/get_frame_shift.sh", data_folder])
    return float(process.stdout.decode("utf-8"))


def get_utt2dur(data_folder):
    run(
        [
            "utils/data/get_utt2dur.sh",
            data_folder,
        ]
    )


def gen_utt2len(data_folder):
    run(
        [
            "feat-to-len",
            f"{data_folder}/feats.scp",
            "ark,t:{data_folder}/utt2len",
        ]
    )


def filter_scp(id_list=None, exclude=False, input_list=None, output_list=None):
    with open(output_list, "w") as opf:
        cmd = ["utils/filter_scp.pl"]
        if exclude:
            cmd.append("--exclude")
        cmd += [id_list, input_list]
        subprocess.run(cmd, stdout=opf)
        opf.close()


def shuffle(input_file, output_file):
    with open(output_file, "w") as opf:
        subprocess.run(["utils/shuffle_list.pl", input_file], stdout=opf)
        opf.close()


def split_scp(input_file, prefix="", suffix="", num_splits=-1):
    if num_splits < 1:
        raise ValueError("num splits should be positive integer")
    if num_splits == 1:
        raise NotImplementedError
    out_scp = [f"{prefix}{i}{suffix}" for i in range(1, num_splits + 1)]

    subprocess.run(["utils/split_scp.pl", input_file, *out_scp])


def parseval(s):
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    return s


def read_kaldi_conf(filename):
    with open(filename) as f:
        file_content = "[dummy_section]\n" + f.read()
    config = configparser.RawConfigParser()
    config.read_string(file_content)
    config = config["dummy_section"]
    return {
        k.replace("--", "").replace("-", "_"): parseval(v) for k, v in config.items()
    }


def kaldifeat_set_option(opts, filename):
    for kaldi_opts, val in read_kaldi_conf(filename).items():
        # translate to python option lib 'kaldifeat'
        translate_opts = {"sample_frequency": "samp_freq", "num_mel_bins": "num_bins"}
        kaldi_opts = (
            translate_opts[kaldi_opts] if kaldi_opts in translate_opts else kaldi_opts
        )

        if kaldi_opts in dir(opts):
            setattr(opts, kaldi_opts, val)
        elif kaldi_opts in dir(opts.frame_opts):
            setattr(opts.frame_opts, kaldi_opts, val)
        elif kaldi_opts in dir(opts.mel_opts):
            setattr(opts.mel_opts, kaldi_opts, val)
        else:
            logging.error(
                f"Kaldi option '{kaldi_opts}' not compatible with python lib 'kaldifeat'"
            )
            logging.error(
                f"Change the translation dict according to https://github.com/csukuangfj/kaldifeat/blob/bac4db61c3d0d72e28fa3713c4c234ffa3dca4cb/doc/source/usage.rst#fbankoptions"
            )
            sys.exit(1)

    return opts


def read_wav_scp(wav_scp):
    """Reads wav.scp file and returns a dictionary

    Args:
        wav_scp: a string, contains the path to wav.scp

    Returns:
        utt2wav: a dictionary, keys are the first column of wav.scp
            and values are the second column
    """
    utt2wav = {}
    with open(wav_scp) as ipf:
        for line in ipf:
            lns = line.strip().split()
            uttname = lns[0]
            utt2wav[uttname] = " ".join(lns[1:])
    return utt2wav


def load_wav_from_scp(wav):
    """Reads a wav.scp entry like kaldi with embeded unix command
    and returns a pytorch tensor like it was open with torchaudio.load()
    (within some tolerance due to numerical precision)

    signal, _ = torchaudio.load("XX/1272-128104-0000.flac")
    signalv2 = prepare('flac -c -d -s XX/1272-128104-0000.flac |')
    signalv3 = prepare('XX/1272-128104-0000.flac')

    print("all close:", torch.allclose(signal, signalv2, rtol=1e-1))
    print("all close:", torch.allclose(signal, signalv3, rtol=1e-1))

    Args:
        wav: a list containing the scp entry

    Returns:
        feats_torch torch.tensor dtype float32
    """
    if wav.strip().endswith("|"):
        devnull = open(os.devnull, "w")
        try:
            wav_read_process = subprocess.Popen(
                wav.strip()[:-1], stdout=subprocess.PIPE, shell=True, stderr=devnull
            )
            sample, sr = soundfile.read(
                io.BytesIO(wav_read_process.communicate()[0]),
            )
        except Exception as e:
            raise IOError("Error processing wav file: {}\n{}".format(wav, e))
    else:
        sample, sr = soundfile.read(wav)
    feats_torch = torch.tensor(
        sample, dtype=torch.float32, requires_grad=False
    ).unsqueeze(0)
    return feats_torch, sr
