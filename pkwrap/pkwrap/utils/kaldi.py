"""Wrappers to call kaldi's utils/ scripts"""
from ..script_utils import run
import subprocess
import configparser
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
