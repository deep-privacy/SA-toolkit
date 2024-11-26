"""Wrappers to call kaldi's utils/ scripts"""
import io
import os
import subprocess

import torch
import torchaudio

from ..script_utils import run, read_wav_scp


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


def read_utt2len_file(utt2len_file):
    """read utt2len file, second column is the number of output frames"""
    utt2len = {}
    with open(utt2len_file) as utt2len_f:
        for line in utt2len_f:
            lns = line.strip().split()
            utt2len[lns[0]] = float(lns[1])
    return utt2len



def load_wav_from_scp(wav, frame_offset: int = 0,  num_frames: int = -1):
    """Reads a wav.scp entry like kaldi with embeded unix command
    and returns a pytorch tensor like it was open with torchaudio.load()
    (within some tolerance due to numerical precision))

    import satools
    signalv2, _ = satools.utils.kaldi.load_wav_from_scp("/lium/raid01_b/pchampi/lab/asr-based-privacy-preserving-separation/pkwrap/egs/librispeech/v1/corpora/LibriSpeech/train-clean-360/100/121669/100-121669-0000.flac")
    signalv3, _ = satools.utils.kaldi.load_wav_from_scp("flac -c -d -s /lium/raid01_b/pchampi/lab/asr-based-privacy-preserving-separation/pkwrap/egs/librispeech/v1/corpora/LibriSpeech/train-clean-360/100/121669/100-121669-0000.flac |")
    import torchaudio
    import torch
    signal, _ = torchaudio.load("/lium/raid01_b/pchampi/lab/asr-based-privacy-preserving-separation/pkwrap/egs/librispeech/v1/corpora/LibriSpeech/train-clean-360/100/121669/100-121669-0000.flac")

    print("all close:", torch.allclose(signal, signalv2, rtol=1e-1))
    print("all close:", torch.allclose(signal, signalv3, rtol=1e-1))

    signalv2, _ = satools.utils.kaldi.load_wav_from_scp("/lium/raid01_b/pchampi/lab/asr-based-privacy-preserving-separation/pkwrap/egs/librispeech/v1/corpora/LibriSpeech/train-clean-360/100/121669/100-121669-0000.flac", frame_offset=300, num_frames=1000)
    signalv3, _ = satools.utils.kaldi.load_wav_from_scp("flac -c -d -s /lium/raid01_b/pchampi/lab/asr-based-privacy-preserving-separation/pkwrap/egs/librispeech/v1/corpora/LibriSpeech/train-clean-360/100/121669/100-121669-0000.flac |", frame_offset=300, num_frames=1000)
    signal, _ = torchaudio.load("/lium/raid01_b/pchampi/lab/asr-based-privacy-preserving-separation/pkwrap/egs/librispeech/v1/corpora/LibriSpeech/train-clean-360/100/121669/100-121669-0000.flac", frame_offset=300, num_frames=1000)

    print("all close:", torch.allclose(signal, signalv2, rtol=1e-1))
    print("all close:", torch.allclose(signal, signalv3, rtol=1e-1))                                                                                                                                                                                                          #


    Args:
        wav: a list containing the scp entry

    Returns:
        out_feats: torch.tensor or numpy array dtype float32 (default)
    """
    if wav.strip().endswith("|"):
        devnull = open(os.devnull, "w")
        try:
            wav_read_process = subprocess.Popen(
                wav.strip()[:-1], stdout=subprocess.PIPE, shell=True, stderr=devnull
            )
            sample, sr = torchaudio.load(
                io.BytesIO(wav_read_process.communicate()[0]),
                frame_offset=frame_offset, num_frames=num_frames
            )
        except Exception as e:
            raise IOError("Error processing wav file: {}\n{}".format(wav, e))
    else:
        sample, sr = torchaudio.load(wav, frame_offset=frame_offset, num_frames=num_frames)

    return sample, sr
