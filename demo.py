import torch, torchaudio
import numpy as np
import tempfile, os, subprocess
import json

import re
import sys

import pkwrap
import importlib
from types import SimpleNamespace
import kaldiio

import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
import librosa
from scipy.io import wavfile

import logging

logging.getLogger("matplotlib").setLevel(level=logging.CRITICAL)

MAX_WAV_VALUE = 32768.0

speech_resynthesis = __import__("speech-resynthesis")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_speech_synthesis_model(config_file, weight_file, root_dir=os.getcwd()):
    b = os.getcwd()
    os.chdir(root_dir)

    try:
        with open(config_file) as f:
            data = f.read()
    except Exception as e:
        os.chdir(b)
        raise e
    json_config = json.loads(data)
    h = speech_resynthesis.utils.AttrDict(json_config)

    generator = speech_resynthesis.models.CodeGenerator(h).to(device)
    print("Loading '{}'".format(weight_file))
    checkpoint_dict = torch.load(weight_file, map_location="cpu")
    os.chdir(b)

    generator.load_state_dict(checkpoint_dict["generator"])
    generator.eval()
    generator.remove_weight_norm()

    @torch.no_grad()
    def _forward(bn, f0):
        spk = torch.tensor([[0]]).to(device)  # single speaker for now
        y_g_hat = generator(asr_bn=bn, f0=f0, spkr=spk)
        if type(y_g_hat) is tuple:
            y_g_hat = y_g_hat[0]
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype("int16")
        audio = librosa.util.normalize(audio.astype(np.float32))
        return audio

    return _forward, h.sampling_rate


def init_pkwrap_model(model, exp_path, pkwrap_vq_dim):
    pkwrap_path = pkwrap.__path__[0] + "/../egs/librispeech/v1/"
    model_weight = "final.pt"

    config_path = pkwrap_path + model

    if not os.path.exists(config_path):
        raise FileNotFoundError("No file found at location {}".format(config_path))
    spec = importlib.util.spec_from_file_location("config", config_path)
    asr_model_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(asr_model_file)

    args = SimpleNamespace()
    if int(pkwrap_vq_dim) != -1:
        args = SimpleNamespace(
            freeze_encoder=True,
            codebook_size=int(pkwrap_vq_dim),
        )

    asr_net = asr_model_file.build(args)

    print("Loading '{}'".format(exp_path + model_weight))

    pkwrap_chain = pkwrap.chain.ChainE2EModel(
        asr_net,
        cmd_line=False,
        **{
            "dir": pkwrap_path + exp_path,
            "base_model": pkwrap_path + exp_path + model_weight,
        },
    )
    net = pkwrap_chain.get_forward(device=device)

    return net


def kaldi_asr_decode(out, get_align=False):
    kaldiark = tempfile.NamedTemporaryFile(suffix=".ark").name
    writer = kaldiio.WriteHelper(f"ark,t:{kaldiark}")
    writer("test_utts", out[0].cpu().numpy())
    writer.close()

    pkwrap_path = pkwrap.__path__[0] + "/../egs/librispeech/v1/"  # kaldi/pkwrap egs dir

    res = subprocess.run(
        f"cat {kaldiark} | grep -m1 nan",
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        shell=True,
    )
    if res.returncode == 0:
        return ""

    decode = f"cd {pkwrap_path}; . ./path.sh; cat {kaldiark} | {pkwrap_path}/shutil/decode/latgen-faster-mapped.sh {pkwrap_path}/exp/chain/e2e_biphone_tree/graph_tgsmall/words.txt {pkwrap_path}/exp/chain/e2e_tdnnf/0.trans_mdl exp/chain/e2e_biphone_tree/graph_tgsmall/HCLG.fst {kaldiark}.ark_lat.1.gz"
    text = subprocess.check_output(decode + " 2>&1| grep ^test_utts", shell=True)
    text = text.decode("UTF-8").replace("test_utts", "").replace("\\n", "").strip()

    if get_align:
        res = subprocess.run(
            f"""
            cd {pkwrap_path}; . ./path.sh; \
            ./local/show_align_fromlat.sh {kaldiark}.ark_lat.1.gz
            """,
            shell=True,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )
        if res.returncode != 0:
            print("Error getting phone alignment")

        f = open(f"{kaldiark}.ark_lat.1.gz.out_state_seq", "r")
        lines = f.readlines()[0]
        return text, lines.replace("  ", " ").split(" ")[1:-1]

    return text


f0_stats = None


def get_f0(audio, rate=16000, interp=False, f0_stats_file=None):
    audio = audio.squeeze().numpy()
    audio *= 2 ** 15

    audio = audio / MAX_WAV_VALUE
    audio = librosa.util.normalize(audio) * 0.95

    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0).numpy()

    frame_length = 20.0
    to_pad = int(frame_length / 1000 * rate) // 2

    f0s = []
    for y in audio.astype(np.float64):
        y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0)
        signal = basic.SignalObj(y_pad, rate)
        pitch = pYAAPT.yaapt(
            signal,
            **{
                "frame_length": frame_length,
                "frame_space": 5.0,
                "nccf_thresh1": 0.25,
                "tda_frame_length": 25.0,
            },
        )
        if interp:
            f0s += [pitch.samp_interp[None, None, :]]
        else:
            f0s += [pitch.samp_values[None, None, :]]

    f0 = np.vstack(f0s)

    f0 = torch.tensor(f0.astype(np.float32)).to(device)

    if f0_stats_file == None:
        return f0

    global f0_stats
    if f0_stats == None:
        f0_stats = torch.load(f0_stats_file)

    mean = f0_stats["f0_mean"]
    std = f0_stats["f0_std"]

    # Always mean normalize (TODO add other type of norm like in speech-resynthesis/dataset.py)
    ii = f0 != 0
    f0[ii] = (f0[ii] - mean) / std

    return f0


if __name__ == "__main__":

    waveform, sr = torchaudio.load(
        "pkwrap/egs/librispeech/v1/data/vctk_test/wav/p227/p227_001_mic2.wav"
    )

    synthesis_model, synthesis_sr = init_speech_synthesis_model(
        "checkpoints/lj_vq_tdnnf_asr/config.json",
        "checkpoints/lj_vq_tdnnf_asr/g_00080000",
        root_dir="speech-resynthesis/",
    )

    pk_model = init_pkwrap_model(
        model="local/chain/e2e/tuning/tdnnf.py",
        exp_path="exp/chain/e2e_tdnnf/",
        pkwrap_vq_dim="-1",
    )

    out, net = pk_model(waveform.to(device))
    text = kaldi_asr_decode(out, rescore=True)

    f0 = get_f0(
        waveform, f0_stats_file="speech-resynthesis/datasets/LJSpeech/f0_stats.pth"
    )
    audio = synthesis_model(net.bottleneck_out.permute(0, 2, 1).contiguous(), f0)

    wavfile.write("test.wav", synthesis_sr, audio)
    print(text)


"""
For IPyhon dislpay of WER:
"""


def ComputeEditDistanceMatrix(hs, rs):
    """Compute edit distance between two list of strings.
    Args:
      hs: the list of words in the hypothesis sentence
      rs: the list of words in the reference sentence
    Returns:
      Edit distance matrix (in the format of list of lists), where the first
      index is the reference and the second index is the hypothesis.
    """
    dr, dh = len(rs) + 1, len(hs) + 1
    dists = [[]] * dr

    # Initialization.
    for i in range(dr):
        dists[i] = [0] * dh
        for j in range(dh):
            if i == 0:
                dists[0][j] = j
            elif j == 0:
                dists[i][0] = i

    # Do dynamic programming.
    for i in range(1, dr):
        for j in range(1, dh):
            if rs[i - 1] == hs[j - 1]:
                dists[i][j] = dists[i - 1][j - 1]
            else:
                tmp0 = dists[i - 1][j - 1] + 1
                tmp1 = dists[i][j - 1] + 1
                tmp2 = dists[i - 1][j] + 1
                dists[i][j] = min(tmp0, tmp1, tmp2)

    return dists


def _GenerateAlignedHtml(hyp, ref, err_type):
    """Generate a html element to highlight the difference between hyp and ref.
    Args:
      hyp: Hypothesis string.
      ref: Reference string.
      err_type: one of 'none', 'sub', 'del', 'ins'.
    Returns:
      a html string where disagreements are highlighted.
        - hyp highlighted in green, and marked with <del> </del>
        - ref highlighted in yellow
    """

    highlighted_html = ""
    if err_type == "none":
        highlighted_html += "%s " % hyp

    elif err_type == "sub":
        highlighted_html += """<span style="background-color: greenyellow">
        <del>%s</del></span><span style="background-color: yellow">
        %s </span> """ % (
            hyp,
            ref,
        )

    elif err_type == "del":
        highlighted_html += """<span style="background-color: yellow">
        %s</span> """ % (
            ref
        )

    elif err_type == "ins":
        highlighted_html += """<span style="background-color: greenyellow">
        <del>%s</del> </span> """ % (
            hyp
        )

    else:
        raise ValueError("unknown err_type " + err_type)

    return highlighted_html


def GenerateSummaryFromErrs(nref, errs):
    """Generate strings to summarize word errors.
    Args:
      nref: integer of total words in references
      errs: dict of three types of errors. e.g. {'sub':10, 'ins': 15, 'del': 3}
    Returns:
      str1: string summarizing total error, total word, WER,
      str2: string breaking down three errors: deleting, insertion, substitute
    """

    total_error = sum(errs.values())
    str_sum = "total error = %d, total word = %d, wer = %.2f%%" % (
        total_error,
        nref,
        total_error * 100.0 / nref,
    )

    str_details = "Error breakdown: del = %.2f%%, ins=%.2f%%, sub=%.2f%%" % (
        errs["del"] * 100.0 / nref,
        errs["ins"] * 100.0 / nref,
        errs["sub"] * 100.0 / nref,
    )

    return str_sum, str_details


def ComputeWER(hyp, ref, diagnosis=True):
    """Computes WER for ASR by ignoring diff of punctuation, space, captions.
    Args:
      hyp: Hypothesis string.
      ref: Reference string.
      diagnosis (optional): whether to generate diagnosis str (in html format)
    Returns:
      dict of three types of errors. e.g. {'sub':0, 'ins': 0, 'del': 0}
      num of reference words, integer
      aligned html string for diagnois (empty if diagnosis = False)
    """

    # Compute edit distance.
    hs = hyp.split()
    rs = ref.split()
    distmat = ComputeEditDistanceMatrix(hs, rs)

    # Back trace, to distinguish different errors: insert, deletion, substitution.
    ih, ir = len(hs), len(rs)
    errs = {"sub": 0, "ins": 0, "del": 0}
    aligned_html = ""
    while ih > 0 or ir > 0:
        err_type = ""

        # Distinguish error type by back tracking
        if ir == 0:
            err_type = "ins"
        elif ih == 0:
            err_type = "del"
        else:
            if hs[ih - 1] == rs[ir - 1]:  # correct
                err_type = "none"
            elif distmat[ir][ih] == distmat[ir - 1][ih - 1] + 1:  # substitute
                err_type = "sub"
            elif distmat[ir][ih] == distmat[ir - 1][ih] + 1:  # deletion
                err_type = "del"
            elif distmat[ir][ih] == distmat[ir][ih - 1] + 1:  # insert
                err_type = "ins"
            else:
                raise ValueError("fail to parse edit distance matrix")

        # Generate aligned_html
        if diagnosis:
            if ih == 0 or not hs:
                tmph = " "
            else:
                tmph = hs[ih - 1]
            if ir == 0 or not rs:
                tmpr = " "
            else:
                tmpr = rs[ir - 1]
            aligned_html = _GenerateAlignedHtml(tmph, tmpr, err_type) + aligned_html

        # If no error, go to previous ref and hyp.
        if err_type == "none":
            ih, ir = ih - 1, ir - 1
            continue

        # Update error.
        errs[err_type] += 1

        # Adjust position of ref and hyp.
        if err_type == "del":
            ir = ir - 1
        elif err_type == "ins":
            ih = ih - 1
        else:  # err_type == 'sub'
            ih, ir = ih - 1, ir - 1

    assert distmat[-1][-1] == sum(errs.values())

    # Num of words. For empty ref we set num = 1.
    nref = max(len(rs), 1)

    if aligned_html:
        str1, str2 = GenerateSummaryFromErrs(nref, errs)
        aligned_html = str1 + " (" + str2 + ")" + "<br>" + aligned_html

    return errs, nref, aligned_html
