import torch, torchaudio
import numpy as np
import tempfile, os, subprocess
import pkwrap
import importlib
from types import SimpleNamespace
import kaldiio
import librosa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def init_asr_model(model, exp_path, pkwrap_vq_dim=-1, get_model_module=False):
    pkwrap_path = pkwrap.__path__[0] + "/../egs/librispeech/v1/"
    model_weight = "final.pt"

    config_path = pkwrap_path + model

    if not os.path.exists(config_path):
        raise FileNotFoundError("No file found at location {}".format(config_path))
    spec = importlib.util.spec_from_file_location("config", config_path)
    asr_model_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(asr_model_file)

    args = SimpleNamespace()
    if pkwrap_vq_dim != -1:
        args = SimpleNamespace(
            freeze_encoder=True,
            codebook_size=pkwrap_vq_dim,
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
    forward, net = pkwrap_chain.get_forward(
        device=device, share_memory=True, get_model_module=True
    )

    return forward, net


def init_synt_model(
    model,
    exp_path,
    asr_bn_model,
    f0_quant_state="exp/f0_vq/g_best",
    model_weight="g_best",
    hifigan_upsample_rates="5, 4, 4, 3, 2",
    asrbn_interpol_bitrate=-1,
):
    pkwrap_path_asr = pkwrap.__path__[0] + "/../egs/librispeech/v1/"
    pkwrap_path = pkwrap.__path__[0] + "/../egs/LJSpeech/"
    model_weight = "/" + model_weight

    config_path = pkwrap_path + model

    if not os.path.exists(config_path):
        raise FileNotFoundError("No file found at location {}".format(config_path))
    spec = importlib.util.spec_from_file_location("config", config_path)
    model_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_file)

    args = SimpleNamespace(
        f0_quant_state=pkwrap_path + "/" + f0_quant_state,
        hifigan_upsample_rates=hifigan_upsample_rates,
        asrbn_interpol_bitrate=asrbn_interpol_bitrate,
    )

    synt_net = model_file.build(args)(
        load_f0_asr_weight=False, asr_bn_model=asr_bn_model
    )

    print("Loading '{}'".format(exp_path + model_weight))
    model_state = torch.load(pkwrap_path + exp_path + model_weight, map_location="cpu")
    synt_net.load_state_dict(model_state["generator"])

    generator = synt_net.to(device)

    generator.eval()
    generator.core_hifigan.remove_weight_norm()
    generator.share_memory()

    @torch.no_grad()
    def _forward(**kwargs):
        y_g_hat = generator(**kwargs)
        if type(y_g_hat) is tuple:
            y_g_hat = y_g_hat[0]
        audios = []
        for i in range(kwargs["f0"].shape[0]):
            audio = y_g_hat[i].squeeze()
            if "real_shape" in kwargs:
                audio = audio[: kwargs["real_shape"][i]]
            audio = audio * pkwrap.hifigan.f0.MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype("int16")
            audio = librosa.util.normalize(audio.astype(np.float32))
            audios.append(audio)
        return audios

    return _forward, generator


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


def computeWER(hyp, ref, diagnosis=True):
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
