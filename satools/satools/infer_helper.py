import importlib
import json
import os
import logging
import subprocess
import tempfile
from types import SimpleNamespace

import kaldiio
import librosa
import numpy as np
import torch

import satools

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def load_model(file, load_weight=True):
    model_state = torch.load(file)

    install_path = os.path.dirname(os.path.dirname(satools.__path__[0])) # dir to git clone
    if model_state["install_path"] != install_path:
        logging.debug("Loading a model from someone else")

    config_path = install_path + "/" + model_state["task_path"] + "/" + model_state["base_model_path"]
    logging.info(f"Loading {config_path} with args \"{str(model_state['base_model_args'])}\" and params \"{str(model_state['base_model_params'])}\"" )
    if not os.path.exists(config_path):
        raise FileNotFoundError("No file found at location {}".format(config_path))
    spec = importlib.util.spec_from_file_location("config", config_path)
    model_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_file)
    args = SimpleNamespace(**model_state['base_model_args'])
    net = model_file.build(args)(**model_state["base_model_params"])
    if load_weight:
        net.load_state_dict(model_state["base_model_state_dict"])
    return net

def init_asr_model(
    model,
    exp_path,
    dp_dim=-1,
    vq_dim=-1,
    get_model_module=False,
    load_model=True,
    egs_path="asr/librispeech/",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    satools_path = os.path.join(satools.__path__[0], "../../egs", egs_path)
    model_weight = "final.pt"

    config_path = os.path.join(satools_path, model)

    if not os.path.exists(config_path):
        raise FileNotFoundError("No file found at location {}".format(config_path))
    spec = importlib.util.spec_from_file_location("config", config_path)
    asr_model_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(asr_model_file)

    args = SimpleNamespace()
    if dp_dim != -1:
        args = SimpleNamespace(
            freeze_encoder=True,
            epsilon=str(dp_dim),
        )
    if vq_dim != -1:
        args = SimpleNamespace(
            freeze_encoder=True,
            codebook_size=vq_dim,
        )

    asr_net = asr_model_file.build(args)

    if load_model:
        print("Loading '{}'".format(os.path.join(exp_path, model_weight)))

    satools_chain = satools.chain.ChainE2EModel(
        asr_net,
        cmd_line=False,
        **{
            "dir": os.path.join(satools_path, exp_path),
            "base_model": os.path.join(satools_path, exp_path, model_weight),
        },
    )
    forward, net = satools_chain.get_forward(
        device=device,
        share_memory=True,
        get_model_module=True,
        load_model=load_model,
    )

    return forward, net


def init_synt_hifigan_w2v2(
    model,
    exp_path,
    asr_bn_model,
    model_weight,
    json_stats_file=None,
    no_spk_info=False,
    egs_path="vc/libritts/",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    satools_path = os.path.join(satools.__path__[0], "../../egs", egs_path)
    config_path = os.path.join(satools_path, model)

    if not os.path.exists(config_path):
        raise FileNotFoundError("No file found at location {}".format(config_path))
    spec = importlib.util.spec_from_file_location("config", config_path)
    model_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_file)

    args = SimpleNamespace(no_spk_info=no_spk_info)

    satools.hifigan.f0.set_cache_file(".f0_sr-320.cache")
    satools.hifigan.f0.set_yaapt_opts(
        {
            "frame_length": 35.0,
            "frame_space": 20.0,
            "nccf_thresh1": 0.25,
            "tda_frame_length": 25.0,
        }
    )

    if json_stats_file == None:
        f0_stats = open(f"{satools_path}/data/LibriTTS/stats.json", "r").readline()
    else:
        f0_stats = open(json_stats_file).readline()
    spkids = list(json.loads(f0_stats).keys())
    spkids.sort()

    synt_net = model_file.build(args, spkids)(
        load_asr_weight=False, asr_bn_model=asr_bn_model
    )

    print("Loading '{}'".format(os.path.join(exp_path, model_weight)))
    model_state = torch.load(os.path.join(satools_path, exp_path, model_weight), map_location="cpu")
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
            audio = audio * satools.hifigan.f0.MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype("int16")
            audio = librosa.util.normalize(audio.astype(np.float32))
            audios.append(audio)
        return audios

    return _forward, generator


def init_synt_model(
    model,
    exp_path,
    asr_bn_model,
    f0_quant_state="exp/f0_vq/g_best",
    model_weight="g_best",
    hifigan_upsample_rates="5, 4, 4, 3, 2",
    asr_interpol_bitrate=-1,
    egs_path="vc/libritts/",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    satools_path = os.path.join(satools.__path__[0], "../../egs", egs_path)
    model_weight = "/" + model_weight

    config_path = satools_path + model

    if not os.path.exists(config_path):
        raise FileNotFoundError("No file found at location {}".format(config_path))
    spec = importlib.util.spec_from_file_location("config", config_path)
    model_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_file)

    args = SimpleNamespace(
        f0_quant_state=satools_path + "/" + f0_quant_state,
        hifigan_upsample_rates=hifigan_upsample_rates,
        asr_interpol_bitrate=asr_interpol_bitrate,
    )

    synt_net = model_file.build(args)(
        load_f0_asr_weight=False, asr_bn_model=asr_bn_model
    )

    print("Loading '{}'".format(exp_path + model_weight))
    model_state = torch.load(satools_path + exp_path + model_weight, map_location="cpu")
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
            audio = audio * satools.hifigan.f0.MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype("int16")
            audio = librosa.util.normalize(audio.astype(np.float32))
            audios.append(audio)
        return audios

    return _forward, generator


def kaldi_asr_decode(out, get_align=False, egs_path="asr/librispeech/"):
    kaldiark = tempfile.NamedTemporaryFile(suffix=".ark").name
    writer = kaldiio.WriteHelper(f"ark,t:{kaldiark}")
    writer("test_utts", out[0].detach().cpu().numpy())
    writer.close()

    satools_path = os.path.join(satools.__path__[0], "../../egs", egs_path)  # kaldi/satools egs dir

    res = subprocess.run(
        f"cat {kaldiark} | grep -m1 nan",
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        shell=True,
    )
    if res.returncode == 0:
        return ""

    decode = f"cd {satools_path}; . ./path.sh; cat {kaldiark} | {satools_path}/shutil/decode/latgen-faster-mapped.sh {satools_path}/exp/chain/e2e_biphone_tree/graph_tgsmall/words.txt {satools_path}/exp/chain/e2e_tdnnf/0.trans_mdl exp/chain/e2e_biphone_tree/graph_tgsmall/HCLG.fst {kaldiark}.ark_lat.1.gz"
    text = subprocess.check_output(decode + " 2>&1| grep ^test_utts", shell=True)
    text = text.decode("UTF-8").replace("test_utts", "").replace("\\n", "").strip()

    if get_align:
        res = subprocess.run(
            f"""
            cd {satools_path}; . ./path.sh; \
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
