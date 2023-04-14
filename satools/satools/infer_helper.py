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
        logging.debug("Loading a model from someone else install")
        install_path = model_state["install_path"]

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
