
# Optional import, not necessary when using torch.hub inference, but required for training/testing
from .utils.generalimport import generalimport
generalimport("feerci")
generalimport("kaldiio")
generalimport("librosa")
generalimport("matplotlib")
generalimport("pandas")
generalimport("scikit_learn")
generalimport("scipy")
generalimport("seaborn")
generalimport("tqdm")

import os
import torch
import logging
import importlib
from types import SimpleNamespace

import satools

def load_model(file, load_weight=True, from_file=None):
    if file.startswith("http"):
        model_state = torch.hub.load_state_dict_from_url(file)
    else:
        if from_file:
            if file.startswith("../../"):
                file = os.path.dirname(__file__) + "/../../egs/PLACE/PLACE/" + file
                file = os.path.normpath(file)
        if not os.path.exists(file):
            logging.warning(f"File {file} does not exsist, attempting to downloading it from github releases..")
            url = f"https://github.com/deep-privacy/SA-toolkit/releases/download/{os.path.basename(os.path.dirname(file))}_v1/final.pt"
            os.makedirs(os.path.dirname(file), exist_ok=True)
            torch.hub.download_url_to_file(url, file, hash_prefix="")
        model_state = torch.load(file)

    install_path = model_state["install_path"]
    install_path_sa = os.path.dirname(os.path.dirname(satools.__path__[0])) # dir to git clone
    if install_path != install_path_sa:
        logging.debug("Loading a model from someone else install")
        install_path = install_path_sa

    config_path = install_path + "/" + model_state["task_path"] + "/" + model_state["base_model_path"]
    logging.info(f"Loading {config_path}")
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

