import os
import os
import torch
import logging
import importlib
import importlib.util
from types import SimpleNamespace

import satools

def load_model(file, load_weight=True, version="v1", from_file=None):
    if file.startswith("http"):
        model_dir = os.path.join(torch.hub.get_dir(), 'checkpoints', os.path.basename(os.path.dirname(file)))
        model_state = torch.hub.load_state_dict_from_url(file, model_dir=model_dir)
    else:
        if from_file:
            if file.startswith("../../"):
                file = os.path.dirname(__file__) + "/../../egs/PLACE/PLACE/" + file
                file = os.path.normpath(file)
        if not os.path.exists(file):
            logging.warning(f"File {file} does not exsist, attempting to downloading it from github releases..")
            url = f"https://github.com/deep-privacy/SA-toolkit/releases/download/{os.path.basename(os.path.dirname(file))}_{version}/final.pt"
            if not load_model:
                url = f"https://github.com/deep-privacy/SA-toolkit/releases/download/{os.path.basename(os.path.dirname(file))}_{version}/conf.pt"
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

