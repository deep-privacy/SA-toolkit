

import os
import torch
import logging
import importlib
from types import SimpleNamespace
import satools

def load_model(file, load_weight=True):
    if file.startswith("http"):
        model_state = torch.hub.load_state_dict_from_url(file)
    else:
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
