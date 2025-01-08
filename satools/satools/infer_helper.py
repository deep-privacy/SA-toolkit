import os
import os
import torch
import logging
import importlib
import importlib.util

import satools

def load_model(file, load_weight=True, version="v1", from_file=None, option_args=None):
    cpu_cuda_load_opts = {}
    if not torch.cuda.is_available():
        cpu_cuda_load_opts["map_location"] = torch.device('cpu')

    if file.startswith("http"):
        model_dir = os.path.join(torch.hub.get_dir(), 'checkpoints', os.path.basename(os.path.dirname(file)))
        model_state = torch.hub.load_state_dict_from_url(file, model_dir=model_dir, **cpu_cuda_load_opts)
    else:
        if not load_weight:
                file = os.path.join(os.path.dirname(file), "conf.pt")
        if from_file:
            if file.startswith("../../"):
                file = os.path.dirname(__file__) + "/../../egs/PLACE/PLACE/" + file
                file = os.path.normpath(file)
        if not os.path.exists(file):
            url = f"https://github.com/deep-privacy/SA-toolkit/releases/download/{os.path.basename(os.path.dirname(file))}_{version}/final.pt"
            if not load_weight:
                url = f"https://github.com/deep-privacy/SA-toolkit/releases/download/{os.path.basename(os.path.dirname(file))}_{version}/conf.pt"
            os.makedirs(os.path.dirname(file), exist_ok=True)
            logging.warning(f"File {file} does not exsist, attempting to downloading it from github releases..")
            torch.hub.download_url_to_file(url, file, hash_prefix="")
        model_state = torch.load(file, weights_only=False, **cpu_cuda_load_opts)

    install_path = model_state["install_path"]
    install_path_sa = os.path.dirname(os.path.dirname(satools.__path__[0])) # dir to git clone
    if install_path != install_path_sa:
        logging.debug("Loading a model from someone else install")
        install_path = install_path_sa

    config_path = install_path + "/" + model_state["task_path"] + "/" + model_state["base_model_path"]
    logging.info(f"Loading {config_path}")
    if not os.path.exists(config_path):
        logging.info(f"Downloading https://raw.githubusercontent.com/deep-privacy/SA-toolkit/refs/heads/master"+model_state["task_path"] + "/" + model_state["base_model_path"])
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        torch.hub.download_url_to_file(
             "https://raw.githubusercontent.com/deep-privacy/SA-toolkit/refs/heads/master"+ model_state["task_path"] + "/" + model_state["base_model_path"],
             config_path
        )
    spec = importlib.util.spec_from_file_location("config", config_path)
    model_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_file)
    if option_args:
        for k, v in option_args.items():
            model_state['base_model_args'][k] = v
    args = satools.utils.SimpleNamespace(**model_state['base_model_args'])
    net = model_file.build(args)(**model_state["base_model_params"])
    if load_weight:
        net.load_state_dict(model_state["base_model_state_dict"])
    return net
