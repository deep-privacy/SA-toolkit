import os
import sys
import torch
# Optional list of dependencies required by the package
dependencies = ['torch', 'torchaudio', 'numpy']

hub_repo_name = "deep-privacy_SA-toolkit"

def anonymization(pretrained=True, tag_version='hifigan_bn_tdnnf_wav2vec2_vq_48_v1', device='cpu'):
    """Loads an anonymization model

    Arguments:
        pretrained (bool): load pretrained weights into the model
        tag_version (str): the tag/release from github corresponding to a model type (check: https://github.com/deep-privacy/SA-toolkit/releases)
        device (str): 'cuda' or 'cpu'
    Returns:
        torch.nn.model
    """

    
    local_hub_dir = torch.hub.get_dir()
    dir_list = os.listdir(local_hub_dir)
    matching_dirs = [d for d in dir_list if d.startswith(hub_repo_name)]

    if len(matching_dirs) > 0:
        sys.path.insert(0,f"{os.path.join(local_hub_dir, matching_dirs[0])}/satools/")
        if len(matching_dirs) > 1:
            print(f"Multple '{hub_repo_name}' repo in {local_hub_dir}, using '{sys.path[0]}'")
    else:
        print(f"No matching {local_hub_dir} + '{hub_repo_name}*' directory found.")
        sys.exit(1)
    from satools.infer_helper import load_model

    weight_url = f"https://github.com/deep-privacy/SA-toolkit/releases/download/{tag_version}/final.pt"
    return load_model(weight_url)
