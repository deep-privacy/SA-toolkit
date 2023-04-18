import os
import sys
import torch
# Optional list of dependencies required by the package
dependencies = ['torch', 'torchaudio', 'numpy']

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
    matching_dirs = [d for d in dir_list if d.startswith('deep-privacy_SA-toolkit')]

    if len(matching_dirs) > 0:
        sys.path.insert(0,f"{os.path.join(local_hub_dir, matching_dirs[0])}/deep-privacy_SA-toolkit_refactoring/satools/")
    else:
        print(f"No matching {local_hub_dir} + 'deep-privacy_SA-toolkit*' directory found.")
        sys.exit(1)
    import satools

    weight_url = f"https://github.com/deep-privacy/SA-toolkit/releases/download/{tag_version}/final.pt"
    return satools.infer_helper.load_model(weight_url)

