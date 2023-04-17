# Optional list of dependencies required by the package
dependencies = ['torch', 'torchaudio', 'numpy']

from satools.satools.infer_helper import load_model

def anonymization(pretrained=True, tag_version='hifigan_bn_tdnnf_wav2vec2_vq_48_v1', device='cpu'):
    """Loads an anonymization model

    Arguments:
        pretrained (bool): load pretrained weights into the model
        tag_version (str): the tag/release from github corresponding to a model type (check: https://github.com/deep-privacy/SA-toolkit/releases)
        device (str): 'cuda' or 'cpu'
    Returns:
        torch.nn.model
    """
    weight_url = f"https://github.com/deep-privacy/SA-toolkit/releases/download/{tag_version}/final.pt"
    return load_model(weight_url)
