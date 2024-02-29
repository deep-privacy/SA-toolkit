import os
import sys
import torch
import http.client
import json
# List of dependencies required by the package
dependencies = ['torch', 'torchaudio', 'soundfile', 'numpy', 'configargparse']

hub_repo_name = "deep-privacy_SA-toolkit"

def anonymization(pretrained=True, tag_version='hifigan_bn_tdnnf_wav2vec2_vq_48_v1', device='cpu', exit_if_new_version=True):
    """Loads an anonymization model

    Arguments:
        pretrained (bool): load pretrained weights into the model
        tag_version (str): the tag/release from github corresponding to a model type (check: https://github.com/deep-privacy/SA-toolkit/releases)
        device (str): 'cuda' or 'cpu'
    Returns:
        torch.nn.model
    """

    os.environ["SA_JIT_TWEAK"] = "true"

    if check_new_commit_github(save_sha=not exit_if_new_version) and exit_if_new_version:
        print(f"'torch.hub.load' called with 'exit_if_new_version' enabled, existing..", file=sys.stderr)
        sys.exit(1)

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

def check_new_commit_github(save_sha):
    conn = http.client.HTTPSConnection("api.github.com")
    try:
        conn.request("GET", "/repos/deep-privacy/SA-toolkit/branches/master", headers={"User-Agent": "python/torch_hub_info"})
        resp = conn.getresponse()
        body = resp.read()
        github_sha = json.loads(body)["commit"]["sha"]
    except:
        return False

    local_hub_dir = torch.hub.get_dir()
    latest_commit_info = os.path.join(local_hub_dir, "latest_commit_"+hub_repo_name)
    cache_commit = github_sha
    if os.path.exists(latest_commit_info):
        with open(latest_commit_info) as ipf:
            cache_commit = ipf.readline().strip()

    if (save_sha or not os.path.exists(latest_commit_info)) and cache_commit == github_sha:
        with open(latest_commit_info, "w") as opf:
            opf.write("{}".format(github_sha))

    if cache_commit != github_sha:
        print("!!!!!!", file=sys.stderr)
        print(f"A new commit is available for 'deep-privacy/SA-toolkit'\nPlease use 'torch.hub.load' with 'force_reload=True' to get the latest version!", file=sys.stderr)
        print("!!!!!!", file=sys.stderr)
    return cache_commit != github_sha

