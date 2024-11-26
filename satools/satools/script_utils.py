"""
 in this file we have functions that will simply writing
 recipes for chain modelling
"""

try:
    from _satools import kaldi  # lazy import (kaldi-free decoding)
except ImportError as error:
    #  print("ERROR IMPORT KALDI")
    pass

import math
import re
import os
import requests
import subprocess
from urllib.parse import quote
from dataclasses import dataclass
import sys

# copied from kaldi. Copyright and license apply to the original authors
def get_current_num_jobs(it, num_it, start, step, end):
    "Get number of jobs for iteration number 'it' of range('num_it')"

    ideal = float(start) + (end - start) * float(it) / num_it
    if ideal < step:
        return int(0.5 + ideal)
    else:
        return int(0.5 + ideal / step) * step


def get_learning_rate(
    iter,
    num_jobs,
    num_iters,
    num_archives_processed,
    num_archives_to_process,
    initial_effective_lrate,
    final_effective_lrate,
    schedule_type="linear",
):
    """Get learning rate for current iteration based on the learning rate schedule.

    This function implements 3 learning rate schedules (see description of 'schedule_type'
    argument for more information). It returns the learning based on the schedule selected.
    This function is stateless, so we don't assume that the entire learning process
    uses the same schedule.

    Args:
        iter (int): current iteration
        num_jobs (int): number of jobs to be run for this iteration
        num_iters (int): total number of iters
        num_archives_processed (int): number of archives processed till now
        num_archives_to_process (int): number of archives to be processed
        initial_effective_lrate (float): initial learning rate
        final_effective_lrate (float): final learning rate
        schedule_type (str): Default value is 'linear'. The function supports 3 modes
            1. 'none': no change in learning rate. the initial_effective_lrate will be used
            2. 'linear' (default): reduce the learning rate linearly
            3. 'exponential': this is what Kaldi implements. the learning rate reduces exponentially.

    Returns:
        Effective learning rate: learning rate based on the schedule multiplied by the number
            of jobs based on the mode (for linear and exponential only)
    """
    if schedule_type == "none":
        return initial_effective_lrate
    elif schedule_type == "linear":
        epoch_no = (num_archives_processed // num_archives_to_process) + 1
        return (initial_effective_lrate / epoch_no) * num_jobs
    elif schedule_type == "exponential":
        if iter + 1 >= num_iters:
            effective_learning_rate = final_effective_lrate
        else:
            effective_learning_rate = initial_effective_lrate * math.exp(
                num_archives_processed
                * math.log(float(final_effective_lrate) / initial_effective_lrate)
                / num_archives_to_process
            )

        return num_jobs * effective_learning_rate


def get_egs_info(egs_dir):
    """Get number of archives in the egs directory

    This function reads the info/num_archives file in the egs directory

    Args:
        egs_dir: a string containing the path to the egs directory

    Returns:
        Number of archives in the egs directory
    """
    # TODO: use context manager
    ipf = open(os.path.join(egs_dir, "info", "num_archives"))
    num_archives = int(ipf.readline().strip())
    ipf.close()

    return num_archives


def egs_reader(egs_rspec):
    """Read a compressed examples (cegs) file in kaldi

    This function is useful for reading the features and keys
    for decoding the validation set.

    Args:
        egs_rspec: a rspecifier as the ones used in Kaldi. For example, if it is
            a validation diagnostic compressed egs file we may specify it as

            ```
            ark:/path/to/valid_diagnostic.cegs
            ```
    Returns:
        an iterable to iterate over each (key, utterance) tuple in egs_rspec
    """
    reader = kaldi.nnet3.SequentialNnetChainExampleReader(egs_rspec)
    return reader


def egs_reader_gen(egs_rspec):
    """A generator function that calls compressed feat_reader to return pytorch Tensors"""
    reader = egs_reader(egs_rspec)
    while not reader.Done():
        yield reader.Key(), kaldi.chain.GetFeaturesFromCompressedEgs(reader.Value())
        reader.Next()


def feat_reader(feature_rspec):
    """Read a matrix scp file in kaldi

    This function is useful during decoding when we have to read features (during
    training we assume we have egs files, but even that may be changed in future)

    Args:
        feature_rspec: a rspecifier as the ones used in Kaldi. For example, if it is
            an scp file we may specify it as

            ```
            scp:/path/to/feat.scp
            ```

            Similar option exists for ark files as we completely rely on Kaldi's
            functions here.

    Returns:
        an iterable to iterate over each utterance in feature_rspec
    """
    reader = kaldi.matrix.SequentialBaseFloatMatrixReader(feature_rspec)
    return reader


def feat_reader_gen(feature_rspec):
    """A generator function that calls feat_reader to return pytorch Tensors"""
    reader = feat_reader(feature_rspec)
    while not reader.Done():
        yield reader.Key(), kaldi.matrix.KaldiMatrixToTensor(reader.Value())
        reader.Next()


def feat_writer(feature_wspec):
    """Write kaldi matrices to feature_wspec"""
    writer = kaldi.matrix.BaseFloatMatrixWriter(feature_wspec)
    return writer


def run(cmd, quit_on_error=True, shell=False, on_error=lambda x:x):
    """Run a command using subprocess, quit if return code is non-zero"""
    if shell == True and isinstance(cmd, list):
        _str_cmd = [str(c) for c in cmd] # if cmd is pathlib
        cmd = " ".join(_str_cmd)
    p = subprocess.run(cmd, shell=shell, stdout=subprocess.PIPE)
    if quit_on_error and p.returncode != 0:
        import time
        time.sleep(2)
        on_error(p.returncode)
        on_error(p.returncode)
        sys.exit(p.returncode)
    return p


def copy_file(src, dest):
    """Copy a file from source to destination

    This function calls subprocess to use the 'cp' command
    in the shell. In the future we will just use the python
    function.

    Args:
        src: path to source file, a string
        dest: path to destination file, a string
    """
    subprocess.run(["cp", src, dest])


def copy_folder(src, dest):
    """Copy src folder to destination

    This function calls subprocess.run to run 'cp -r' command.
    In the future, we will just use the python function in the
    standard library to do this.

    Args:
        src: source folder
        dest: destination folder
    """
    subprocess.run(["cp", "-r", src, dest])


def read_single_param_file(src, typename=int):
    """Read a file with one value

    This function can be used to read files in Kaldi which
    has parameter values stored in them. E.g. egs/info/num_archives.
    Pass the typename in advance to return the value without errors.

    Args:
        src: the path to file that contains the parameter value
        typename (type): The type of value to be expected in the file.
            Default is int. Any custom value that can take a string
            can be passed.

    Returns:
        Value in the "src" file casted into type "typename"

    Raises:
        AssertionError if parameter value is empty.
    """
    param = None
    with open(src) as ipf:
        param = typename(ipf.readline().strip())
    assert param is not None
    return param


def write_single_param_file(value, filename):
    with open(filename, "w") as opf:
        opf.write("{}".format(value))
        opf.close()


def vartoml(item):
    """
    vartoml allows using variables in TOML config files.
    [default]
    basedir = "/myproject"
    bindir = "${:basedir}/bin"
    datadir = "${:basedir}/data"
    """

    RE_VAR = re.compile(r"""
             [$][{]
             (
                [:]([a-zA-Z0-9_-]+)+     # variable name
             )
             [}]
    """, re.VERBOSE)
    var = {}
    if "var" in dict(item.items()):
        var = dict(item.items())["var"]
    var = dict(var)

    def _var_replace(x):
        if x.groups()[1] in os.environ:
            print(f"Replacing '{x.groups()[0]}' with var from the env: {os.environ[x.groups()[1]]}", file=sys.stderr)
            return os.environ[x.groups()[1]]
        if x.groups()[1] not in var:
            print(f"No {x.groups()[1]} in variable section of config file\nplease add to your config:\n[var]\n{x.groups()[1]} = Something", file=sys.stderr)
            sys.exit(1)
        return var[x.groups()[1]]


    inline_comment_re = re.compile(r"\s+#")
    def rm_inline_comment(value):
        match = inline_comment_re.search(value)
        if match:
            return value[:match.start()].strip()
        else:
            return None

    out = {}
    data = dict(item.items())
    for key, val in data.items():
        out[key] = {}

    for key, val in data.items():
        for _k, _v in dict(val.items()).items():
            wihtout_inline_comment = rm_inline_comment(_v)
            if wihtout_inline_comment:
                value = wihtout_inline_comment
            else:
                value = _v

            out[key][_k] = value
            if re.search(RE_VAR, value):
                r = re.sub(RE_VAR, _var_replace, value)
                out[key][_k] = r
    return out


def get_github_repo():
    ''' Retrieve the repo part of the git URL '''
    cmd = ['git', 'remote', 'get-url', 'origin']
    repo = subprocess.check_output(cmd).decode('utf-8').strip()
    repo = re.search(r'github.com.(.+?)\.git$', repo)
    if repo:
        return repo.group(1)
    return "deep-privacy/sa-toolkit"

def get_github_token():
    try:
        GITHUB_TOKEN = os.environ['GITHUB_TOKEN']
    except KeyError:
        sys.stderr.write('ERROR: This utility requires a Github '
                         'token for accessing the Github release API.\n')
        sys.stderr.write('       The variable should be held in an '
                         'environment variable named GITHUB_TOKEN.\n')
        sys.exit(1)
    return GITHUB_TOKEN


def get_release(tag_name):
    ''' Retrieve the upload URL for the given tag '''
    res = requests.get(
        'https://api.github.com/repos/{}/releases/tags/{}'.format(get_github_repo(), tag_name),
        headers=dict(Authorization='token {}'.format(get_github_token()),
                     Accept='application/vnd.github.v3+json'))

    if res.status_code == 401:
        raise RuntimeError('Github API error: {}'.format(res.json()["message"]))
    if res.status_code < 400:
        return res.json()

    target_commitish = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()

    # If the tag name does not exist, create a new release
    res = requests.post(
        'https://api.github.com/repos/{}/releases'.format(get_github_repo()),
        headers=dict(Authorization='token {}'.format(get_github_token()),
                     Accept='application/vnd.github.v3+json'),
        json=dict(tag_name=tag_name, target_commitish=target_commitish))

    res = requests.get(
        'https://api.github.com/repos/{}/releases/tags/{}'.format(get_github_repo(), tag_name),
        headers=dict(Authorization='token {}'.format(get_github_token()),
                     Accept='application/vnd.github.v3+json'))

    if res.status_code == 401:
        raise RuntimeError('Github API error: {}'.format(res.json()["message"]))
    if res.status_code < 400:
        return res.json()

    raise RuntimeError('Could not locate tag name: {}, error: {}'.format(tag_name, str(res.json())))


def upload_asset(url, filename, _as=None):
    '''
    Upload an asset to a release

    POST :server/repos/:owner/:repo/releases/:release_id/assets?name=:asset_filename

    '''
    upload_url = url + '?name={}'.format(quote(os.path.split(filename)[-1]))
    if _as:
        upload_url = url + '?name={}'.format(quote(_as))
    with open(filename, 'rb') as asset_file:
        requests.post(
            upload_url,
            headers={'Authorization': 'token {}'.format(get_github_token()),
                     'Accept': 'application/vnd.github.v3+json',
                     'Content-Type': 'application/octet-stream'},
            data=asset_file
        )

def delete_asset(asset_id):
    ''' Delete the resource at the given ID '''
    requests.delete(
        'https://api.github.com/repos/{}/releases/assets/{}'.format(get_github_repo(), asset_id),
        headers=dict(Authorization='token {}'.format(get_github_token()),
                     Accept='application/vnd.github.v3+json'),
        json=dict(asset_id=asset_id))

def push_github_model(tag_name, up_assets, up_as_name={}, force=True):
    release = get_release(tag_name)
    upload_url = release['upload_url'].split('{')[0]
    gh_assets = {x['name']: x for x in release['assets']}

    for asset in up_assets:
        filename = os.path.split(asset)[-1]
        if asset in up_as_name:
            filename = up_as_name[asset]
        if filename in gh_assets:
            if force: #  or asset in up_as_name
                delete_asset(gh_assets[filename]['id'])
            else:
                sys.stderr.write("Github WARNING: Asset already exists: {}\n".format(asset))
                continue

        if asset in up_as_name:
            sys.stderr.write("Github UPLOADING: {}\n".format(up_as_name[asset]))
            upload_asset(upload_url, asset, _as=up_as_name[asset])
        else:
            sys.stderr.write("Github UPLOADING: {}\n".format(asset))
            upload_asset(upload_url, asset)


def can_be_cast_to_list(value, to_type=str):
    try:
        return True, [to_type(char) for char in value if char.isdigit()]
    except:
        return False, None


def can_be_cast_to_int(value):
    try:
        return True, int(value)
    except:
        return False, None


class ngpu:
    pass


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def find_conv_bool(value):
    if isinstance(value, str):
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
    return value


@dataclass
class ConfigParser:
    __was_parserd__: bool = True

    def get_forcmd(self, key):
        if getattr(self, key) == "":
            return []
        return [f"--{key.replace('_', '-')}", str(getattr(self, key)).replace("\n", "")]

    def load_from_config(self, cfg):
        for key, value in cfg.items():
            if hasattr(self, key):
                type_of_value = self.__annotations__[key]
                if type_of_value == bool:
                    setattr(self, key, str2bool(value))
                    continue
                if type_of_value == ngpu:
                    from safe_gpu import safe_gpu
                    if isinstance(value, str):
                        free_gpus = safe_gpu.get_free_gpus()
                        if value.lower() == 'all' and len(free_gpus) != 0:
                            setattr(self, key, free_gpus)
                            continue
                        if value.lower() == 'all-force':
                            gpus = {uuid.strip(): idx.strip() for idx, uuid in (line.split(',') for line in subprocess.check_output(
                                ["nvidia-smi", "--format=csv,noheader", "--query-gpu=index,gpu_bus_id"]).decode().strip().split('\n'))}
                            setattr(self, key, list(gpus.values()))
                            continue
                    castable, newvalue = can_be_cast_to_int(value)
                    if castable:
                        nfreeg = safe_gpu.get_free_gpus()
                        if newvalue > len(nfreeg):
                            raise Exception(f"Not enough free gpu")
                        if newvalue == 0:
                            raise Exception(f"0 not a valide value")
                        setattr(self, key, nfreeg[:newvalue])
                        continue
                    castable, newvalue = can_be_cast_to_list(value)
                    if castable:
                        nfreeg = safe_gpu.get_free_gpus()
                        if len(newvalue) > len(nfreeg):
                            raise Exception(
                                f"Not enough free gpu (free:{nfreeg}), you may try 'all-force' to use all gpus even if they are already in use.")
                        for g in newvalue:
                            if g not in nfreeg:
                                raise Exception(f"GPU id {g} not available")
                        if len(newvalue) != 0:
                            setattr(self, key, newvalue)
                            continue
                    raise Exception(f"Not a valid option of ngpu: {value}, you may try 'all-force' to use all gpus even if they are already in use.")

                # Normal processing
                setattr(self, key, type_of_value(value))
        return self


def split_array(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))

def split_dict(a, n):
    keys = list(a.keys())
    k, m = divmod(len(keys), n)
    return [
        {key: a[key] for key in keys[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]}
        for i in range(n)
    ]


def read_wav_scp(wav_scp):
    """Reads wav.scp file and returns a dictionary

    Args:
        wav_scp: a string, contains the path to wav.scp

    Returns:
        utt2wav: a dictionary, keys are the first column of wav.scp
            and values are the second column
    """
    utt2wav = {}
    with open(wav_scp) as ipf:
        for line in ipf:
            lns = line.strip().split()
            uttname = lns[0]
            utt2wav[uttname] = " ".join(lns[1:])
    return utt2wav
