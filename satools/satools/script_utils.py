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
import os
import subprocess
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
        time.sleep(1)
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
    import re
    import toml
    decoder = toml.TomlDecoder()

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

    out = {}
    data = dict(item.items())
    for key, val in data.items():
        out[key] = {}

    for key, val in data.items():
        for _k, _v in dict(val.items()).items():
            out[key][_k] = _v
            if re.search(RE_VAR, _v):
                r = re.sub(RE_VAR, _var_replace, _v)
                out[key][_k] = r
    return out
