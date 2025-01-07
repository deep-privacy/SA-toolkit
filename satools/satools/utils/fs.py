"""
 SCPCache for functions
"""

import uuid
import logging
import copy
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import torch

from tqdm import tqdm

from . import scp_io
# import satools.utils.scp_io as scp_io


def split_data(dirname, num_jobs=0):
    """Call's Kaldi's utils/split_data.sh script

    Args:
        dirname: feature directory to be split
        num_jobs: number of splits
    """
    from .. import script_utils
    run = script_utils.run
    if num_jobs == 0:
        spk2utt_filename = os.path.join(dirname, "spk2utt")
        num_jobs = num_lines(spk2utt_filename)
    script_utils.run(["utils/split_data.sh", dirname, str(num_jobs)])
    return int(num_jobs)


def num_lines(filename):
    try:
        p = subprocess.check_output(["wc", "-l", filename])
        return int(p.decode().strip().split()[0])
    except subprocess.CalledProcessError as cpe:
        sys.exit(cpe.returncode)


def creation_date_file(file):
    created = os.stat(file).st_ctime
    return datetime.fromtimestamp(created)


def scans_directory_for_ext(root_data, extension):
    logging.info(f"Locating {extension}(s)")
    _path = []
    if extension.startswith("."):
        extension = extension[1:]
    pbar = tqdm(Path(root_data).rglob(f"*.{extension}"))
    _count = 0
    for file_path in pbar:
        _count += 1
        _path.append(str(file_path))
        pbar.set_description(f"file count : {_count}")

    return _path


class SCPCache:
    """
    A class for caching the output of a given function in a remote file system using scp_io module.

    Args:
        enabled (bool): A flag to enable/disable caching. Default is True.
        specifier (str): A specifier string in the format 'proto,file'. Default is 'scp,ark:cache.scp,cache.zarr'.
        specifier_formatter (dict): A dictionary of formatting keys and values to format the specifier string.
        key (callable): A function to generate a cache key for the input arguments of the decorated function.

    Example:
        print(extract_stuff(item("testing")))
        print(extract_stuff(item("testing3")))
        print(extract_stuff(item("testing3"))) # cached
        scp_cache.update_formatter({"param": 20})
        print(extract_stuff(item("testing3"))) # not, new file cached
        print(extract_stuff(item("testing2"))) # not, new file cached

        print(extract_stuff(item("coucou"), cache_formatter={"param": 10})) # use other
        print(extract_stuff(item("coucou2"), cache_formatter={"param": 20})) # use other

        reader = {key: value for key, value in scp_io.file_reader_helper("ark:cache_10.zarr")}
        load = scp_io.file_reader_helper("ark:cache_10.zarr")
        print("TEST reader:", reader, "load:", load)
        print(load.get("testing"))
        reader = {key: value for key, value in scp_io.file_reader_helper("scp:cache_10.scp")}
        load = scp_io.file_reader_helper("scp:cache_10.scp")
        print("TEST reader:", reader, "load:", load)
        print(load.get("testing"))
    """

    def __init__(self, enabled=True, specifier="scp,ark:cache.scp,cache.ark", specifier_formatter={}, key=lambda x:x):
        self.enabled = enabled
        self.specifier = specifier
        self.specifier_formatter_to_rw = {}
        self.specifier_formatter = {}
        self.default_formatter = self.hash(specifier_formatter)
        self.specifier_formatter_to_rw[self.default_formatter] = None
        self.specifier_formatter[self.default_formatter] = specifier_formatter
        self.cache_hit = 0
        self.key = key

    def hash(self, specifier_formatter):
        return hash(frozenset(specifier_formatter.items()))

    def __str__(self):
        return f"SCPCache(enabeled={self.enabled}, specifier={self.specifier}, specifier_formatter={self.specifier_formatter}, ref={id(self)})"

    def update_formatter(self, formatter):
        _hash = self.hash(formatter)
        if _hash not in self.specifier_formatter_to_rw:
            self.specifier_formatter_to_rw[_hash] = None
            self.specifier_formatter[_hash] = formatter
            self.default_formatter = _hash
            self.load_reader_writer(_hash)

    def load_reader_writer(self, at_hash):
        if not self.enabled:
            return

        specifier_formatter = copy.deepcopy(self.specifier_formatter[at_hash])
        specifier_formatter.update({"rand": str(uuid.uuid4().fields[-1])[:5]})
        specifier = self.specifier.format(**specifier_formatter)

        self.specifier_formatter_to_rw[at_hash] = (scp_io.file_writer_helper(specifier),
                                                   scp_io.file_reader_helper(specifier))

    def decorate(self):
        def decorator(func):
            def scp_cache_wrapper(*args, **kwargs):
                if not self.enabled:
                    if "cache_formatter" in kwargs:
                        del kwargs["cache_formatter"]
                    if "from_cache_or_none" in kwargs:
                        return None
                    return func(*args, **kwargs)

                if "cache_formatter" in kwargs:
                    _hash = self.hash(kwargs["cache_formatter"])
                    if _hash not in self.specifier_formatter_to_rw:
                        self.load_reader_writer(_hash)
                    del kwargs["cache_formatter"]

                    fw, local_reader = self.specifier_formatter_to_rw[_hash]

                else:
                    if self.specifier_formatter_to_rw[self.default_formatter] == None:
                        self.load_reader_writer(self.default_formatter)
                    fw, local_reader = self.specifier_formatter_to_rw[self.default_formatter]

                if "from_cache_or_none" in kwargs:
                    del kwargs["from_cache_or_none"]
                    _key = self.key(*args, **kwargs)
                    if local_reader.has(_key):
                        return torch.tensor(local_reader.get(_key))
                    return None

                _key = self.key(*args, **kwargs)
                if local_reader.has(_key):
                    return torch.tensor(local_reader.get(_key))
                else:
                    result = func(*args, **kwargs)
                    if torch.is_tensor(result):
                        result = result.detach().cpu()
                    fw[_key] = result.numpy()
                    local_reader.add(_key, fw.file(), fw.id(), fw.reader())
                    return result

            return scp_cache_wrapper
        return decorator

if __name__ == "__main__":
    import numpy as np
    import sys
    class item():
        def __init__(self, a):
            self.key = a
            self.other = "_"

    scp_cache = SCPCache(key=lambda x:x.key, specifier="scp,ark:cache_{param}.scp,cache_{param}.zarr",
                   specifier_formatter={"param":10})
    @scp_cache.decorate()
    def extract_stuff(a):
        print("Compute", a.key, flush=True)
        return torch.tensor([1,2,3,4])

    print(extract_stuff(item("testing")))
    print(extract_stuff(item("testing3")))
    print(extract_stuff(item("testing3"))) # cached
    scp_cache.update_formatter({"param": 20})
    print(extract_stuff(item("testing30"))) # not, new file cached

    print(extract_stuff(item("coucou"), cache_formatter={"param": 10})) # use other
    print(extract_stuff(item("coucou2"), cache_formatter={"param": 20})) # use other

    reader = {key: value for key, value in scp_io.file_reader_helper("ark:cache_10.zarr")}
    load = scp_io.file_reader_helper("ark:cache_10.zarr")
    print("TEST reader:", reader, "load:", load)
    print(load.get("testing"))
    reader = {key: value for key, value in scp_io.file_reader_helper("scp:cache_10.scp")}
    load = scp_io.file_reader_helper("scp:cache_10.scp")
    print("TEST reader:", reader, "load:", load)
    print(load.get("testing"))
