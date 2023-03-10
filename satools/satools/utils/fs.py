"""
 wrap scripts in kaldi utils directory
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

import satools.utils.scp_io as scp_io


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

def scp_cache(enabled=True, specifier="scp,ark:cache_213.scp,cache_213.h5", specifier_formatter=None, key=lambda x:x):
    """
    A decorator function for caching the output of a given function in a remote file system using scp_io module.

    Args:
        enabled (bool): A flag to enable/disable caching. Default is True.
        specifier (str): A specifier string in the format 'proto,file'. Default is 'scp,ark:cache.scp,cache.h5'.
        specifier_formatter (dict): A dictionary of formatting keys and values to format the specifier string.
        key (callable): A function to generate a cache key for the input arguments of the decorated function.

    Returns:
        callable: A decorator function that takes a callable and returns a callable.

    Example:
        @scp_cache(key=lambda x:x.key)
        def extract_stuff(a):
            return torch.tensor([1,2,3,4])
        
        @scp_cache(key=lambda x:x.key, specifier="scp,ark:cache_{param}.scp,cache_{param}.h5",
                   specifier_formatter=lambda:{"param":10})
        def extract_stuff(a):
            return torch.tensor([1,2,3,4])

        print(extract_stuff(item("testing")))
        print(extract_stuff(item("testing3")))
        print(extract_stuff(item("testing3"))) # cached

        reader = {key: value for key, value in scp_io.file_reader_helper("ark:cache_10.h5")}
        load = scp_io.file_reader_helper("ark:cache.h5")
        print("TEST reader:", reader, "load:", load)
        print(load.get("testing"))
        reader = {key: value for key, value in scp_io.file_reader_helper("scp:cache_10.scp")}
        load = scp_io.file_reader_helper("scp:cache.scp")
        print("TEST reader:", reader, "load:", load)
        print(load.get("testing"))
    """
    if not enabled:
        def decorator(func):
            def wrapper(item):
                return func(item)
            return wrapper
        return decorator

    if specifier_formatter != None:
        sp = {"rand": str(uuid.uuid4().fields[-1])[:5]}
        sp.update(specifier_formatter())
        specifier = specifier.format(**sp)

    def decorator(func):
        fw = scp_io.file_writer_helper(specifier)
        local_reader = scp_io.file_reader_helper(specifier)
        def wrapper(item):

            _key = key(item)
            if local_reader.has(_key):
                return local_reader.get(_key)
            else:
                result = func(item)
                if isinstance(result, torch.Tensor):
                    _result = result.detach().cpu()
                fw[_key] = _result
                local_reader.add(_key, fw.file(), _key, fw.reader())
            return result

        return wrapper
    return decorator




class SCPCache:
    """
    A class for caching the output of a given function in a remote file system using scp_io module.

    Args:
        enabled (bool): A flag to enable/disable caching. Default is True.
        specifier (str): A specifier string in the format 'proto,file'. Default is 'scp,ark:cache.scp,cache.h5'.
        specifier_formatter (dict): A dictionary of formatting keys and values to format the specifier string.
        key (callable): A function to generate a cache key for the input arguments of the decorated function.

    Example:
        @SCPCache(key=lambda x:x.key).decorate()
        def extract_stuff(a):
            return torch.tensor([1,2,3,4])
        
        @SCPCache(key=lambda x:x.key, specifier="scp,ark:cache_{param}.scp,cache_{param}.h5",
                   specifier_formatter={"param":10}).decorate()
        def extract_stuff(a):
            return torch.tensor([1,2,3,4])

        print(extract_stuff(item("testing")))
        print(extract_stuff(item("testing3")))
        print(extract_stuff(item("testing3"))) # cached

        reader = {key: value for key, value in scp_io.file_reader_helper("ark:cache.h5")}
        load = scp_io.file_reader_helper("ark:cache.h5")
        print("TEST reader:", reader, "load:", load)
        print(load.get("testing"))
        reader = {key: value for key, value in scp_io.file_reader_helper("scp:cache.scp")}
        load = scp_io.file_reader_helper("scp:cache.scp")
        print("TEST reader:", reader, "load:", load)
        print(load.get("testing"))
    """

    def __init__(self, enabled=True, specifier="scp,ark:cache.scp,cache.h5", specifier_formatter=None, key=lambda x:x):
        self.enabled = enabled
        self.specifier = specifier
        self.specifier_formatter = specifier_formatter
        self.key = key
        self.local_reader = None
        self.fw = None

    def update_formatter(self, formatter):
        if self.specifier == None: # mostly self.enabled == False
            return
        reload = False
        if self.specifier_formatter == None:
            self.specifier_formatter = formatter
            retload = True
        else:
            old = copy.deepcopy(self.specifier_formatter)
            self.specifier_formatter.update(formatter)
            reload = self.specifier_formatter.values() != old.values()
        if reload:
            self.load_reader_writer()

    def load_reader_writer(self):
        if self.fw != None:
            self.fw.close()
        if self.specifier_formatter is not None:
            specifier_formatter = copy.deepcopy(self.specifier_formatter)
            specifier_formatter.update({"rand": str(uuid.uuid4().fields[-1])[:5]})
            specifier = self.specifier.format(**specifier_formatter)
        else:
            specifier = self.specifier

        self.fw = scp_io.file_writer_helper(specifier)
        self.local_reader = scp_io.file_reader_helper(specifier)

    def decorate(self):
        def decorator(func):
            def scp_cache_wrapper(*args, **kwargs):

                print("Enabled:", self.enabled, flush=True)
                if not self.enabled:
                    print("Direct return!")
                    return func(*args, **kwargs)

                print("Store!: enabled:", self.enabled)
                if self.fw == None:
                    self.load_reader_writer()

                _key = self.key(*args, **kwargs)
                if self.local_reader.has(_key):
                    return torch.tensor(self.local_reader.get(_key))
                else:
                    result = func(*args, **kwargs)
                    if isinstance(result, torch.Tensor):
                        _result = result.detach().cpu()
                    self.fw[_key] = _result
                    self.local_reader.add(_key, self.fw.file(), _key, self.fw.reader())
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

    scp_cache = SCPCache(key=lambda x:x.key, specifier="scp,ark:cache_{param}.scp,cache_{param}.h5",
                   specifier_formatter={"param":10})
    @scp_cache.decorate()
    def extract_stuff(a):
        return torch.tensor([1,2,3,4])

    scp_cache.update_formatter({"param": 10})
    print(extract_stuff(item("testing")))
    print(extract_stuff(item("testing3")))
    print(extract_stuff(item("testing3"))) # cached
    scp_cache.update_formatter({"param": 20})
    print(extract_stuff(item("testing3"))) # not, new file cached
    print(extract_stuff(item("testing2"))) # not, new file cached

    reader = {key: value for key, value in scp_io.file_reader_helper("ark:cache_10.h5")}
    load = scp_io.file_reader_helper("ark:cache_10.h5")
    print("TEST reader:", reader, "load:", load)
    print(load.get("testing"))
    reader = {key: value for key, value in scp_io.file_reader_helper("scp:cache_10.scp")}
    load = scp_io.file_reader_helper("scp:cache_10.scp")
    print("TEST reader:", reader, "load:", load)
    print(load.get("testing"))
