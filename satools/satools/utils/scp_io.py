from typing import Dict
import numpy as np
import io
import os
import logging
import sys

import kaldiio

"""
Example:
    python3 -c 'import satools.utils; print(satools.utils.scp_io.file_reader_helper("scp:get_f0_dev_set_split2.scp").get("8842_302203_000010_000001"))'
    python3 -c 'import satools.utils; print(dict(satools.utils.scp_io.file_reader_helper("scp:get_f0_dev_set_split2.scp")))'

"""

def seekable(f):
    if hasattr(f, "seekable"):
        return f.seekable()

def File(file, mode='r'):
    specifier = parse_wspecifier(file)
    if mode == 'a':
        class writer():
            def __init__(self):
                self.file = specifier["ark"]
            def __setitem__(self, id, array):
                self.last_write = save_ark(specifier["ark"], {id: array},
                 scp=specifier["scp"], append=True)

        return writer()
    if mode == "r":
        class reader():
            def __init__(self):
                self.file = file

            def keys(self):
                with kaldiio.matio.open_or_fd(specifier["ark"], "rb") as fd:
                    while True:
                        token = kaldiio.matio.read_token(fd)
                        key = fd.tell()
                        if token is None:
                            break
                        max_flag_length = len(b"AUDIO")
                        binary_flag = fd.read(max_flag_length)
                        if seekable(fd):
                            fd.seek(-max_flag_length, 1)
                        if binary_flag[:3] == b"NPY":
                            fd.read(3)
                            length_ = kaldiio.matio._read_length_header(fd)
                            fd.seek(length_, 1)
                        else:
                            kaldiio.matio.read_matrix_or_vector(fd, return_size=True)
                        yield token, key

            def __getitem__(self, id):
                v = kaldiio.load_mat(f"{specifier['ark']}:{id}")
                if isinstance(v, np.lib.npyio.NpzFile):
                    v = v['arr_0']
                return v

        return reader()

def file_writer_helper(
    wspecifier: str,
    overwrite=False,
):
    """Write matrices in kaldi style

    Args:
        wspecifier: e.g. ark,scp:out.ark,out.scp

    Write in kaldi-matrix-ark with "kaldi-scp" file:

    >>> with file_writer_helper('ark,scp:out.ark,out.scp') as f:
    >>>     f['uttid'] = array

    This "scp" has the following format:

        uttidA out.ark:1234
        uttidB out.ark:2222

    where, 1234 and 2222 points the strating byte address of the matrix.
    (For detail, see official documentation of Kaldi)
    """

    if wspecifier != None and wspecifier.startswith("scp:"):
        wspecifier = wspecifier.replace("scp:", "scp,ark:") + "," + wspecifier.split(":")[1].replace("scp", "ark")
    return Writer(
        wspecifier,
        overwrite=overwrite
    )


def parse_wspecifier(wspecifier: str) -> Dict[str, str]:
    """Parse wspecifier to dict

    Examples:
        >>> parse_wspecifier('ark,scp:out.ark,out.scp')
        {'ark': 'out.ark', 'scp': 'out.scp'}

    """
    ark_scp, filepath = wspecifier.split(":", 1)
    if ark_scp not in ["ark", "scp,ark", "ark,scp"]:
        raise ValueError("{} is not allowed: {}".format(ark_scp, wspecifier))
    ark_scps = ark_scp.split(",")
    filepaths = filepath.split(",")
    if len(ark_scps) != len(filepaths):
        raise ValueError("Mismatch: {} and {}".format(ark_scp, filepath))
    spec_dict = dict(zip(ark_scps, filepaths))
    return spec_dict

class Writer():
    """Writer

    Examples:
        >>> with Writer('ark:out.ark') as f:
        ...     f['key'] = array
    """

    def __init__(self, wspecifier, overwrite=False):
        self.writer = File(wspecifier, mode="a")
        self.specifier = wspecifier
        spec_dict = parse_wspecifier(wspecifier)
        if overwrite:
            with open(spec_dict["ark"], 'w') as f:
                f.truncate(0)
            with open(spec_dict["scp"], 'w') as f:
                f.truncate(0)
        self.filename = spec_dict["ark"]

    def id(self):
        return self.writer.last_write

    def file(self):
        return self.filename

    def reader(self):
        return File(self.specifier, mode="r")

    def __setitem__(self, key, value):
        self.writer[key] = value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def file_reader_helper(
    rspecifier: str,
    return_shape: bool = False,
):
    """Read uttid and array in kaldi style

    This function might be a bit confusing as "ark" is used
    for ark to imitate "kaldi-rspecifier".

    Args:
        rspecifier: Give as "ark:feats.ark" or "scp:feats.scp"
        return_shape: Return the shape of the matrix,
            instead of the matrix. This can reduce IO cost for ark.
    Returns:
        Generator[Tuple[str, np.ndarray], None, None]:

    Examples:
        Read from kaldi-matrix ark file:

        >>> for u, array in file_reader_helper('ark:feats.ark'):
        ...     array

    """
    return Reader(rspecifier, return_shape=return_shape)


class Reader:
    def __init__(self, rspecifier, return_shape=False):
        if ":" not in rspecifier:
            raise ValueError(
                'Give "rspecifier" such as "ark:some.ark: {}"'.format(rspecifier)
            )
        self.rspecifier = rspecifier
        self.ark_or_scp, self.filepath = self.rspecifier.split(":", 1)
        if self.ark_or_scp == "scp,ark": # load scp
            self.ark_or_scp, self.filepath = self.rspecifier.split(":", 1)
            self.ark_or_scp = self.ark_or_scp.split(",")[0]
            self.filepath = self.filepath.split(",")[0]
        if self.ark_or_scp == "ark,scp": # load scp
            self.ark_or_scp, self.filepath = self.rspecifier.split(":", 1)
            self.ark_or_scp = self.ark_or_scp.split(",")[1]
            self.filepath = self.filepath.split(",")[1]
        if self.ark_or_scp not in ["ark", "scp"]:
            raise ValueError(f"Must be scp or ark: {self.ark_or_scp}")

        self.key_to_item = None
        self.container_dict = None
        self.return_shape = return_shape

    def parse_line(self, line):
        key, value = line.rstrip().split(None, 1)

        if ":" not in value:
            raise RuntimeError(
                "scp file for ark should be like: "
                '"uttid filepath.container:key": {}({})'.format(
                    line, self.filepath
                )
            )
        split_list = value.split(":")
        result = [":".join(split_list[:-1]), split_list[-1]]
        path, z_key = result[0], result[1]
        return key, path, z_key

    def has(self, key):
        self.init()
        if self.key_to_item == None:
            return False
        return key in self.key_to_item

    def get(self, key):
        self.init()
        container_dict = self.container_dict
        try:
            path, z_key = self.key_to_item[key]
            data = container_dict[path][z_key]
        except Exception:
            logging.error(
                "Error when loading key={}".format(key)
            )
            raise
        if self.return_shape:
            return data.shape
        else:
            return data[()]

    def add(self, key, path, z_key, file):
        self.init()
        """
        Do not write to file, just udpate entry
        """
        self.key_to_item[key] = (path, z_key)
        self.container_dict[path] = file

    def merge(self, other):
        self.init()
        other.init()
        if other.key_to_item != None:
            self.key_to_item.update(other.key_to_item)
            self.container_dict.update(other.container_dict)


    def __repr__(self):
        return str(self.key_to_item)


    def init(self):
        if self.key_to_item != None:
            return self
        if self.ark_or_scp == "scp":
            key_to_item = {}
            container_dict = {}
            if not os.path.isfile(self.filepath):
                return self
            with open(self.filepath, "r", encoding="utf-8") as f:
                for line in f:
                    key, path, z_key = self.parse_line(line)

                    # try to guess the path to the z file from current path
                    if not os.path.isfile(path):
                        dir_path = os.path.dirname(path)
                        path_to_check = os.path.realpath(dir_path)
                        full_path = os.path.realpath(os.getcwd())
                        common_path = os.path.commonpath([path_to_check, full_path])
                        path = os.path.join(common_path, os.path.basename(path))

                    key_to_item[key] = (path, z_key)

                    container_file = container_dict.get(path)
                    if container_file is None:
                        try:
                            container_file = File(f"ark:{path}", mode="r")
                        except Exception:
                            logging.error("Error when loading {}".format(path))
                            raise
                        container_dict[path] = container_file
            self.key_to_item = key_to_item
            self.container_dict = container_dict
            return self
        else:
            filepath = self.filepath
            file = File(f"ark:{filepath}", mode="r")
            self.key_to_item = {v[0]:(filepath, v[1]) for v in file.keys()}
            self.container_dict = {filepath:file}
            return self

    def __iter__(self):
        self.init()
        container_dict = self.container_dict
        for key, (path, z_key) in self.key_to_item.items():
            try:
                data = container_dict[path][z_key]
            except Exception:
                logging.error(
                    "Error when loading {} with key={}".format(path, z_key)
                )
                raise

            if self.return_shape:
                yield key, data.shape
            else:
                yield key, data[()]






# Pattch of :https://github.com/nttcslab-sp/kaldiio/blob/60c3c928e4fb499d8adbf08fa7297c55551277d7/kaldiio/matio.py#L328
def save_ark(
    ark,
    array_dict,
    scp=None,
    append=False,
    text=False,
    endian="<",
    compression_method=None,
    write_function="numpy",
):
    """Write ark
    Args:
        ark (str or fd):
        array_dict (dict):
        scp (str or fd):
        append (bool): If True is specified, open the file
            with appendable mode
        text (bool): If True, saving in text ark format.
        endian (str):
        compression_method (int):
        write_function: (str):
    """
    if isinstance(ark, kaldiio.matio.string_types):
        seekable = True
    # Maybe, never match with this
    elif not hasattr(ark, "tell"):
        seekable = False
    else:
        try:
            ark.tell()
            seekable = True
        except Exception:
            seekable = False

    if scp is not None and not isinstance(ark, kaldiio.matio.string_types):
        if not seekable:
            raise TypeError(
                "scp file can be created only "
                "if the output ark file is a file or "
                "a seekable file descriptor."
            )

    # Write ark
    mode = "ab" if append else "wb"
    pos_list = []
    with kaldiio.matio.open_or_fd(ark, mode) as fd:
        if seekable:
            offset = fd.tell()
        else:
            offset = 0
        size = 0
        for key in array_dict:
            encode_key = (key + " ").encode(encoding=kaldiio.matio.default_encoding)
            fd.write(encode_key)
            size += len(encode_key)
            pos_list.append(size)
            data = array_dict[key]

            if write_function == "numpy":
                def _write_function(fd, data):
                    # Write numpy file in BytesIO
                    _fd = io.BytesIO()
                    np.savez_compressed(_fd, data)

                    fd.write(b"NPY")
                    buf = _fd.getvalue()

                    # Write the information for the length
                    bytes_length = kaldiio.matio._write_length_header(fd, len(buf))

                    # Write numpy to real file object
                    fd.write(buf)

                    return len(buf) + len(b"NPY") + bytes_length
                size += _write_function(fd, data)

            elif isinstance(data, (list, tuple)):
                rate, array = data
                size += kaldiio.matio.write_wav(fd, rate, array)
            elif text:
                size += kaldiio.matio.write_array_ascii(fd, data, endian)
            else:
                size += kaldiio.matio.write_array(fd, data, endian, compression_method)

    # Write scp
    mode = "a" if append else "w"
    if scp is not None:
        name = ark if isinstance(ark, kaldiio.matio.string_types) else ark.name
        with kaldiio.matio.open_or_fd(scp, mode) as fd:
            for key, position in zip(array_dict, pos_list):
                fd.write(key + " " + name + ":" + str(position + offset) + "\n")
        return position + offset
