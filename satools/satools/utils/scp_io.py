# Taken from: https://github.com/espnet/espnet/blob/master/espnet/utils/cli_writers.py

from typing import Dict
import io
import logging
import sys

import h5py
import kaldiio

def file_writer_helper(
    wspecifier: str,
    filetype: str = "mat",
    write_num_frames: str = None,
    compress: bool = False,
    compression_method: int = 2
):
    """Write matrices in kaldi style

    Args:
        wspecifier: e.g. ark,scp:out.ark,out.scp
        filetype: "mat" is kaldi-martix, "hdf5": HDF5
        write_num_frames: e.g. 'ark,t:num_frames.txt'
        compress: Compress or not
        compression_method: Specify compression level

    Write in kaldi-matrix-ark with "kaldi-scp" file:

    >>> with file_writer_helper('ark,scp:out.ark,out.scp') as f:
    >>>     f['uttid'] = array

    This "scp" has the following format:

        uttidA out.ark:1234
        uttidB out.ark:2222

    where, 1234 and 2222 points the strating byte address of the matrix.
    (For detail, see official documentation of Kaldi)

    Write in HDF5 with "scp" file:

    >>> with file_writer_helper('ark,scp:out.h5,out.scp', 'hdf5') as f:
    >>>     f['uttid'] = array

    This "scp" file is created as:

        uttidA out.h5:uttidA
        uttidB out.h5:uttidB

    HDF5 can be, unlike "kaldi-ark", accessed to any keys,
    so originally "scp" is not required for random-reading.
    Nevertheless we create "scp" for HDF5 because it is useful
    for some use-case. e.g. Concatenation, Splitting.

    """
    if filetype == "mat":
        return KaldiWriter(
            wspecifier,
            write_num_frames=write_num_frames,
            compress=compress,
            compression_method=compression_method,
        )
    elif filetype == "hdf5":
        return HDF5Writer(
            wspecifier, write_num_frames=write_num_frames, compress=compress
        )
    else:
        raise NotImplementedError(f"filetype={filetype}")


class BaseWriter:
    def __setitem__(self, key, value):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        try:
            self.writer.close()
        except Exception:
            pass

        if self.writer_scp is not None:
            try:
                self.writer_scp.close()
            except Exception:
                pass

        if self.writer_nframe is not None:
            try:
                self.writer_nframe.close()
            except Exception:
                pass


def get_num_frames_writer(write_num_frames: str):
    """get_num_frames_writer

    Examples:
        >>> get_num_frames_writer('ark,t:num_frames.txt')
    """
    if write_num_frames is not None:
        if ":" not in write_num_frames:
            raise ValueError(
                'Must include ":", write_num_frames={}'.format(write_num_frames)
            )

        nframes_type, nframes_file = write_num_frames.split(":", 1)
        if nframes_type != "ark,t":
            raise ValueError(
                "Only supporting text mode. "
                "e.g. --write-num-frames=ark,t:foo.txt :"
                "{}".format(nframes_type)
            )

    return open(nframes_file, "w", encoding="utf-8")


class KaldiWriter(BaseWriter):
    def __init__(
        self, wspecifier, write_num_frames=None, compress=False, compression_method=2
    ):
        if compress:
            self.writer = kaldiio.WriteHelper(
                wspecifier, compression_method=compression_method
            )
        else:
            self.writer = kaldiio.WriteHelper(wspecifier)
        self.writer_scp = None
        if write_num_frames is not None:
            self.writer_nframe = get_num_frames_writer(write_num_frames)
        else:
            self.writer_nframe = None

    def __setitem__(self, key, value):
        self.writer[key] = value
        if self.writer_nframe is not None:
            self.writer_nframe.write(f"{key} {len(value)}\n")


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


class HDF5Writer(BaseWriter):
    """HDF5Writer

    Examples:
        >>> with HDF5Writer('ark:out.h5', compress=True) as f:
        ...     f['key'] = array
    """

    def __init__(self, wspecifier, write_num_frames=None, compress=False):
        spec_dict = parse_wspecifier(wspecifier)
        self.filename = spec_dict["ark"]

        if compress:
            self.kwargs = {"compression": "gzip"}
        else:
            self.kwargs = {}
        self.writer = h5py.File(spec_dict["ark"], "w")
        if "scp" in spec_dict:
            self.writer_scp = open(spec_dict["scp"], "w", encoding="utf-8")
        else:
            self.writer_scp = None
        if write_num_frames is not None:
            self.writer_nframe = get_num_frames_writer(write_num_frames)
        else:
            self.writer_nframe = None

    def __setitem__(self, key, value):
        self.writer.create_dataset(key, data=value, **self.kwargs)

        if self.writer_scp is not None:
            self.writer_scp.write(f"{key} {self.filename}:{key}\n")
        if self.writer_nframe is not None:
            self.writer_nframe.write(f"{key} {len(value)}\n")



def file_reader_helper(
    rspecifier: str,
    filetype: str = "mat",
    return_shape: bool = False,
    segments: str = None,
):
    """Read uttid and array in kaldi style

    This function might be a bit confusing as "ark" is used
    for HDF5 to imitate "kaldi-rspecifier".

    Args:
        rspecifier: Give as "ark:feats.ark" or "scp:feats.scp"
        filetype: "mat" is kaldi-martix, "hdf5": HDF5
        return_shape: Return the shape of the matrix,
            instead of the matrix. This can reduce IO cost for HDF5.
    Returns:
        Generator[Tuple[str, np.ndarray], None, None]:

    Examples:
        Read from kaldi-matrix ark file:

        >>> for u, array in file_reader_helper('ark:feats.ark', 'mat'):
        ...     array

        Read from HDF5 file:

        >>> for u, array in file_reader_helper('ark:feats.h5', 'hdf5'):
        ...     array

    """
    if filetype == "mat":
        return KaldiReader(rspecifier, return_shape=return_shape, segments=segments)
    elif filetype == "hdf5":
        return HDF5Reader(rspecifier, return_shape=return_shape)
    else:
        raise NotImplementedError(f"filetype={filetype}")


class KaldiReader:
    def __init__(self, rspecifier, return_shape=False, segments=None):
        self.rspecifier = rspecifier
        self.return_shape = return_shape
        self.segments = segments

    def __iter__(self):
        with kaldiio.ReadHelper(self.rspecifier, segments=self.segments) as reader:
            for key, array in reader:
                if self.return_shape:
                    array = array.shape
                yield key, array


class HDF5Reader:
    def __init__(self, rspecifier, return_shape=False):
        if ":" not in rspecifier:
            raise ValueError(
                'Give "rspecifier" such as "ark:some.ark: {}"'.format(self.rspecifier)
            )
        self.rspecifier = rspecifier
        self.ark_or_scp, self.filepath = self.rspecifier.split(":", 1)
        if self.ark_or_scp not in ["ark", "scp"]:
            raise ValueError(f"Must be scp or ark: {self.ark_or_scp}")

        self.return_shape = return_shape

    def __iter__(self):
        if self.ark_or_scp == "scp":
            hdf5_dict = {}
            with open(self.filepath, "r", encoding="utf-8") as f:
                for line in f:
                    key, value = line.rstrip().split(None, 1)

                    if ":" not in value:
                        raise RuntimeError(
                            "scp file for hdf5 should be like: "
                            '"uttid filepath.h5:key": {}({})'.format(
                                line, self.filepath
                            )
                        )
                    path, h5_key = value.split(":", 1)

                    hdf5_file = hdf5_dict.get(path)
                    if hdf5_file is None:
                        try:
                            hdf5_file = h5py.File(path, "r")
                        except Exception:
                            logging.error("Error when loading {}".format(path))
                            raise
                        hdf5_dict[path] = hdf5_file

                    try:
                        data = hdf5_file[h5_key]
                    except Exception:
                        logging.error(
                            "Error when loading {} with key={}".format(path, h5_key)
                        )
                        raise

                    if self.return_shape:
                        yield key, data.shape
                    else:
                        yield key, data[()]

            # Closing all files
            for k in hdf5_dict:
                try:
                    hdf5_dict[k].close()
                except Exception:
                    pass

        else:
            if self.filepath == "-":
                # Required h5py>=2.9
                filepath = io.BytesIO(sys.stdin.buffer.read())
            else:
                filepath = self.filepath
            with h5py.File(filepath, "r") as f:
                for key in f:
                    if self.return_shape:
                        yield key, f[key].shape
                    else:
                        yield key, f[key][()]

