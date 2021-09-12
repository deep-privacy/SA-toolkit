from .fs import split_data, num_lines, make_soft_link, touch_file, \
        cat, creation_date_file
from . import kaldi

from .kaldi import read_kaldi_conf

def get_uttid_str(tensor_str):
    res = []
    for t in tensor_str:
        uttid = ""
        for c in t[0]:
            _chr = chr(int(c))
            if int(c) == 0:  # "\0" terminator
                break
            uttid += _chr
        res.append(uttid)
    return res

