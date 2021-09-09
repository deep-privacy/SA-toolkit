from .fs import split_data, num_lines, make_soft_link, touch_file, \
        cat
from . import kaldi


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


def parseval(s):
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    return s
