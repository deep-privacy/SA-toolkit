from .fs import (
    split_data,
    num_lines,
    make_soft_link,
    touch_file,
    cat,
    creation_date_file,
)
from . import kaldi

from .kaldi import read_kaldi_conf, kaldifeat_set_option

from .torch import *


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
