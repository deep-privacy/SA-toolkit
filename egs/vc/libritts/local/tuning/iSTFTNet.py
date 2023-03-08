#!/usr/bin/env python3

import json
import logging
import sys

import configargparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from satools.hifigan import HifiGanModel, archi
from torch.nn.utils import weight_norm, remove_weight_norm

from typing import Tuple, Union, Dict, Optional

import satools
import satools.infer_helper

logging.basicConfig(level=logging.INFO)
logging.getLogger("geocoder").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def build(args):
    class Net(nn.Module):
        def init(self):
            logging.info("Init epoch 0")
            self.bn_extractor = satools.infer_helper.load_model(self.bn_extractor_model)

        def __init__(self):
            super().__init__()
            self.bn_extractor_model = "../../asr/librispeech/exp/chain/bn_tdnnf_t100_aug/final.pt"
            self.bn_extractor = satools.infer_helper.load_model(self.bn_extractor_model, load_weight=False)

            iSTFTNet_n_fft = 16
            self.hifigan = archi.CoreHifiGan(
                iSTFTNetout=True,
                iSTFTNet_n_fft = iSTFTNet_n_fft,
                upsample_rates=[10,8],
                upsample_kernel_sizes=[20,16],
            )
            self.iSTFTNet = archi.iSTFTNet(n_fft=iSTFTNet_n_fft, hop_length=4, win_length=iSTFTNet_n_fft)

            x = torch.rand(1, 257, 50)
            a = self.forward(x)
            print("INIT test", a.shape, a.shape[-1]/x.shape[-1], flush=True)

        def remove_weight_norm(self):
            self.hifigan.remove_weight_norm()

        def forward(self, x):
            print("IN:", x.shape, flush=True)
            spec, phase = self.hifigan(x)
            x = self.iSTFTNet.inverse(spec, phase)
            print("OUT:", x.shape, flush=True)
            return x
    return Net

if __name__ == "__main__":
    parser = configargparse.ArgumentParser(description="Model config args")
    args, remaining_argv = parser.parse_known_args()
    sys.argv = sys.argv[:1] + remaining_argv + ["--base-model-args", json.dumps(vars(args))]
    HifiGanModel(build(args), cmd_line=True)
