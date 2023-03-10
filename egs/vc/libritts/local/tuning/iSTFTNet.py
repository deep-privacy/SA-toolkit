#!/usr/bin/env python3

import json
import logging
import sys

import configargparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

from typing import Tuple, Union, Dict, Optional

from satools import hifigan
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
            self.bn_extractor_type = "bn_tdnnf_t100_aug"
            self.bn_extractor_model = "../../asr/librispeech/exp/chain/" + self.bn_extractor_type + "/final.pt"
            self.bn_extractor = satools.infer_helper.load_model(self.bn_extractor_model, load_weight=False)
            self.bn_extractor.eval()

            self.f0_yaapt_opts = {
                "frame_length": 35.0,
                "frame_space": 20.0,
                "nccf_thresh1": 0.25,
                "tda_frame_length": 25.0,
            }

            iSTFTNet_n_fft = 16
            self.hifigan = hifigan.archi.CoreHifiGan(
                iSTFTNetout=True,
                iSTFTNet_n_fft = iSTFTNet_n_fft,
                imput_dim=255+1,  # BN asr = 256 dim + F0 dim + ....
                upsample_rates=[10,8],
                upsample_kernel_sizes=[20,16],
            )
            self.iSTFTNet = hifigan.archi.iSTFTNet(n_fft=iSTFTNet_n_fft, hop_length=4, win_length=iSTFTNet_n_fft)

        def remove_weight_norm(self):
            self.hifigan.remove_weight_norm()

        def train(self, mode=True):
            super().train(mode)
            self.bn_extractor.eval()

        def forward(self, x):
            x = self.bn_extractor.extract_bn(x).permute(0, 2, 1)
            with torch.cuda.amp.autocast(enabled=True):
                spec, phase = self.hifigan(x)
                x = self.iSTFTNet.inverse(spec, phase)
            return x

        @hifigan.dataset.register_feature_extractor(compute_device="gpu")
        def get_bn(self, egs: hifigan.dataset.WavInfo):
            return self.bn_extractor.extract_bn(egs.wav).permute(0, 2, 1)

        @hifigan.dataset.register_feature_extractor(compute_device="cpu", scp_cache="scp,ark:{dir}F0_cache{rand}.scp,{dir}F0_cache{rand}.h5")
        def get_f0(self, egs: hifigan.dataset.WavInfo):
            print("TEST")
            return hifigan.yaapt.yaapt(egs.wav, self.f0_yaapt_opts).samp_values


    return Net

if __name__ == "__main__":
    parser = configargparse.ArgumentParser(description="Model config args")
    args, remaining_argv = parser.parse_known_args()
    sys.argv = sys.argv[:1] + remaining_argv + ["--base-model-args", json.dumps(vars(args))]
    hifigan.HifiGanModel(build(args), cmd_line=True)
