#!/usr/bin/env python3

import json
import logging
import sys

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm

from typing import Union, Optional

from satools import hifigan
import satools
import satools.infer_helper


"""
No target speaker, using single target during training

Many to One VC (m2o)
"""

def build(args):
    class Net(nn.Module):
        def init(self):
            logging.info("Init epoch 0")
            self.bn_extractor = satools.infer_helper.load_model(self.bn_extractor_model, from_file=__file__)

        def __init__(self, utt2spk):
            super().__init__()
            self.bn_extractor_model = args.asrbn_model
            self.bn_extractor = satools.infer_helper.load_model(self.bn_extractor_model, from_file=__file__, load_weight=False)
            self.bn_extractor.eval()

            self.f0_yaapt_opts = {
                "frame_length": 35.0,
                "frame_space": 20.0,
                "nccf_thresh1": 0.25,
                "tda_frame_length": 25.0,
            }
            self.f0_norm = satools.cmvn.UttCMVN(var_norm=True, keep_zeros=True)

            self.f0: Optional[torch.Tensor] = None # hack to compute the F0 on CPU


            self.hifigan = (hifigan.archi.CoreHifiGan(
                imput_dim=256+1,  # BN asr = 256 dim + F0 dim
                upsample_rates=[5,4,4,2,2],
                upsample_kernel_sizes=[11,8,8,4,4],
            ))

        def remove_weight_norm(self):
            self.hifigan.remove_weight_norm()

        def train(self, mode=True):
            super().train(mode)
            self.bn_extractor.eval()

        @torch.jit.export
        def extract_features(self, x):
            x = satools.utils.WavInfo(wav=x, name="default_utts", filename="default_utt_filenames")
            f0 = self.get_f0(x).unsqueeze(0)
            bn = self.get_bn(x)
            return (f0, bn)

        @torch.jit.export
        def convert(self, x):
            (f0, bn) = self.extract_features(x)
            return self._forward(f0, bn).squeeze(0)

        @torch.jit.unused
        def f0_transformation(self, f0):
            if args.f0_transformation and "quant" in args.f0_transformation:
                f0 = hifigan.nn.quantize_f0(f0, num_bins=args.f0_transformation)
            if args.f0_transformation and "awgn" in args.f0_transformation:
                f0 = hifigan.nn.awgn_f0(f0, target_noise_db=args.f0_transformation)
            if args.f0_transformation and "mean-reverv" in args.f0_transformation:
                f0 = hifigan.nn.mean_reverv_f0(f0, alpha=args.f0_transformation)
            return f0

        def _forward(self, f0, bn):
            f0 = self.f0_norm(f0)
            if f0.dim() == 2:
                f0 = f0.unsqueeze(0)
            f0 = f0.permute(1, 0, 2)
            f0 = self.f0_transformation(f0)
            f0_inter = F.interpolate(f0, bn.shape[-1])
            x = torch.cat([bn, f0_inter], dim=1)

            with torch.amp.autocast('cuda', enabled=True):
                x, _ = self.hifigan(x)
            x = x.to(torch.float32)
            return x


        def forward(self, egs_with_feat: hifigan.dataset.Egs):
            return self._forward(egs_with_feat["get_f0"],
                                 egs_with_feat["get_bn"])

        @satools.utils.register_feature_extractor(compute_device="cuda", scp_cache=True)
        def get_bn(self, wavinfo: Union[satools.utils.WavInfo, torch.Tensor]):
            wav = satools.utils.parse_wavinfo_wav(wavinfo)
            return self.bn_extractor.extract_bn(wav).permute(0, 2, 1)

        def set_f0(self, f0):
            self.f0 = f0

        @satools.utils.register_feature_extractor(compute_device="cpu", scp_cache=True)
        def get_f0(self, wavinfo: Union[satools.utils.WavInfo, torch.Tensor]):
            if self.f0 != None:
                f0, self.f0 = self.f0, None
                return f0
            wav = satools.utils.parse_wavinfo_wav(wavinfo)
            return hifigan.yaapt.yaapt(wav, self.f0_yaapt_opts)


    return Net

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model config args")
    parser.add_argument("--asrbn-model", default="", type=str)
    parser.add_argument("--f0-transformation", default="", type=str)
    args, remaining_argv = parser.parse_known_args()
    sys.argv = sys.argv[:1] + remaining_argv + ["--base-model-args", json.dumps(vars(args))]
    hifigan.HifiGanModel(build(args), cmd_line=True)