#!/usr/bin/env python3

import json
import logging
import sys

import configargparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

from typing import List, Union, Dict, Callable, Any

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

        def __init__(self, utt2spk):
            super().__init__()
            self.bn_extractor_model = args.asrbn_model
            self.bn_extractor = satools.infer_helper.load_model(self.bn_extractor_model, load_weight=False)
            self.bn_extractor.eval()

            self.f0_yaapt_opts = {
                "frame_length": 35.0,
                "frame_space": 20.0,
                "nccf_thresh1": 0.25,
                "tda_frame_length": 25.0,
            }
            self.f0_norm = satools.cmvn.UttCMVN(var_norm=True, keep_zeros=True)

            self.utt2spk = utt2spk
            self.spk = sorted(set([v for v in utt2spk.values()]))
            self.target = ""


            self.hifigan = (hifigan.archi.CoreHifiGan(
                imput_dim=256+1+len(self.spk),  # BN asr = 256 dim + F0 dim + One hot spk....
                upsample_rates=[5,4,4,2,2],
                upsample_kernel_sizes=[11,8,8,4,4],
            ))

        def remove_weight_norm(self):
            self.hifigan.remove_weight_norm()

        def train(self, mode=True):
            super().train(mode)
            self.bn_extractor.eval()

        @torch.jit.export
        def extract_features(self, x, target:str="6081"):
            x = hifigan.dataset.Wavinfo(wav=x, name="default_utts", filename="default_utt_filenames")
            f0 = self.get_f0(x).unsqueeze(0)
            bn = self.get_bn(x)
            self.target = target
            spk_id = self.get_spk_id(x).unsqueeze(0)
            return (f0, bn, spk_id)

        @torch.jit.export
        def convert(self, x, target:str="6081"):
            (f0, bn, spk_id) = self.extract_features(x, target)
            return self._forward(f0, bn, spk_id).squeeze(0)

        def _forward(self, f0, bn, spk_id):
            f0 = self.f0_norm(f0)
            f0 = f0.unsqueeze(1)
            spk_id = spk_id.unsqueeze(2).to(torch.float32)
            f0_inter = F.interpolate(f0, bn.shape[-1])
            x = torch.cat([bn, f0_inter], dim=1)

            spk_id_inter = F.interpolate(spk_id, x.shape[-1])
            x = torch.cat([x, spk_id_inter], dim=1)

            with torch.cuda.amp.autocast(enabled=True):
                x, _ = self.hifigan(x)
            x = x.to(torch.float32)
            return x

        def forward(self, egs_with_feat: hifigan.dataset.Egs):
            return self._forward(egs_with_feat["get_f0"],
                                 egs_with_feat["get_bn"],
                                 egs_with_feat["get_spk_id"])

        @satools.utils.register_feature_extractor(compute_device="cuda", scp_cache=True)
        def get_bn(self, wavinfo: hifigan.dataset.Wavinfo):
            return self.bn_extractor.extract_bn(wavinfo.wav.detach().clone()).permute(0, 2, 1)

        @satools.utils.register_feature_extractor(compute_device="cpu", scp_cache=True)
        def get_f0(self, wavinfo: hifigan.dataset.Wavinfo):
            return hifigan.yaapt.yaapt(wavinfo.wav.detach().clone(), self.f0_yaapt_opts).samp_values

        @satools.utils.register_feature_extractor(compute_device="cpu", scp_cache=False, sequence_feat=False)
        def get_spk_id(self, wavinfo: hifigan.dataset.Wavinfo):
            if self.target == "":
                if not self.training: # testing
                    target = "6081"
                else:
                    target = self.utt2spk[wavinfo.name]
            else:
                target = self.target
            index_spk = self.spk.index(target)
            return F.one_hot(torch.tensor(index_spk), num_classes=len(self.spk))


    return Net

if __name__ == "__main__":
    parser = configargparse.ArgumentParser(description="Model config args")
    parser.add("--asrbn-model", default="", type=str)
    args, remaining_argv = parser.parse_known_args()
    sys.argv = sys.argv[:1] + remaining_argv + ["--base-model-args", json.dumps(vars(args))]
    hifigan.HifiGanModel(build(args), cmd_line=True)
