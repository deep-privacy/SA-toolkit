#!/usr/bin/env python3

import json
import logging
import sys

import configargparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

from typing import List, Union, Dict

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

            self.utt2spk = utt2spk
            self.spk = sorted(set([v for v in utt2spk.values()]))

            iSTFTNet_n_fft = 16
            self.hifigan = hifigan.archi.CoreHifiGan(
                iSTFTNetout=True,
                iSTFTNet_n_fft = iSTFTNet_n_fft,
                imput_dim=256+1+len(self.spk),  # BN asr = 256 dim + F0 dim + One hot spk....
                upsample_rates=[5,4,4],
                upsample_kernel_sizes=[11,8,8],
            )
            self.iSTFTNet = hifigan.archi.iSTFTNet(n_fft=iSTFTNet_n_fft, hop_length=4, win_length=iSTFTNet_n_fft)

        def remove_weight_norm(self):
            self.hifigan.remove_weight_norm()

        def train(self, mode=True):
            super().train(mode)
            self.bn_extractor.eval()

        @torch.jit.export
        def convert(self, x, names:List[str]=["default_utts"], filenames:List[str]=["default_utt_filenames"]):
            x = hifigan.dataset.Egs(wavs=x, names=names, filenames=filenames)
            bn = self.get_bn(x)
            f0 = self.get_bn(x)

        def forward(self, egs_with_feat: hifigan.dataset.Egs):
            bn = egs_with_feat["get_bn"]
            f0 = egs_with_feat["get_f0"].unsqueeze(1)
            spk_id = egs_with_feat["get_spk_id"].unsqueeze(2).to(torch.float32)

            f0_inter = F.interpolate(f0, bn.shape[-1])
            x = torch.cat([bn, f0_inter], dim=1)

            spk_id_inter = F.interpolate(spk_id, x.shape[-1])
            x = torch.cat([x, spk_id_inter], dim=1)

            with torch.cuda.amp.autocast(enabled=True):
                spec, phase = self.hifigan(x)
            x = self.iSTFTNet.inverse(spec, phase)
            return x

        @satools.utils.register_feature_extractor(compute_device="cuda", scp_cache=True)
        def get_bn(self, wavinfo: hifigan.dataset.Wavinfo):
            return self.bn_extractor.extract_bn(wavinfo.wav).permute(0, 2, 1)

        @satools.utils.register_feature_extractor(compute_device="cpu", scp_cache=True)
        def get_f0(self, wavinfo: hifigan.dataset.Wavinfo):
            return satools.cmvn.UttCMVN(var_norm=True, keep_zeros=True)(hifigan.yaapt.yaapt(wavinfo.wav, self.f0_yaapt_opts).samp_values)

        @satools.utils.register_feature_extractor(compute_device="cpu", scp_cache=False, sequence_feat=False)
        def get_spk_id(self, wavinfo: hifigan.dataset.Wavinfo):
            if not self.training: # testing
                target = "6081"
            else:
                target = self.utt2spk[wavinfo.name]
            index_spk = self.spk.index(target)
            return F.one_hot(torch.tensor(index_spk), num_classes=len(self.spk))


    return Net

if __name__ == "__main__":
    parser = configargparse.ArgumentParser(description="Model config args")
    args, remaining_argv = parser.parse_known_args()
    sys.argv = sys.argv[:1] + remaining_argv + ["--base-model-args", json.dumps(vars(args))]
    hifigan.HifiGanModel(build(args), cmd_line=True)
