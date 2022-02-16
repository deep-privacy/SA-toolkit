import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import weight_norm, remove_weight_norm

import random
import pkwrap
from pkwrap.hifigan import HifiGanModel

# IMPORT F0 Quant
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__)))
from f0_quant import build as f0_quant_builder

# IMPORT BN
import importlib.util

import configargparse
from types import SimpleNamespace

import logging
import time

logging.basicConfig(level=logging.INFO)
logging.getLogger("geocoder").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def build(args):
    class CoreHifiGan(torch.nn.Module):
        def __init__(
            self,
            upsample_rates=[5, 4, 4, 3, 2],
            upsample_kernel_sizes=[11, 8, 8, 6, 4],
            imput_dim=384,  # BN asr = 256 dim + F0 quant dim = 128
            upsample_initial_channel=512,
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        ):
            super().__init__()

            self.num_kernels = len(resblock_kernel_sizes)
            self.num_upsamples = len(upsample_rates)
            self.conv_pre = weight_norm(
                nn.Conv1d(imput_dim, upsample_initial_channel, 7, 1, padding=3)
            )

            resblock = pkwrap.nn.ResBlock1
            #  resblock = pkwrap.nn.ResBlock2

            self.ups = nn.ModuleList()
            for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
                self.ups.append(
                    weight_norm(
                        nn.ConvTranspose1d(
                            upsample_initial_channel // (2 ** i),
                            upsample_initial_channel // (2 ** (i + 1)),
                            k,
                            u,
                            padding=(k - u) // 2,
                        )
                    )
                )

            self.resblocks = nn.ModuleList()
            for i in range(len(self.ups)):
                ch = upsample_initial_channel // (2 ** (i + 1))
                for j, (k, d) in enumerate(
                    zip(resblock_kernel_sizes, resblock_dilation_sizes)
                ):
                    self.resblocks.append(resblock(ch, k, d))

            self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
            self.ups.apply(pkwrap.nn.init_weights)
            self.conv_post.apply(pkwrap.nn.init_weights)

        def forward(self, x):
            x = self.conv_pre(x)
            for i in range(self.num_upsamples):
                x = F.leaky_relu(x, pkwrap.nn.LRELU_SLOPE)
                x = self.ups[i](x)
                xs = None
                for j in range(self.num_kernels):
                    if xs is None:
                        xs = self.resblocks[i * self.num_kernels + j](x)
                    else:
                        xs += self.resblocks[i * self.num_kernels + j](x)
                x = xs / self.num_kernels
            x = F.leaky_relu(x)
            x = self.conv_post(x)
            x = torch.tanh(x)

            return x

        def remove_weight_norm(self):
            for l in self.ups:
                remove_weight_norm(l)
            for l in self.resblocks:
                l.remove_weight_norm()
            remove_weight_norm(self.conv_pre)
            remove_weight_norm(self.conv_post)

    class Net(nn.Module):
        def after_load_hook(self):
            # called by trainer to overwrite some modules weights
            self.f0_quant_state = torch.load(args.f0_quant_state, map_location="cpu")[
                "generator"
            ]
            self.f0_quant.load_state_dict(self.f0_quant_state)

            pkwrap_path = pkwrap.__path__[0] + "/../egs/librispeech/v1/"
            model_weight = "final.pt"

            # loading from args
            model = args.asrbn_tdnnf_model  # eg: "local/chain/e2e/tuning/tdnnf.py"
            exp_path = args.asrbn_tdnnf_exp_path  # eg: "exp/chain/e2e_tdnnf/"

            num_pdfs_train = pkwrap.script_utils.read_single_param_file(
                pkwrap_path + exp_path + "/num_pdfs"
            )
            assert num_pdfs_train == self.bn_asr_output_dim

            self.bn_model_state = torch.load(
                pkwrap_path + exp_path + model_weight, map_location="cpu"
            )
            self.bn_asr.load_state_dict(self.bn_model_state)

        def __init__(self, load_f0_asr_weight=True, asr_bn_model=None):
            super().__init__()

            self.validating = False
            self.sample_size = None
            # Hifigan Model
            self.core_hifigan = CoreHifiGan(
                upsample_rates=list(map(int, args.hifigan_upsample_rates.split(",")))
            )

            # No spk features (any/many to one speaker conversion)
            #  self.spkr = nn.Embedding(num_spkr, embedding_dim)

            # F0 quant model
            self.f0_quant = f0_quant_builder(SimpleNamespace())()
            self.f0_quant.eval()

            if asr_bn_model != None:
                self.bn_asr = asr_bn_model
            else:
                # IMPORT LIB
                # ASR-BN features
                bnargs = SimpleNamespace()
                if args.asrbn_tdnnf_vq != -1:
                    bnargs = SimpleNamespace(
                        freeze_encoder=True,
                        codebook_size=args.asrbn_tdnnf_vq,  # eg: 16
                    )
                if args.asrbn_tdnnf_dp != -1:
                    bnargs = SimpleNamespace(
                        freeze_encoder=True,
                        epsilon=str(args.asrbn_tdnnf_dp),  # eg: 180000
                    )

                pkwrap_path = pkwrap.__path__[0] + "/../egs/librispeech/v1/"
                model = args.asrbn_tdnnf_model  # eg: "local/chain/e2e/tuning/tdnnf.py"

                config_path = pkwrap_path + model
                if not os.path.exists(config_path):
                    raise FileNotFoundError(
                        "No file found at location {}".format(config_path)
                    )
                spec = importlib.util.spec_from_file_location("config", config_path)
                asr_model_file = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(asr_model_file)

                self.bn_asr_output_dim = 3280
                self.bn_asr = asr_model_file.build(bnargs)(
                    output_dim=self.bn_asr_output_dim
                )
                self.bn_asr.eval()

            if load_f0_asr_weight:
                self.after_load_hook()

        # keep some Torch submodule in eval mode
        def train(self, mode=True):
            super().train(mode)
            self.f0_quant.eval()
            self.bn_asr.eval()
            return self

        @torch.no_grad()
        def validate_model(self, device="cpu"):
            self.validating = True
            N = 2
            C = 86640 // 80
            f0 = torch.arange(N * C).reshape(N, 1, C).float().to(device)

            N = 2
            C = 125760
            x = torch.arange(N * C).reshape(N, C).float().to(device)

            nnet_output = self.forward(f0=f0, audio=x)
            assert (
                nnet_output.shape[2] >= x.shape[-1] - 10000
                and nnet_output.shape[2] <= x.shape[-1] + 10000
            ), f"Mismatch too high in vocoder output shape - {nnet_output.shape} != {x.shape}"

            self.eval()
            nnet_output = self.forward(f0=f0, audio=x)
            assert (
                nnet_output.shape[2] >= x.shape[-1] - 10000
                and nnet_output.shape[2] <= x.shape[-1] + 10000
            ), f"Mismatch too high in vocoder output shape - {nnet_output.shape} != {x.shape}"
            self.train()
            self.validating = False

        def get_feat_f0_len(self, kwargs, b_id):
            if "full_audio_to_cache" not in kwargs:
                raise ValueError("'full_audio_to_cache' not in 'kwargs'")

            feats, f0s = kwargs["audio"][b_id], kwargs["f0"][b_id]
            ori_feats, ori_lengths, ori_f0s, ori_ys = kwargs["full_audio_to_cache"]

            ori_feats, ori_lengths, ori_f0s, ori_ys = (
                ori_feats[b_id],
                ori_lengths[b_id],
                ori_f0s[b_id],
                ori_ys[b_id],
            )

            f0_feat_subsampling = ori_feats.shape[-1] // ori_f0s.shape[-1]

            feat_padding_end_len = ori_feats.shape[-1] - ori_lengths.item()
            f0_padding_end_len = (
                ori_feats.shape[-1] - ori_lengths.item()
            ) // f0_feat_subsampling

            feat_len = ori_feats.shape[-1] - feat_padding_end_len
            f0_len = ori_f0s.shape[-1] - f0_padding_end_len

            return feat_len, f0_len

        def querry_cache(self, kwargs):
            # Always re-compute F0 feats
            f0_h_q = self.extract_f0_only(kwargs["f0"])

            bn_asr_h = []
            asrbn_cache = os.path.join(args.checkpoint_path, "cache_asrbn_train")
            for b_id in range(len(kwargs["filenames"])):
                feat_len, f0_len = self.get_feat_f0_len(kwargs, b_id)
                key = kwargs["filenames"][b_id]

                bn_asr_h_ori = pkwrap.utils.fs.get_from_cache(key, asrbn_cache)
                if bn_asr_h_ori == None:
                    ori_feats, _, ori_f0s, _ = kwargs["full_audio_to_cache"]
                    bn_asr_h_ori = self.extract_bn_only(
                        ori_feats[b_id].unsqueeze(0)[..., :feat_len]
                    )
                    bn_asr_h_ori = bn_asr_h_ori.squeeze(0)
                    pkwrap.utils.fs.put_in_cache(bn_asr_h_ori, key, asrbn_cache)

                feat_subsampling = feat_len / bn_asr_h_ori.shape[-1]
                _feats_idx, _f0s_idx = kwargs["sampling_iter_idx"][b_id]
                feats_start, feats_end = _feats_idx

                # fmt: off
                if self.sample_size == None:
                    logging.info("ASR subsampling:" + str(feat_subsampling))
                    self.sample_size = int(int(feats_end - feats_start) / feat_subsampling)
                start = int(feats_start / feat_subsampling)
                end = start + self.sample_size
                #  print(feat_len / bn_asr_h_ori.shape[-1], start, start + sample_size, sample_size)
                bn_asr_h += [bn_asr_h_ori[ ..., start : end, ].to(kwargs["audio"].device).permute(1,0)]

            bn_asr_h = pad_sequence(bn_asr_h, batch_first=True).permute(0,2,1)
            # fmt: on
            f0_h_q = F.interpolate(f0_h_q, bn_asr_h.shape[-1])
            return f0_h_q, bn_asr_h

        @torch.no_grad()
        def extract_bn_only(self, audio):
            post, asr_out_xent = self.bn_asr(audio)
            bn_asr_h = self.bn_asr.bottleneck_out.permute(0, 2, 1)
            if self.validating:
                logging.info("ASR subsampling:" + str(audio.shape[1] / bn_asr_h.shape[-1]))
            if args.asrbn_interpol_bitrate != -1:
                target_fr = audio.shape[1] / args.asrbn_interpol_bitrate  # Original VQ exp sub_sampling
                bn_asr_h = torch.nn.functional.interpolate(bn_asr_h, int(target_fr))
                if self.validating:
                    logging.info("ASR subsampling after interpol:" + str(audio.shape[1] / bn_asr_h.shape[-1]))

            return bn_asr_h

        @torch.no_grad()
        def extract_f0_only(self, f0):
            # Extract F0 quant features
            f0_h = self.f0_quant.encoder(f0)
            f0_h = [x.detach() for x in f0_h]
            id_f0_q, f0_h_q, _, _ = self.f0_quant.vq(f0_h)
            f0_h_q = torch.stack(f0_h_q, dim=0).squeeze()

            if f0_h_q.ndim == 2:
                f0_h_q = f0_h_q.unsqueeze(0)
            return f0_h_q

        @torch.no_grad()
        def extract_features(self, f0, audio):
            """
            Takes F0 features and the raw audio features
            returns F0 representation and audio representation (from ASR/SSL models)
            """
            f0_h_q = self.extract_f0_only(f0)
            bn_asr_h = self.extract_bn_only(audio)

            #  bn_asr_h = F.interpolate(bn_asr_h, f0_h_q.shape[-1])
            f0_h_q = F.interpolate(f0_h_q, bn_asr_h.shape[-1])

            return f0_h_q, bn_asr_h

        def forward(self, **kwargs):

            if (
                "full_audio_to_cache" in kwargs
                and not args.no_caching
                and self.training
            ):
                f0_h_q, bn_asr_h = self.querry_cache(kwargs)
            else:
                f0_h_q, bn_asr_h = self.extract_features(kwargs["f0"], kwargs["audio"])

            x = torch.cat([bn_asr_h, f0_h_q], dim=1)

            #  spkr = self.spkr(id_spk).transpose(1, 2)
            #  spkr = F.interpolate(spkr, x.shape[-1])
            #  x = torch.cat([x, spkr], dim=1)

            out = self.core_hifigan(x)
            return out

    return Net


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(description="Model config args")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--checkpoint_path", default="exp/hifigan", type=str)
    parser.add_argument("--init_weight_model", default="last", type=str)
    parser.add_argument("--f0_quant_state", default="exp/f0_vq/g_best", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--training_epochs", default=1500, type=int)
    parser.add_argument("--cold_restart", default=False, action="store_true")
    parser.add_argument("--no-caching", default=False, action="store_true")
    parser.add_argument(
        "--asrbn_tdnnf_model", default="local/chain/e2e/tuning/tdnnf.py", type=str
    )
    parser.add_argument(
        "--asrbn_tdnnf_exp_path", default="exp/chain/e2e_tdnnf/", type=str
    )
    parser.add_argument(
        "--asrbn_interpol_bitrate", default=-1, type=int
    )
    parser.add_argument("--asrbn_tdnnf_vq", default=-1, type=int)
    parser.add_argument("--asrbn_tdnnf_dp", default=-1, type=int)
    parser.add_argument("--hifigan_upsample_rates", default="5, 4, 4, 3, 2", type=str)
    args, remaining_argv = parser.parse_known_args()
    sys.argv = sys.argv[:1] + remaining_argv

    torch.cuda.manual_seed(52)
    random.seed(52)

    wav_list = pkwrap.utils.fs.scans_directory_for_ext(
        "./data/LJSpeech-1.1/wavs_16khz", "wav"
    )
    wav_list.sort()
    random.shuffle(wav_list)
    split = int(0.92 * len(wav_list))
    train_list = ",".join(wav_list[:split])
    dev_list = ",".join(wav_list[split:])

    HifiGanModel(
        build(args),
        **{
            "mode": "train",
            "training_epochs": args.training_epochs,
            "cold_restart": args.cold_restart,
            "num_workers": 4,
            "rank": args.local_rank,
            "checkpoint_path": args.checkpoint_path,
            "init_weight_model": args.init_weight_model,
            "segment_size": 16640,
            "minibatch_size": args.batch_size,
            "train_utterances": train_list,
            "test_utterances": dev_list,
        },
    )
