# IMPORT BN
import importlib.util
import json
import logging
import os
import random
import sys
from types import SimpleNamespace

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from satools.hifigan import HifiGanModel
from torch.nn.utils import weight_norm, remove_weight_norm

import satools

logging.basicConfig(level=logging.INFO)
logging.getLogger("geocoder").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

def build(args, spkids):
    class CoreHifiGan(torch.nn.Module):
        def __init__(
            self,
            upsample_rates=[5,4,4,2,2],
            upsample_kernel_sizes=[11,8,8,4,4],
            imput_dim=256+1+len(spkids),  # BN asr = 256 dim + F0 dim + One Hot encoding spk
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

            resblock = satools.nn.ResBlock1
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
            self.ups.apply(satools.nn.init_weights)
            self.conv_post.apply(satools.nn.init_weights)

        def forward(self, x):
            x = self.conv_pre(x)
            for i in range(self.num_upsamples):
                x = F.leaky_relu(x, satools.nn.LRELU_SLOPE)
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

            satools_path = os.path.join(satools.__path__[0], "../../egs/asr", args.bn_dataset)
            model_weight = "final.pt"

            # loading from args
            model = args.asr_tdnnf_model  # eg: "local/chain/e2e/tuning/tdnnf.py"
            exp_path = args.asr_tdnnf_exp_path  # eg: "exp/chain/e2e_tdnnf/"
            num_pdfs_train = satools.script_utils.read_single_param_file(
                os.path.join(satools_path, exp_path, "num_pdfs")
            )
            assert num_pdfs_train == self.bn_asr_output_dim

            self.bn_model_state = torch.load(
                os.path.join(satools_path, exp_path, model_weight), map_location="cpu", weights_only=False
            )
            self.bn_asr.load_state_dict(self.bn_model_state)

        def __init__(self, load_asr_weight=True, asr_bn_model=None):
            super().__init__()

            self.validating = False
            self.sample_size = None
            # Hifigan Model
            self.core_hifigan = CoreHifiGan()

            # No spk features (any/many to one speaker conversion)
            #  self.spkr = nn.Embedding(num_spkr, embedding_dim)

            if asr_bn_model != None:
                self.bn_asr = asr_bn_model
            else:
                # IMPORT LIB
                # ASR features
                bnargs = SimpleNamespace()
                if args.asr_tdnnf_vq != -1:
                    bnargs = SimpleNamespace(
                        freeze_encoder=True,
                        codebook_size=args.asr_tdnnf_vq,  # eg: 16
                    )
                if args.asr_tdnnf_dp != -1:
                    bnargs = SimpleNamespace(
                        freeze_encoder=True,
                        epsilon=str(args.asr_tdnnf_dp),  # eg: 180000
                    )

                satools_path = os.path.join(satools.__path__[0], "../../egs/asr", args.bn_dataset)
                model = args.asr_tdnnf_model  # eg: "local/chain/e2e/tuning/tdnnf.py"

                config_path = os.path.join(satools_path, model)
                if not os.path.exists(config_path):
                    raise FileNotFoundError(
                        "No file found at location {}".format(config_path)
                    )
                spec = importlib.util.spec_from_file_location("config", config_path)
                asr_model_file = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(asr_model_file)

                if args.bn_dataset == "mls":
                    self.bn_asr_output_dim = 3120
                else:
                    self.bn_asr_output_dim = 3280
                self.bn_asr = asr_model_file.build(bnargs)(
                    output_dim=self.bn_asr_output_dim
                )
                self.bn_asr.eval()

            if load_asr_weight:
                self.after_load_hook()

        # keep some Torch submodule in eval mode
        def train(self, mode=True):
            super().train(mode)
            self.bn_asr.eval()
            return self

        @torch.no_grad()
        def validate_model(self, device="cpu"):
            self.validating = True

            N = 1
            C = 125760
            x = torch.arange(N * C).reshape(N, C).float().to(device)
            f0 = satools.hifigan.f0.get_f0(
                x.cpu(),
                {"a": {"f0_mean": 209.04, "f0_std": 58.75}},
                cache_with_filename="/t/a"
            ).to(device)

            nnet_output = self.forward(f0=f0, audio=x, target=[spkids[0]])
            assert (
                nnet_output.shape[2] >= x.shape[-1] - 10000
                and nnet_output.shape[2] <= x.shape[-1] + 10000
            ), f"Mismatch too high in vocoder output shape - {nnet_output.shape} != {x.shape}"

            self.eval()
            nnet_output = self.forward(f0=f0, audio=x, target=[spkids[0]])
            assert (
                nnet_output.shape[2] >= x.shape[-1] - 10000
                and nnet_output.shape[2] <= x.shape[-1] + 10000
            ), f"Mismatch too high in vocoder output shape - {nnet_output.shape} != {x.shape}"
            self.train()
            self.validating = False

        @torch.no_grad()
        def extract_bn_only(self, audio):
            post, asr_out_xent = self.bn_asr(audio)
            bn_asr_h = self.bn_asr.bottleneck_out.permute(0, 2, 1)
            return bn_asr_h

        @torch.no_grad()
        @torch.amp.autocast('cuda', enabled=False)
        def extract_features(self, f0, audio):
            """
            Takes F0 features and the raw audio features
            returns F0 representation and audio representation (from ASR/SSL models)
            """
            bn_asr_h = self.extract_bn_only(audio)

            if self.validating:
                logging.info("F0 subsampling:" + str(audio.shape[1] / f0.shape[-1]))
                logging.info("ASR subsampling:" + str(audio.shape[1] / bn_asr_h.shape[-1]))

            return f0, bn_asr_h

        def forward(self, **kwargs):

            f0_h_q, bn_asr_h = self.extract_features(kwargs["f0"], kwargs["audio"])

            if spkids != None and "filenames" in kwargs:
                spk_ids = []
                for f in kwargs["filenames"]:
                    spk_id = os.path.basename(f).split("_")[0] # LibriTTS Training only
                    sid = [i for i,x in enumerate(spkids) if x == spk_id][0]
                    spk_ids.append(sid)
                one_hot = F.one_hot(torch.tensor(spk_ids), num_classes=len(spkids)).unsqueeze(1).to(kwargs["audio"].device)
            elif spkids != None and "target" in kwargs:
                spk_ids = []
                for s in kwargs["target"]:
                    sid = [i for i,x in enumerate(spkids) if str(x) == str(s)][0]
                    spk_ids.append(sid)
                one_hot = F.one_hot(torch.tensor(spk_ids), num_classes=len(spkids)).unsqueeze(1).to(kwargs["audio"].device)
            else:
                raise ValueError("Missing target speaker")

            f0_h_q = F.interpolate(f0_h_q, bn_asr_h.shape[-1])
            x = torch.cat([bn_asr_h, f0_h_q], dim=1)

            spkr = F.interpolate(one_hot.to(torch.float32).permute(0, 2, 1), x.shape[-1])
            x = torch.cat([x, spkr], dim=1)

            out = self.core_hifigan(x)
            return out

    return Net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model config args")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--lr", default=0.0002, type=float)
    parser.add_argument("--checkpoint_path", default="exp/hifigan_w2w2", type=str)
    parser.add_argument("--init_weight_model", default="last", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--training_epochs", default=1500, type=int)
    parser.add_argument("--cold_restart", default=False, action="store_true")
    parser.add_argument(
        "--asr_tdnnf_model", default="local/chain/e2e/tuning/tdnnf_wav2vec2.py", type=str
    )
    parser.add_argument(
        "--asr_tdnnf_exp_path", default="exp/chain/e2e_tdnnf_wav2vec2/", type=str
    )
    parser.add_argument("--asr_tdnnf_vq", default=-1, type=int)
    parser.add_argument("--asr_tdnnf_dp", default=-1, type=int)
    parser.add_argument("--bn_dataset", default="librispeech", type=str)
    parser.add_argument("--data_dir", default="./data/LibriTTS/wavs_16khz", type=str)
    parser.add_argument("--f0_stats", default="./data/LibriTTS/stats.json", type=str)
    parser.add_argument("--f0_cache", default=".f0_sr-320.cache", type=str)
    args, remaining_argv = parser.parse_known_args()
    sys.argv = sys.argv[:1] + remaining_argv

    torch.cuda.manual_seed(52)
    random.seed(52)
    wav_list = satools.utils.fs.scans_directory_for_ext(
        args.data_dir, "wav"
    )
    wav_list.sort()
    random.shuffle(wav_list)
    split = int(0.98 * len(wav_list))
    train_list = ",".join(wav_list) # All training data
    dev_list = ",".join(wav_list[split:])

    def _norm(f0, f0_stats, filename):
        # LibriTTS file format to extact spk id
        spk_id = os.path.basename(filename).split("_")[0]
        return satools.hifigan.f0.m_std_norm(f0, f0_stats[spk_id], filename)

    satools.hifigan.f0.set_norm_func(_norm)

    satools.hifigan.f0.set_cache_file(args.f0_cache)
    satools.hifigan.f0.set_yaapt_opts({
            "frame_length": 35.0,
            "frame_space": 20.0,
            "nccf_thresh1": 0.25,
            "tda_frame_length": 25.0,
        })

    f0_stats = open(args.f0_stats,'r').readline()
    spkids = list(json.loads(f0_stats).keys())
    spkids.sort()

    HifiGanModel(
        build(args, spkids),
        **{
            "mode": "train",
            "lr": args.lr,
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
            "f0_stats": f0_stats,
        },
    )
