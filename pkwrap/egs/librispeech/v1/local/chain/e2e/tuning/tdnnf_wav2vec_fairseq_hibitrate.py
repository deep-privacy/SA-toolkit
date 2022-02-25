#!/usr/bin/env python3
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Written by Apoorv Vyas <apoorv.vyas@idiap.ch>
#             Srikanth Madikeri <srikanth.madikeri@idiap.ch>

# tg results on dev_clean
#  ??
# after fg rescoring
#  ??

import torch
import torch.nn.functional as F
import torch.nn as nn
import pkwrap
from pkwrap.nn import (
    TDNNFBatchNorm,
    NaturalAffineTransform,
    OrthonormalLinear,
    VectorQuantizerEMA,
    TDNNFBatchNorm_LD,
)
from pkwrap.chain import ChainE2EModel
import numpy as np
from torch.nn.utils import clip_grad_value_
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("hydra").setLevel(logging.WARNING)
import sys
import os
import configargparse
import fairseq


def build(args):
    class Net(nn.Module):
        def __init__(
            self,
            output_dim,
            hidden_dim=1024,
            bottleneck_dim=128,
            prefinal_bottleneck_dim=256,
            # fmt: off
            kernel_size_list=       [3, 3, 3, 1, 3, 3, 3, 3, 3], # context_len=3 out = N-2
            subsampling_factor_list=[1, 1, 1, 3, 1, 1, 1, 1, 1],
            # fmt: on
            frame_subsampling_factor=3,
            p_dropout=0.1,
        ):
            super().__init__()

            #  https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt
            #  https://dl.fbaipublicfiles.com/voxpopuli/models/wav2vec2_base_en_v2.pt
            #  https://dl.fbaipublicfiles.com/voxpopuli/models/wav2vec2_large_west_germanic_v2.pt
            model = "wav2vec2_large_west_germanic_v2.pt"
            url = "https://dl.fbaipublicfiles.com/voxpopuli/models/"
            model_cache_file = os.path.join(torch.hub.get_dir(), model)
            if not os.path.exists(model_cache_file):
                os.makedirs(torch.hub.get_dir(), exist_ok=True)
                torch.hub.download_url_to_file(f"{url}{model}", model_cache_file, hash_prefix="")
            (
                feat_model,
                cfg,
                task,
            ) = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [str(model_cache_file)]
            )

            self.preprocessor = feat_model[0]
            input_dim = 1024  # self.preprocessor output dim

            # at present, we support only frame_subsampling_factor to be 3
            assert frame_subsampling_factor == 3

            assert len(kernel_size_list) == len(subsampling_factor_list)
            num_layers = len(kernel_size_list)

            # input_dim = feat_dim * 3 + ivector_dim
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.output_subsampling = frame_subsampling_factor

            # manually calculated applied to the left and right x2
            self.padding = 3
            self.padding_after_ld = 5
            self.frame_subsampling_factor = frame_subsampling_factor

            def bottleneck_ld(x):
                self.bottleneck_out = x
                return x

            self.tdnn1 = TDNNFBatchNorm(
                input_dim,
                hidden_dim,
                bottleneck_dim=bottleneck_dim,
                context_len=kernel_size_list[0],
                subsampling_factor=subsampling_factor_list[0],
                orthonormal_constraint=-1.0,
            )
            self.dropout1 = nn.Dropout(p_dropout)
            tdnnfs = []
            for i in range(1, num_layers):
                kernel_size = kernel_size_list[i]
                subsampling_factor = subsampling_factor_list[i]
                layer = TDNNFBatchNorm(
                    hidden_dim,
                    hidden_dim,
                    bottleneck_dim=bottleneck_dim,
                    context_len=kernel_size,
                    subsampling_factor=subsampling_factor,
                    orthonormal_constraint=-1.0,
                )

                if (
                    i + 2 < len(kernel_size_list)
                    and subsampling_factor_list[i + 1] == 3
                ):
                    layer = TDNNFBatchNorm_LD(
                        hidden_dim,
                        hidden_dim,
                        bottleneck_dim=prefinal_bottleneck_dim,
                        context_len=kernel_size,
                        subsampling_factor=subsampling_factor,
                        orthonormal_constraint=-1.0,
                        bottleneck_ld=bottleneck_ld,
                        bypass_scale=0.0,  # no skip connection to constrain to the output of LD
                    )
                    assert layer.tdnn.use_bypass == False

                tdnnfs.append(layer)
                dropout_layer = nn.Dropout(p_dropout)
                tdnnfs.append(dropout_layer)

            # tdnnfs requires [N, C, T]
            self.tdnnfs = nn.ModuleList(tdnnfs)

            # prefinal_l affine requires [N, C, T]
            self.prefinal_chain_vq = TDNNFBatchNorm(
                hidden_dim,
                hidden_dim,
                bottleneck_dim=prefinal_bottleneck_dim,
                context_len=1,
                orthonormal_constraint=-1.0,
            )

            self.prefinal_xent = TDNNFBatchNorm(
                hidden_dim,
                hidden_dim,
                bottleneck_dim=prefinal_bottleneck_dim,
                context_len=1,
                orthonormal_constraint=-1.0,
            )

            self.chain_output = pkwrap.nn.NaturalAffineTransform(hidden_dim, output_dim)
            self.chain_output.weight.data.zero_()
            self.chain_output.bias.data.zero_()

            self.xent_output = pkwrap.nn.NaturalAffineTransform(hidden_dim, output_dim)
            self.xent_output.weight.data.zero_()
            self.xent_output.bias.data.zero_()

            self.validate_model()

        def set_lr_layers_for_optim(self, get_optimizer, lr, weight_decay, iter=0, total_iter=-1):

            def set_parameter_requires_grad(model, yes=False):
                for param in model.parameters():
                    param.requires_grad = yes

            self.preprocessor.train()
            if iter < total_iter * 0.02 or iter > total_iter * 0.90:
                logging.info("Preprocesor in eval mode!")
                set_parameter_requires_grad(self.preprocessor)
                self.preprocessor.eval()

            wav2vec = []
            tdnnf = []
            for name, param in self.named_parameters():
                if "preprocessor" in name:
                    wav2vec.append(param)
                else:
                    tdnnf.append(param)
            opti = get_optimizer(
                [{"params": wav2vec}, {"params": tdnnf}], lr, weight_decay
            )

            opti.param_groups[0]["lr"] = lr / 20
            if iter > total_iter * 0.20 and iter < total_iter * 0.70:
                opti.param_groups[0]["lr"] = lr / 5
            opti.param_groups[1]["lr"] = lr
            logging.info("LR: " + str(opti.param_groups[1]["lr"]))
            logging.info("Preprocesor LR: " + str(opti.param_groups[0]["lr"]))

            return opti

        @torch.no_grad()
        def validate_model(self):
            N = 2
            C = 10 * self.frame_subsampling_factor * 320 # 320 been the wav2vec2 subsampling factor
            x = torch.arange(N * C).reshape(N, C).float()
            nnet_output, xent_output = self.forward(x)
            assert (
                nnet_output.shape[1] == 10
            ), f"{nnet_output.shape[1]} != expected frame subsampling"

            self.eval()
            nnet_output, xent_output = self.forward(x)
            assert (
                nnet_output.shape[1] == 10
            ), f"{nnet_output.shape[1]} != expected frame subsampling"
            self.train()

        def pad_input(self, x, pad_value=0):
            if pad_value > 0:
                N, T, C = x.shape
                left_pad = x[:, 0:1, :].repeat(1, pad_value, 1).reshape(N, -1, C)
                right_pad = x[:, -1, :].repeat(1, pad_value, 1).reshape(N, -1, C)
                x = torch.cat([left_pad, x, right_pad], axis=1)
            return x

        def forward(self, x, spec_augment=lambda x: x):
            assert x.ndim == 2
            # input x is of shape: [batch_size, wave] = [N, C]

            with torch.cuda.amp.autocast():
                #  print(x.shape)
                #  a = x.shape[1]
                x = self.preprocessor(x, mask=False, features_only=True)["x"]
                x = x.transpose(2, 1)
                x = torch.nn.functional.pad(x, (0, 1), "replicate")
                x = x.transpose(2, 1)
                #  print(x.shape, a / x.shape[1])

            assert x.ndim == 3
            #  print("help start:", x.shape)
            x = self.pad_input(x, pad_value=self.padding)
            # x is of shape: [batch_size, seq_len, feat_dim] = [N, T, C]
            # at this point, x is [N, T, C]
            x = self.tdnn1(x)
            x = self.dropout1(x)

            # tdnnf requires input of shape [N, C, T]
            for i in range(len(self.tdnnfs)):
                #  print("help:", x.shape)
                if isinstance(self.tdnnfs[i], TDNNFBatchNorm):
                    if self.tdnnfs[i-2].tdnn.subsampling_factor == 3: # For that network topoligie (tdnn+drop)
                        # padding after bottleneck 'LD' extraction and network subsampling
                        x = self.pad_input(x, pad_value=self.padding_after_ld)
                x = self.tdnnfs[i](x)
            #  print("help:", x.shape)


            chain_prefinal_out = self.prefinal_chain_vq(x)
            xent_prefinal_out = self.prefinal_xent(x)

            chain_out = self.chain_output(chain_prefinal_out)
            xent_out = self.xent_output(xent_prefinal_out)
            return chain_out, F.log_softmax(xent_out, dim=2)

    return Net


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(description="Model config args")
    args, remaining_argv = parser.parse_known_args()
    sys.argv = sys.argv[:1] + remaining_argv
    if os.environ.get("TESTING", "0") == "1":
        model = build(args)(output_dim=1233).cuda()
        for C in [8000, 16000, 32000, 48000, 64000]:
            N = 1
            #  C = 8000
            x = torch.arange(N * C).reshape(N, C).float().cuda()
            nnet_output, xent_output = model.forward(x)
            print(C, model.bottleneck_out.shape, C / model.bottleneck_out.shape[1])
            print(nnet_output.shape, C / nnet_output.shape[1])
        sys.exit(0)

    ChainE2EModel(build(args), cmd_line=True)
