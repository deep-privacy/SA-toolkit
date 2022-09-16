#!/usr/bin/env python3

# tg results on dev_clean
#  %WER 7.82 [ 4254 / 54402, 468 ins, 486 del, 3300 sub ]
# after fg rescoring
#  %WER 5.12 [ 2787 / 54402, 316 ins, 326 del, 2145 sub ]

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from satools.chain import ChainE2EModel
from satools.nn import (
    TDNNFBatchNorm,
    TDNNFBatchNorm_LD,
)

import satools

logging.basicConfig(level=logging.DEBUG)
import sys
import configargparse

import kaldifeat


def build(args):
    class Net(nn.Module):
        def __init__(
            self,
            output_dim,
            hidden_dim=1024, # 1600
            bottleneck_dim=128, # 160
            prefinal_bottleneck_dim=256,
            kernel_size_list=[3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3], # TODO : ajouter hidden layer 17
            subsampling_factor_list=[1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1],
            frame_subsampling_factor=3,
            p_dropout=0.1,
        ):
            super().__init__()

            # Preprocessor
            opts = kaldifeat.FbankOptions()
            self.features_opts = satools.utils.kaldifeat_set_option(
                opts,
                satools.__path__[0]
                + "/../egs/librispeech/v1/"
                + "./configs/fbank_hires.conf",
            )
            self.fbank = kaldifeat.Fbank(self.features_opts)

            # at present, we support only frame_subsampling_factor to be 3
            assert frame_subsampling_factor == 3

            assert len(kernel_size_list) == len(subsampling_factor_list)
            num_layers = len(kernel_size_list)
            input_dim = self.features_opts.mel_opts.num_bins

            self.cmvn = satools.cmvn.UttCMVN()

            # input_dim = feat_dim * 3 + ivector_dim
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.output_subsampling = frame_subsampling_factor

            # manually calculated
            self.padding = 27
            self.frame_subsampling_factor = frame_subsampling_factor

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
                tdnnfs.append(layer)
                dropout_layer = nn.Dropout(p_dropout)
                tdnnfs.append(dropout_layer)

            # tdnnfs requires [N, C, T]
            self.tdnnfs = nn.ModuleList(tdnnfs)

            def bottleneck_ld(x):
                self.bottleneck_out = x
                return x

            # prefinal_l affine requires [N, C, T]
            self.prefinal_chain_vq = TDNNFBatchNorm_LD(
                hidden_dim,
                hidden_dim,
                bottleneck_dim=prefinal_bottleneck_dim,
                context_len=1,
                orthonormal_constraint=-1.0,
                bottleneck_ld=bottleneck_ld,
            )
            self.prefinal_xent = TDNNFBatchNorm(
                hidden_dim,
                hidden_dim,
                bottleneck_dim=prefinal_bottleneck_dim,
                context_len=1,
                orthonormal_constraint=-1.0,
            )

            self.chain_output = satools.nn.NaturalAffineTransform(hidden_dim, output_dim)
            self.chain_output.weight.data.zero_()
            self.chain_output.bias.data.zero_()

            self.xent_output = satools.nn.NaturalAffineTransform(hidden_dim, output_dim)
            self.xent_output.weight.data.zero_()
            self.xent_output.bias.data.zero_()

            self.validate_model()

        @torch.no_grad()
        def validate_model(self):
            N = 2
            C = (10 * self.frame_subsampling_factor) * 274
            x = torch.arange(N * C).reshape(N, C).float()
            nnet_output, xent_output = self.forward(x)
            assert (
                nnet_output.shape[1] == 17
            ), f"{nnet_output.shape[1]} != expected frame subsampling"

            self.eval()
            nnet_output, xent_output = self.forward(x)
            assert (
                nnet_output.shape[1] == 17
            ), f"{nnet_output.shape[1]} != expected frame subsampling"
            self.train()

        def pad_input(self, x):
            if self.padding > 0:
                N, T, C = x.shape
                left_pad = x[:, 0:1, :].repeat(1, self.padding, 1).reshape(N, -1, C)
                right_pad = x[:, -1, :].repeat(1, self.padding, 1).reshape(N, -1, C)
                x = torch.cat([left_pad, x, right_pad], axis=1)
            return x

        def forward(self, x, spec_augment=lambda x: x):
            assert x.ndim == 2
            # input x is of shape: [batch_size, wave] = [N, C]

            if self.features_opts.device != x.device:
                self.features_opts.device = x.device
                self.fbank = kaldifeat.Fbank(self.features_opts)

            # To compute features that are compatible with Kaldi, wave samples have to be scaled to the range [-32768, 32768]
            x *= 32768
            waveform = [*x]  # batch processing with python list (required by kaldifeat)

            x = self.fbank(waveform)
            x = torch.stack(x)  # back to tensor
            assert x.ndim == 3
            x = self.pad_input(x)
            x = self.cmvn(x)
            x = spec_augment(x)
            # x is of shape: [batch_size, seq_len, feat_dim] = [N, T, C]
            # at this point, x is [N, T, C]
            x = self.tdnn1(x)
            x = self.dropout1(x)

            # tdnnf requires input of shape [N, C, T]
            for i in range(len(self.tdnnfs)):
                x = self.tdnnfs[i](x)

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
    ChainE2EModel(build(args), cmd_line=True)
