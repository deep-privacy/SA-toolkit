#!/usr/bin/env python3

import logging

import satools
import torch
import torch.nn as nn
import torch.nn.functional as F
from satools.chain import ChainE2EModel
from satools.nn import (
    TDNNFBatchNorm,
    TDNNFBatchNorm_LD,
)

import sys
import argparse

import torchaudio
from scipy.stats import laplace


def build(args):
    class Net(nn.Module):
        def __init__(
            self,
            output_dim,
            hidden_dim=1024,
            bottleneck_dim=128,
            prefinal_bottleneck_dim=256,
            kernel_size_list=[3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3],
            subsampling_factor_list=[1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1],
            frame_subsampling_factor=3,
            p_dropout=0.1,
        ):
            super().__init__()

            # Preprocessor
            self.fbank = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80)

            # at present, we support only frame_subsampling_factor to be 3
            assert frame_subsampling_factor == 3

            assert len(kernel_size_list) == len(subsampling_factor_list)
            num_layers = len(kernel_size_list)
            input_dim = self.fbank.n_mels

            self.cmvn = satools.cmvn.UttCMVN()

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

            logging.info("Using epsilon: " + args.epsilon)
            self.eps = float(args.epsilon)

            def bottleneck_ld(x):
                x = F.normalize(x, p=1, dim=2)  # L1 norm
                mu = 0
                delta = [(1 - (0)) * (x.shape[0] * x.shape[1]) / self.eps]
                dist = laplace(mu, delta)
                noises = dist.rvs(size=x.shape)
                x = x + torch.from_numpy(noises).float().to(x.device)
                x = F.normalize(x, p=1, dim=2)  # L1 norm
                self.bottleneck_out = x

                return x

            # prefinal_l affine requires [N, C, T]
            self.prefinal_chain_ld = TDNNFBatchNorm_LD(
                hidden_dim,
                hidden_dim,
                bottleneck_dim=prefinal_bottleneck_dim,
                context_len=1,
                orthonormal_constraint=-1.0,
                bottleneck_ld=bottleneck_ld,
                bypass_scale=0.0,  # no skip connection to constrain to the output of LD
            )
            assert self.prefinal_chain_ld.tdnn.use_bypass == False

            ####################
            #  Bigger decoder  #
            ####################
            tdnnfs = []
            for i in range(0, 2):
                kernel_size = 3
                subsampling_factor = 1
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
            self.tdnnfs_decode = nn.ModuleList(tdnnfs)

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

            if args.freeze_encoder == "True":
                logging.debug("Freezing encoder!")

                switch_require_grad = False
                for name, param in self.named_parameters():
                    if name == "tdnnfs.18.tdnn.linearB.weight":
                        switch_require_grad = True
                    param.requires_grad = switch_require_grad
                    logging.debug(name + f" - requires_grad={param.requires_grad}")

            self.validate_model()

        @torch.no_grad()
        def validate_model(self):
            self.eval()
            N = 2
            C = (10 * self.frame_subsampling_factor) * 274
            x = torch.arange(N * C).reshape(N, C).float()
            nnet_output, xent_output = self.forward(x)
            assert (
                nnet_output.shape[1] == 13
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

            x = self.fbank(x).permute(0, 2, 1)
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

            chain_prefinal_out = self.prefinal_chain_ld(x)

            #  xent_prefinal_out = self.prefinal_xent(x)
            x = chain_prefinal_out
            for i in range(len(self.tdnnfs_decode)):
                x = self.tdnnfs_decode[i](x)
            xent_prefinal_out = self.prefinal_xent(x)
            chain_prefinal_out = x

            chain_out = self.chain_output(chain_prefinal_out)
            xent_out = self.xent_output(xent_prefinal_out)
            return chain_out, F.log_softmax(xent_out, dim=2)

    return Net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model config args")
    parser.add_argument("--freeze-encoder", default="False", type=str)
    parser.add_argument("--epsilon", default="1.0", type=str)
    args, remaining_argv = parser.parse_known_args()
    sys.argv = sys.argv[:1] + remaining_argv
    ChainE2EModel(build(args), cmd_line=True)
