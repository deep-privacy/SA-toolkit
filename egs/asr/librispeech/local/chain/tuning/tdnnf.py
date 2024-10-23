#!/usr/bin/env python3

import logging
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import satools
import satools.nn as sann
from satools.chain import ChainE2EModel

import sys
import argparse



def build(args):
    class Net(nn.Module):
        def __init__(
            self,
            output_dim,
            hidden_dim=1024,
            bottleneck_dim=128,
            prefinal_bottleneck_dim=256,
            #                                                         \ /== Extract BN here
            kernel_size_list=       [[3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3], [1,   3, 3, 3]],
            subsampling_factor_list=[[1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1], [1.5, 1, 1, 1]],
            #                                 / \                          /   \== Padding to for sub sampling = 3 necessary for decoding
            #                  Fbank -> 160 -> /2 -> 320 (BN) -> /1.5 -> ~480 ASR
            p_dropout=0.1,
        ):
            super().__init__()

            # Preprocessor
            self.input_dim = 80


            self.cmvn = satools.cmvn.UttCMVN()

            self.output_dim = output_dim

            # applied to the left and right x2
            self.padding = ChainE2EModel.get_padding(kernel_size_list[0], subsampling_factor_list[0]) // 2
            self.padding_after = ChainE2EModel.get_padding(kernel_size_list[1], subsampling_factor_list[1]) // 2

            # input layer
            self.tdnn1 = sann.TDNNFBatchNorm(
                self.input_dim,
                hidden_dim,
                bottleneck_dim=bottleneck_dim,
                context_len=kernel_size_list[0][0],
                subsampling_factor=subsampling_factor_list[0][0],
                orthonormal_constraint=-1.0,
            )
            self.dropout1 = nn.Dropout(p_dropout)
            # 1st layers
            tdnnfs = []
            for i in range(1, len(kernel_size_list[0])-1):
                kernel_size = kernel_size_list[0][i]
                subsampling_factor = subsampling_factor_list[0][i]
                layer = sann.TDNNFBatchNorm(
                    hidden_dim,
                    hidden_dim,
                    bottleneck_dim=bottleneck_dim,
                    context_len=kernel_size,
                    subsampling_factor=subsampling_factor,
                    orthonormal_constraint=-1.0,
                )
                tdnnfs.append(layer)
                tdnnfs.append(nn.Dropout(p_dropout))

            # BN layer
            kernel_size = kernel_size_list[0][i+1]
            subsampling_factor = subsampling_factor_list[0][i+1]
            layer = sann.TDNNFBatchNorm(
                hidden_dim,
                hidden_dim,
                bottleneck_dim=prefinal_bottleneck_dim,
                context_len=kernel_size,
                subsampling_factor=subsampling_factor,
                orthonormal_constraint=-1.0,
                bypass_scale=0.0,  # no skip connections
            )
            assert layer.tdnn.use_bypass == False
            tdnnfs.append(layer)
            tdnnfs.append(nn.Dropout(p_dropout))

            # 2nd layers
            tdnnfs_after = []
            for i in range(0, len(kernel_size_list[1])):
                kernel_size = kernel_size_list[1][i]
                subsampling_factor = subsampling_factor_list[1][i]
                layer = sann.TDNNFBatchNorm(
                    hidden_dim,
                    hidden_dim,
                    bottleneck_dim=bottleneck_dim,
                    context_len=kernel_size,
                    subsampling_factor=subsampling_factor,
                    orthonormal_constraint=-1.0,
                )
                tdnnfs_after.append(layer)
                tdnnfs_after.append(nn.Dropout(p_dropout))

            # tdnnfs requires [N, C, T]
            self.tdnnfs = nn.Sequential(*tdnnfs)
            self.tdnnfs_after = nn.Sequential(*tdnnfs_after)

            # prefinal_l affine requires [N, C, T]
            self.prefinal_chain = sann.TDNNFBatchNorm(
                hidden_dim,
                hidden_dim,
                bottleneck_dim=prefinal_bottleneck_dim,
                context_len=1,
                orthonormal_constraint=-1.0,
            )
            self.prefinal_xent = sann.TDNNFBatchNorm(
                hidden_dim,
                hidden_dim,
                bottleneck_dim=prefinal_bottleneck_dim,
                context_len=1,
                orthonormal_constraint=-1.0,
            )

            self.chain_output = sann.NaturalAffineTransform(hidden_dim, output_dim)
            self.chain_output.weight.data.zero_()
            self.chain_output.bias.data.zero_()

            self.xent_output = sann.NaturalAffineTransform(hidden_dim, output_dim)
            self.xent_output.weight.data.zero_()
            self.xent_output.bias.data.zero_()

            self.validate_model()

        @torch.no_grad()
        def validate_model(self):
            self.eval()
            N = 2
            C = 16000*2
            x = torch.arange(N * C).reshape(N, C).float()
            nnet_output, xent_output = self.forward(x)
            assert (
                nnet_output.shape[1] == 66
            ), f"{nnet_output.shape[1]} != expected frame subsampling"
            self.train()

        def pad_input(self, x, pad_amount:int=0):
            if pad_amount > 0:
                N, T, C = x.shape
                left_pad = x[:, 0:1, :].repeat(1, pad_amount, 1).reshape(N, -1, C)
                right_pad = x[:, -1, :].repeat(1, pad_amount, 1).reshape(N, -1, C)
                x = torch.cat([left_pad, x, right_pad], 1)
            return x

        @torch.jit.export
        def extract_bn(self, x: torch.Tensor) -> torch.Tensor:
            """
            inputs: a 2-dimensional tensor [N, C]
            return: an 3-dimensional tensor [N, T, C]
            """
            x *= 32768
            a = x
            x = satools.kaldifeat.fbank(x, num_mel_bins=self.input_dim, snip_edges=False)

            x = self.cmvn(x)
            x = self.pad_input(x, pad_amount=self.padding)
            # x is of shape: [batch_size, seq_len, feat_dim] = [N, T, C]
            # at this point, x is [N, T, C]
            x = self.tdnn1(x)
            #  x = self.dropout1(x)

            for i, t in enumerate(self.tdnnfs[:-2]):
                x = t.forward(x)
            x = self.tdnnfs[-2].forward(x, return_bottleneck=True)
            #  x = self.tdnnfs[-1].forward(x) # dropout layer
            return x

        def forward(self, x):
            #  assert x.ndim == 2
            # input x is of shape: [batch_size, wave] = [N, C]

            # To compute features that are compatible with Kaldi, wave samples have to be scaled to the range [-32768, 32768]
            x *= 32768
            x = satools.kaldifeat.fbank(x, num_mel_bins=self.input_dim, snip_edges=False)
            #  assert x.ndim == 3

            x = self.cmvn(x)
            x = self.pad_input(x, pad_amount=self.padding)
            # x is of shape: [batch_size, seq_len, feat_dim] = [N, T, C]
            # at this point, x is [N, T, C]
            x = self.tdnn1(x)
            x = self.dropout1(x)

            x = self.tdnnfs(x)
            x = self.pad_input(x, pad_amount=self.padding_after)
            x = self.tdnnfs_after(x)

            chain_prefinal_out = self.prefinal_chain(x)
            xent_prefinal_out = self.prefinal_xent(x)

            chain_out = self.chain_output(chain_prefinal_out)
            xent_out = self.xent_output(xent_prefinal_out)
            return chain_out, F.log_softmax(xent_out, dim=2)

    return Net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model config args")
    args, remaining_argv = parser.parse_known_args()
    sys.argv = sys.argv[:1] + remaining_argv + ["--base-model-args", json.dumps(vars(args))]

    # bash $ tdnnf.py --mode test
    def _test(model):
        model = model(output_dim=1233)
        #  model = torch.jit.script(model)
        #  torch.jit.save(model, "a.jit")

        for C in [8000, 16000, 32000, 48000, 64000, 8192, 16640, 8192*2]:
            N = 1
            #  C = 8000
            x = torch.arange(N * C).reshape(N, C).float()
            bn = model.extract_bn(x)
            print(C, bn.shape, C / bn.shape[1])

    ChainE2EModel(build(args), testfn=_test, cmd_line=True)
