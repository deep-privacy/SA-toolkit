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
from pkwrap.nn import TDNNFBatchNorm, NaturalAffineTransform, OrthonormalLinear, VectorQuantizerEMA, TDNNFBatchNorm_LD
from pkwrap.chain import ChainE2EModel
import numpy as np
from torch.nn.utils import clip_grad_value_
import logging
logging.basicConfig(level=logging.DEBUG)
import sys
import os
import configargparse


def build(args):
    class Net(nn.Module):

        def __init__(self,
                     output_dim,
                     hidden_dim=1024,
                     bottleneck_dim=128,
                     prefinal_bottleneck_dim=256,
                     kernel_size_list=[3, 3, 3, 1, 3, 3, 3, 3, 3],
                     subsampling_factor_list=[1, 1, 1, 3, 1, 1, 1, 1, 1],
                     frame_subsampling_factor=3,
                     p_dropout=0.1):
            super().__init__()

            self.preprocessor = pkwrap.huggingface.HuggingFaceWav2Vec2(
                "facebook/wav2vec2-base-960h",
                freeze=True,
                freeze_feature_extractor=True,
            )
            input_dim = 768 # self.preprocessor output dim
            

            # at present, we support only frame_subsampling_factor to be 3
            assert frame_subsampling_factor == 3

            assert len(kernel_size_list) == len(subsampling_factor_list)
            num_layers = len(kernel_size_list)

            #input_dim = feat_dim * 3 + ivector_dim
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.output_subsampling = frame_subsampling_factor

            # manually calculated
            self.padding = 27
            self.frame_subsampling_factor = frame_subsampling_factor


            self.acc_sum_vq = torch.tensor(0., requires_grad=False)
            self.acc_sum_perplexity = torch.tensor(0., requires_grad=False)
            self.quant = VectorQuantizerEMA(args.codebook_size, prefinal_bottleneck_dim, 0.25, 0.99)
            def bottleneck_ld(x):
                vq_loss, x, perplexity, _, _, encoding_indices, \
                            losses, _, _, _, concatenated_quantized = self.quant(x)

                self.encoding_indices = encoding_indices
                self.vq_loss = vq_loss
                self.perplexity = perplexity
                self.bottleneck_out = x
                return x


            self.tdnn1 = TDNNFBatchNorm(
                input_dim, hidden_dim,
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

                if i+2 < len(kernel_size_list) and kernel_size_list[i+1] == 1:
                    layer = TDNNFBatchNorm_LD(
                        hidden_dim,
                        hidden_dim,
                        bottleneck_dim=prefinal_bottleneck_dim,
                        context_len=kernel_size,
                        subsampling_factor=subsampling_factor,
                        orthonormal_constraint=-1.0,
                        bottleneck_ld=bottleneck_ld,
                        bypass_scale=0.0, # no skip connection to constrain to the output of LD
                    )
                    assert subsampling_factor == 1
                    assert layer.tdnn.use_bypass == False

                tdnnfs.append(layer)
                dropout_layer = nn.Dropout(p_dropout)
                tdnnfs.append(dropout_layer)

            # tdnnfs requires [N, C, T]
            self.tdnnfs = nn.ModuleList(tdnnfs)

            # prefinal_l affine requires [N, C, T]
            self.prefinal_chain_vq = TDNNFBatchNorm(
                hidden_dim, hidden_dim,
                bottleneck_dim=prefinal_bottleneck_dim,
                context_len=1,
                orthonormal_constraint=-1.0,
                bypass_scale=0.0, # no skip connection to constrain to the output of LD
            )
            assert self.prefinal_chain_vq.tdnn.use_bypass == False

            ####################
            #  Bigger decoder  #
            ####################
            #  tdnnfs = []
            #  for i in range(0, 1):
                #  kernel_size = 3
                #  subsampling_factor = 1
                #  layer = TDNNFBatchNorm(
                    #  hidden_dim,
                    #  hidden_dim,
                    #  bottleneck_dim=bottleneck_dim,
                    #  context_len=kernel_size,
                    #  subsampling_factor=subsampling_factor,
                    #  orthonormal_constraint=-1.0,
                #  )
                #  tdnnfs.append(layer)
                #  dropout_layer = nn.Dropout(p_dropout)
                #  tdnnfs.append(dropout_layer)
            #  # tdnnfs requires [N, C, T]
            #  self.tdnnfs_decode = nn.ModuleList(tdnnfs)



            self.prefinal_xent = TDNNFBatchNorm(
                hidden_dim, hidden_dim,
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

            if args.freeze_encoder == "True":
                self.preprocessor.freeze = True
                self.preprocessor.model.eval()
                logging.info("Freezing encoder!")

                switch_require_grad = False
                for name, param in self.named_parameters():
                    if name.startswith("quant"):
                        continue
                    if name=="tdnnfs.0.tdnn.linearB.weight":
                        switch_require_grad = True
                    param.requires_grad = switch_require_grad
                    logging.info(name + f" - requires_grad={param.requires_grad}")

            self.validate_model()

        @torch.no_grad()
        def validate_model(self):
            N = 2
            C = (10 * self.frame_subsampling_factor) * 274
            x = torch.arange(N * C).reshape(N, C).float()
            nnet_output, xent_output = self.forward(x)
            assert nnet_output.shape[1] == 15, f"{nnet_output.shape[1]} != expected frame subsampling"

            self.eval()
            nnet_output, xent_output = self.forward(x)
            assert nnet_output.shape[1] == 15, f"{nnet_output.shape[1]} != expected frame subsampling"
            self.train()

        def pad_input(self, x):
            if self.padding > 0:
                N, T, C = x.shape
                left_pad = x[:,0:1,:].repeat(1,self.padding,1).reshape(N, -1, C)
                right_pad = x[:,-1,:].repeat(1,self.padding,1).reshape(N, -1, C)
                x = torch.cat([left_pad, x, right_pad], axis=1)
            return x


        def vq(self):
            return True

        def forward(self, x, spec_augment=lambda x: x):
            assert x.ndim == 2
            # input x is of shape: [batch_size, wave] = [N, C]

            with torch.cuda.amp.autocast():
                print(x.shape)
                x = self.preprocessor(x)
                print(x.shape)
                x = x.transpose(2,1)
                x = torch.nn.functional.pad(x, (0, 3), 'replicate')
                print(x.shape)
                x = x.transpose(2,1)
                print(x.shape)
                #  x = self.preprocessor(x, spec_aug=self.training)

            assert x.ndim == 3
            x = self.pad_input(x)
            # x is of shape: [batch_size, seq_len, feat_dim] = [N, T, C]
            # at this point, x is [N, T, C]
            x = self.tdnn1(x)
            x = self.dropout1(x)

            # tdnnf requires input of shape [N, C, T]
            for i in range(len(self.tdnnfs)):
                x = self.tdnnfs[i](x)

            chain_prefinal_out = self.prefinal_chain_vq(x)
            xent_prefinal_out = self.prefinal_xent(x)

            #  x = chain_prefinal_out
            #  for i in range(len(self.tdnnfs_decode)):
                #  x = self.tdnnfs_decode[i](x)
            #  xent_prefinal_out = self.prefinal_xent(x)
            #  chain_prefinal_out = x

            chain_out = self.chain_output(chain_prefinal_out)
            xent_out = self.xent_output(xent_prefinal_out)
            return chain_out, F.log_softmax(xent_out, dim=2)


        def codebook_analysis(self):
            return self.quant

        def additional_obj(self, deriv, should_log=False, print_interval=1, tensorboard=None, mb_id=1, for_valid=False):
            if deriv != None and self.vq_loss != None:

                if for_valid and print_interval > 1:
                    logging.info("Overall VQ objf={}".format(self.acc_sum_vq/print_interval))
                    if tensorboard: tensorboard.add_scalar('VQ_objf/valid', self.acc_sum_vq/print_interval, mb_id)
                    logging.info("VQ perplexity ={}".format(self.acc_sum_perplexity/print_interval))
                    if tensorboard: tensorboard.add_scalar('VQ_perplexity/valid', self.acc_sum_perplexity/print_interval, mb_id)
                    self.acc_sum_vq.zero_()
                    self.acc_sum_perplexity.zero_()
                    return

                if for_valid:
                    self.acc_sum_vq.add_(self.vq_loss.item()*deriv) # deriv here is the mini_batchsize*num_seq
                    self.acc_sum_perplexity.add_(self.perplexity.item()*deriv)
                    return

                self.acc_sum_vq.add_(self.vq_loss.item())
                self.acc_sum_perplexity.add_(self.perplexity.item())

                if not self.quant.freeze:
                    deriv += self.vq_loss.to(deriv.device)
                if should_log:
                    logging.info("Overall VQ objf={}".format(self.acc_sum_vq/print_interval))
                    if tensorboard: tensorboard.add_scalar('VQ_objf/train', self.acc_sum_vq/print_interval, mb_id)
                    self.acc_sum_vq.zero_()

                if should_log:
                    logging.info("VQ perplexity ={}".format(self.acc_sum_perplexity/print_interval))
                    if tensorboard: tensorboard.add_scalar('VQ_perplexity/train', self.acc_sum_perplexity/print_interval, mb_id)
                    self.acc_sum_perplexity.zero_()

        def set_lr_layers_for_optim(self, get_optimizer, lr, weight_decay, iter=0):
            TOTAL_ITER = 30

            around_vq = []
            other = []
            for name, param in self.named_parameters():
                                        #0  2  4  6  8  10 12 14
                 #  kernel_size_list=[3, 3, 3, 1, 3, 3, 3, 3, 3],
                if 'tdnnfs.0' in name or 'tdnnfs.2'in name or 'tdnnfs.4' in name or 'tdnnfs.6.tdnn' in name or 'tdnnfs.8.tdnn' in name:
                    around_vq.append(param)
                else:
                    other.append(param)
            opti = get_optimizer([{'params':around_vq}, {'params':other}], lr, weight_decay)

            opti.param_groups[0]['lr'] = lr
            opti.param_groups[1]['lr'] = lr/10

            return opti

    return Net

if __name__ == '__main__':
    parser = configargparse.ArgumentParser(description="Model config args")
    parser.add("--freeze-encoder", default="False", type=str)
    parser.add("--codebook-size", default=255, type=int)
    args, remaining_argv = parser.parse_known_args()
    sys.argv = sys.argv[:1]+remaining_argv
    if os.environ.get('TESTING', "0") == "1":
        model = build(args)(output_dim=1233)
        N = 1
        C = 8960
        x = torch.arange(N * C).reshape(N, C).float()
        model.padding = 27
        nnet_output, xent_output = model.forward(x)
        print(model.bottleneck_out.shape, C/model.bottleneck_out.shape[1])
        sys.exit(0)

    ChainE2EModel(build(args), cmd_line=True)
