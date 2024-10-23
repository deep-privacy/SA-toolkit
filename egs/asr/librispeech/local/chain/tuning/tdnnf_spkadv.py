#!/usr/bin/env python3

# tg results on dev_clean
#  %WER 7.82 [ 4254 / 54402, 468 ins, 486 del, 3300 sub ]
# after fg rescoring
#  %WER 5.12 [ 2787 / 54402, 316 ins, 326 del, 2145 sub ]

import logging
import os

import satools
import torch
import torch.nn as nn
import torch.nn.functional as F
from satools.chain import ChainE2EModel
from satools.nn import (
    TDNNFBatchNorm,
    TDNNFBatchNorm_LD,
    RevGrad,
)

import sys
import argparse

import torchaudio

import sidekit.nnet

#  A gradient reversal function which reverses the gradient in the backward pass.
revgrad = RevGrad.apply

# XVector from sidekit taking asr bn input features
class XVector(nn.Module):
    def __init__(self, asr_out_dim, number_of_spk):
        super().__init__()
        self.asr_out_dim = asr_out_dim # bottleneck_out dim from the acoustic model
        self.number_of_spk = number_of_spk  # Number of speaker in the dataset (Libri train-clean 100)

        self.embedding_size = 256            # dim x-vector
        self.sequence_network = sidekit.nnet.PreHalfResNet34()
        self.stat_pooling = sidekit.nnet.AttentivePooling(8192, 1, global_context=False)
        self.before_speaker_embedding = torch.nn.Linear(in_features = int(((2560*2)/80)*self.asr_out_dim),
                                                            out_features = self.embedding_size)
        self.after_speaker_embedding = sidekit.nnet.ArcMarginProduct(self.embedding_size,
                int(self.number_of_spk),
                s = 30,
                m = 0.2,
                easy_margin = False)

        self.optim = None
        self.optim_checkpoint = "exp/optim_scheduler_dict.pt"

    def save_optimizer(self):
        if self.optim == None:
            return
        logging.info(f"Saving {self.optim_checkpoint}")
        # save asi optimizer and scheduler dicts
        torch.save({"optimizer": self.optim[0].state_dict(), "scheduler": self.optim[1].state_dict()}, self.optim_checkpoint)

    def get_optimizer(self, iter, device="cuda", learning_rate=0.001):

        # remove last optim checkpoint
        if iter == 0 and os.path.isfile(self.optim_checkpoint):
            logging.info("removing last optim_scheduler checkpoint")
            os.remove(self.optim_checkpoint)

        _optimizer = torch.optim.Adam
        _options = {'lr': learning_rate}

        param_list = []
        param_list.append({'params': self.sequence_network.parameters(),
                           'weight_decay': 0.00002})
        param_list.append({'params': self.stat_pooling.parameters(),
                           'weight_decay': 0.00002})
        param_list.append({'params': self.before_speaker_embedding.parameters(),
                           'weight_decay': 0.00002})
        param_list.append({'params': self.after_speaker_embedding.parameters(),
                           'weight_decay': 0.000})
        optimizer = _optimizer(param_list, **_options)

        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer, # TODO: might not be the best scheduler
                                                      base_lr=1.0e-05,
                                                      max_lr=learning_rate,
                                                      step_size_up=2500,
                                                      step_size_down=None,
                                                      cycle_momentum=False,
                                                      mode="triangular2"
                                                      )


        if os.path.isfile(self.optim_checkpoint):
            logging.info(f"Loading {self.optim_checkpoint}")
            checkpoint = torch.load(self.optim_checkpoint, map_location=device, weights_only=False)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])

        return optimizer, scheduler


    def forward(self, x, target=None, norm_embedding=True):
        x = self.sequence_network(x)
        x = self.stat_pooling(x)
        x = self.before_speaker_embedding(x)
        if norm_embedding:
            xvec = F.normalize(x, dim=1)

        speaker_loss, s_layer = self.after_speaker_embedding(xvec, target=target)
        return (speaker_loss, s_layer), xvec



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

            # use additional_obj to do the backward, not done by the trainer
            self.trainer_do_backward = False

            # Adversarial network
            self.acc_sum_asi_loss = torch.tensor(0.0, requires_grad=False) # for logging
            self.acc_sum_asi_accuracy = 0.0 # for logging
            spk2id_lines = [
                line.rstrip("\n").split(" ")
                for line in open(args.spk2id)
            ]
            self.spk2id = dict(map(lambda x: (x[0], x[1]), spk2id_lines))
            self.asi = XVector(asr_out_dim=prefinal_bottleneck_dim, number_of_spk=len(self.spk2id))

            if args.adversarial_training == "True":
                self.asi = revgrad(self.asi)

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


        def after_one_iter_hook(self):
            self.asi.save_optimizer()


        def init_custom_load(self, init_weight_provided):
            """
            Filter the state dict dictionary
            """
            logging.info("Filtering keys with init_custom_load")

            if not args.adversarial_training == "True":
                return init_weight_provided

            # From a sidekit model (sidekit training loop)
            state_dict = init_weight_provided["model_state_dict"]
            state_dict = {k.replace("external_model.model.",""): v for k, v in state_dict.items()}
            return state_dict


        @torch.no_grad()
        def validate_model(self):
            self.eval()
            N = 2
            C = (10 * self.frame_subsampling_factor) * 274
            x = torch.arange(N * C).reshape(N, C).float()
            nnet_output, xent_output = self.forward(x)
            assert (
                nnet_output.shape[1] == 17
            ), f"{nnet_output.shape[1]} != expected frame subsampling"
            self.train()

            #  self.asi(self.bottleneck_out.permute(0, 2, 1).contiguous())

        def pad_input(self, x):
            if self.padding > 0:
                N, T, C = x.shape
                left_pad = x[:, 0:1, :].repeat(1, self.padding, 1).reshape(N, -1, C)
                right_pad = x[:, -1, :].repeat(1, self.padding, 1).reshape(N, -1, C)
                x = torch.cat([left_pad, x, right_pad], axis=1)
            return x


        def set_lr_layers_for_optim(
            self, get_optimizer, lr, weight_decay, iter=0, total_iter=-1
        ):
            asr = []
            asi = []

            switch_require_grad_enc = not args.freeze_encoder == "True"
            for name, param in self.named_parameters():
                if "asi" in name:
                    asi.append(param)
                else:
                    # freeze encoder (up to 'prefinal_chain_vq.tdnn.linearB.bias')
                    if name == "prefinal_chain_vq.tdnn.linearA.weight":
                        switch_require_grad_enc = True
                    param.requires_grad = switch_require_grad_enc
                    asr.append(param)
                    logging.info(name + f" - requires_grad={param.requires_grad}")

            if args.adversarial_training == "True":
                #  apply residual (after having a well trained asi)
                # Shared training of the asi network (same training optimizer and LR)
                opti = get_optimizer(
                    [{"params": asr}, {"params": asi}], lr, weight_decay
                )
            else:
                # train only the asr
                opti = get_optimizer(
                    [{"params": asr}], lr, weight_decay
                )

            # Set the optimizer for the ASI network
            self.asi.optim = self.asi.get_optimizer(iter)

            return opti

        def additional_obj(
            self,
            deriv,
            data_metadata,
            should_log=False,
            print_interval=1,
            tensorboard=None,
            mb_id=1,
            for_valid=False,
        ):
            speech, metadata = data_metadata[0], data_metadata[1]
            # fmt: off
            if deriv != None and self.asi.optim != None:

                # Not shared training of the asi network (own training optimizer and LR)
                self.asi.optim[0].zero_grad() if self.training else None

                label = torch.zeros(self.bottleneck_out.shape[0], dtype=torch.long)
                label = label.to(self.bottleneck_out.device)
                for i, m in enumerate(metadata):
                    label[i] = int(self.spk2id[m.name.split("-")[2]])
                (speaker_loss, cce_prediction), xvec  = self.asi(self.bottleneck_out.permute(0, 2, 1).contiguous(),
                                                          target=label)

                # Accumulate another loss
                if args.adversarial_training == "True":
                    deriv += speaker_loss.to(deriv.device)
                    speaker_loss = deriv

                if self.training:
                    speaker_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.asi.parameters(), 1.)
                    self.asi.optim[0].step() # optim
                    self.asi.optim[1].step() # scheduler

                # Display validation info
                if for_valid and should_log:
                    logging.info("ASI objf={}".format(self.acc_sum_asi_loss / print_interval))
                    logging.info("ASI accuracy={}".format(self.acc_sum_asi_accuracy / print_interval))
                    if tensorboard:
                        tensorboard.add_scalar("ASI/valid_loss", self.acc_sum_asi_loss / print_interval, mb_id)
                        tensorboard.add_scalar("ASI/valid_accuracy", self.acc_sum_asi_accuracy / print_interval, mb_id)
                    self.acc_sum_asi_loss.zero_()
                    self.acc_sum_asi_accuracy = 0.0
                    return

                # Collect validation info
                if for_valid:
                    self.acc_sum_asi_loss.add_(speaker_loss.item() * deriv)  # deriv here is the mini_batchsize*num_seq
                    self.acc_sum_asi_accuracy += ((torch.argmax(cce_prediction.data, 1) == label).sum()).cpu() * deriv
                    return

                # stats
                self.acc_sum_asi_loss.add_(speaker_loss.item())
                accuracy = ((torch.argmax(cce_prediction.data, 1) == label).sum()).cpu()
                self.acc_sum_asi_accuracy += accuracy
                logging.debug("ASI accuracy={}".format(accuracy))

                # Logs stats during training
                if should_log:
                    logging.info("Overall ASI objf={}".format(self.acc_sum_asi_loss / print_interval))
                    logging.info("Overall ASI accuracy={}".format(self.acc_sum_asi_accuracy / print_interval))
                    if tensorboard:
                        tensorboard.add_scalar("ASI/train_loss", self.acc_sum_asi_loss / print_interval, mb_id)
                        tensorboard.add_scalar("ASI/train_accuracy", self.acc_sum_asi_accuracy / print_interval, mb_id)
                    self.acc_sum_asi_loss.zero_()
                    self.acc_sum_asi_accuracy = 0.0
            # fmt: on

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

            chain_prefinal_out = self.prefinal_chain_vq(x)
            xent_prefinal_out = self.prefinal_xent(x)

            chain_out = self.chain_output(chain_prefinal_out)
            xent_out = self.xent_output(xent_prefinal_out)
            return chain_out, F.log_softmax(xent_out, dim=2)

    return Net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model config args")
    parser.add_argument("--freeze-encoder", default="False", type=str)
    parser.add_argument("--adversarial-training", default="False", type=str)
    parser.add_argument("--spk2id", default="./data/spk2id", type=str)
    args, remaining_argv = parser.parse_known_args()
    sys.argv = sys.argv[:1] + remaining_argv

    if os.environ.get("TESTING", "0") == "1":
        args.adversarial_training = "True"
        model = build(args)(output_dim=3280).cuda()

        model.load_state_dict(model.init_custom_load(torch.load("./exp/spk/adv/tmp_model_custom.pt", weights_only=False)))

        import torchaudio
        x,_ = torchaudio.load("/lium/raid01_b/pchampi/lab/Voice-Privacy-Challenge-2022/baseline/corpora/LibriSpeech/train-clean-100/196/122150/196-122150-0032.flac")
        x,_ = torchaudio.load("/lium/raid01_b/pchampi/lab/Voice-Privacy-Challenge-2022/baseline/corpora/LibriSpeech/train-clean-100/3436/172171/3436-172171-0035.flac")
        nnet_output, xent_output = model.forward(x.cuda())
        (_, cce_prediction), xvec  = model.asi(model.bottleneck_out.permute(0, 2, 1).contiguous())
        print("cce_prediction",torch.argmax(cce_prediction.data, 1))

        import satools.infer_helper as infer_helper

        text = infer_helper.kaldi_asr_decode(nnet_output)  # is this even text ?
        print("Text:", text)

        sys.exit(0)

    ChainE2EModel(build(args), cmd_line=True)
