#!/usr/bin/env python3

import json
import logging
import sys


import configargparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from satools import sidekit

logging.basicConfig(level=logging.INFO)
logging.getLogger("geocoder").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def build(args):
    class Net(nn.Module):
        #  def init(self):
            #  logging.info("Init epoch 0")

        def __init__(self, num_speakers):
            super().__init__()
            self.preprocessor = sidekit.preprocessor.MelSpecFrontEnd(n_fft=1024,
                                                win_length=400,
                                                hop_length=160,
                                                n_mels=80)
            self.sequence_network = sidekit.archi.PreHalfResNet34()

            self.embedding_size = 256

            self.before_speaker_embedding = nn.Sequential(OrderedDict([
                ("lin_be", nn.Linear(in_features = 5120, out_features = self.embedding_size, bias=False)),
                ("bn_be", nn.BatchNorm1d(self.embedding_size))
            ]))

            self.stat_pooling = sidekit.pooling.AttentivePooling(256, 10, global_context=True)

            self.after_speaker_embedding = sidekit.loss.ArcMarginProduct(
                self.embedding_size,
                num_speakers,
                s=30,
                m=0.2,
                easy_margin=False)

            self.preprocessor_weight_decay = 0.00002
            self.sequence_network_weight_decay = 0.00002
            self.stat_pooling_weight_decay = 0.00002
            self.before_speaker_embedding_weight_decay = 0.00002
            self.after_speaker_embedding_weight_decay = 0.000


        def forward(self, x, target=None):
            """
            The forward mothod MUST return 2 values:
               - a tuple of: (loss: to train the model, or in testing (target==None) you should return torch.tensor(float('nan')).
                               cross-entroy prediction: raw output of the network to compute accuracy)
                             in this example the returned value handled by: ArcMarginProduct
               - the x-vector embedding
               (loss, cce), x_vector = model([...])
            """

            x = self.preprocessor(x)
            x = self.sequence_network(x)
            x = self.stat_pooling(x)

            x = self.before_speaker_embedding(x)
            x_vector = F.normalize(x, dim=1)

            speaker_loss, s_layer = self.after_speaker_embedding(x, target=target)
            return (speaker_loss, s_layer), x_vector

        
        def set_lr_weight_decay_layers_for_optim(self, _optimizer, _options):
            logging.info("Set lr and weight decay")
            self._optimizer_option = _options
            self._optimizer = _optimizer

            # fmt: off
            param_list = []
            param_list.append({"params": self.preprocessor.parameters(), "weight_decay": self.preprocessor_weight_decay})
            param_list.append({"params": self.sequence_network.parameters(), "weight_decay": self.sequence_network_weight_decay})
            param_list.append({ "params": self.stat_pooling.parameters(), "weight_decay": self.stat_pooling_weight_decay})
            param_list.append({ "params": self.before_speaker_embedding.parameters(), "weight_decay": self.before_speaker_embedding_weight_decay})
            param_list.append({ "params": self.after_speaker_embedding.parameters(), "weight_decay": self.after_speaker_embedding_weight_decay})
            # fmt: on

            self.optimizer = _optimizer(param_list, **_options)

            # example on applying different LR to different layers
            #  self.optimizer.param_groups[0]["lr"] = _options["lr"] / 2

            return self.optimizer

    return Net

if __name__ == "__main__":
    parser = configargparse.ArgumentParser(description="Model config args")
    args, remaining_argv = parser.parse_known_args()
    sys.argv = sys.argv[:1] + remaining_argv + ["--base-model-args", json.dumps(vars(args))]
    sidekit.SidekitModel(build(args), cmd_line=True)

