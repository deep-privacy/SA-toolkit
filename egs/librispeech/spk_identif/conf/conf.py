#!/usr/bin/env python

import os
from damped.utils import spkid_mapper
from damped.nets import BrijSpeakerXvector, grad_reverse
import torch
import torch.nn as nn

# This file will be used to define the domain branch.

# The folowing variable MUST be defined, it will be imported by trainer.py!
# - `net` for the network architecture
# - `criterion` for the loss
# - `optimizer`
# - `mapper` to map the distirb y label to the domain task y label

# parse args injected by damped
argsparser.add("--spk-number", default=251, type=int)  # noqa
argsparser.add("--eproj", default=1024, type=int)  # noqa
argsparser.add("--hidden-units", default=512, type=int)  # noqa
argsparser.add("--rnn-layers", default=3, type=int)  # noqa
argsparser.add("--dropout", default=0.2, type=float)  # noqa
argsparser.add("--grad-reverse", default=False, type=bool)  # noqa
args = argsparser.parse_args()  # noqa

# input: Batch x Tmax X D
# Assuming 1024 dim per frame (T) (encoder projection)
net = BrijSpeakerXvector(args.spk_number, args.eproj, args.hidden_units, args.rnn_layers, args.dropout)

if args.grad_reverse:
    class BrijSpeakerXvectorGradRev(BrijSpeakerXvector):
        def __init__(self, odim, eprojs, hidden_size, rnn_layers, dropout_rate=0.2):
            super().__init__(odim, eprojs, hidden_size, rnn_layers, dropout_rate=0.2)
            self.scale = 0.1
            print("Gradient reversed!")

        def forward(self, hs_pad):
            x = grad_reverse(hs_pad, scale=self.scale)
            return super().forward(x)

    net = BrijSpeakerXvectorGradRev(args.spk_number, args.eproj, args.hidden_units, args.rnn_layers, args.dropout)

#  Binary Cross Entropy
criterion = nn.CrossEntropyLoss()

# Optim
optimizer = torch.optim.Adam(net.parameters())

# mapper used for ../data/spk2id
dir_path = os.path.dirname(os.path.realpath(__file__))
mapper = spkid_mapper(dir_path)
