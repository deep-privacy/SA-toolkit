#!/usr/bin/env python

import os
from damped.utils import gender_mapper
from damped.nets import BrijSpeakerXvector, grad_reverse_net
import torch
import torch.nn as nn

# This file will be used to define the domain branch.

# The folowing variable MUST be defined, it will be imported by trainer.py!
# - `net` for the network architecture
# - `criterion` for the loss
# - `optimizer`
# - `mapper` to map the distirb y label to the domain task y label

# parse args injected by damped
argsparser.add("--eproj", default=1024, type=int)  # noqa
argsparser.add("--hidden-units", default=512, type=int)  # noqa
argsparser.add("--rnn-layers", default=3, type=int)  # noqa
argsparser.add("--dropout", default=0.2, type=float)  # noqa
argsparser.add("--grad-reverse", default=False, type=bool)  # noqa
args = argsparser.parse_args()  # noqa

Net = BrijSpeakerXvector
if args.grad_reverse:
    Net = grad_reverse_net(Net)

net = Net(2, args.eproj, args.hidden_units, args.rnn_layers, args.dropout)

#  Binary Cross Entropy
criterion = nn.CrossEntropyLoss()

# Optim
optimizer = torch.optim.Adam(net.parameters(), lr=0.0000025)
#  optimizer = torch.optim.Adam(net.parameters())

# mapper used for ../data/spk2gender
dir_path = os.path.dirname(os.path.realpath(__file__))
mapper = gender_mapper(dir_path)
