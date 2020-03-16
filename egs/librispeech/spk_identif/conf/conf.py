#!/usr/bin/env python

import os
from damped.utils import spkid_mapper
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
argsparser.add("--spk-number", default=251, type=int)  # noqa
argsparser.add("--eproj", default=1024, type=int)  # noqa
argsparser.add("--hidden-units", default=512, type=int)  # noqa
argsparser.add("--rnn-layers", default=3, type=int)  # noqa
argsparser.add("--dropout", default=0.2, type=float)  # noqa
argsparser.add("--grad-reverse", default=False, type=bool)  # noqa
args, _ = argsparser.parse_known_args()


Net = BrijSpeakerXvector
if args.grad_reverse:
    Net = grad_reverse_net(Net)

net = Net(args.spk_number, args.eproj, args.hidden_units, args.rnn_layers, args.dropout)
#  net = Xtractor(args.spk_number, args.dropout, args.eproj)

#  Binary Cross Entropy
criterion = nn.CrossEntropyLoss()

# Optim
optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)
#  optimizer = torch.optim.Adam(net.parameters())

# mapper used for ../data/spk2id
dir_path = os.path.dirname(os.path.realpath(__file__))
mapper = spkid_mapper(dir_path)
