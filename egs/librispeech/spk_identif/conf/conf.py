#!/usr/bin/env python

import os
from damped.utils import spkid_mapper
from damped.nets import Xtractor
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
argsparser.add("--dropout", default=0.2, type=float)  # noqa
args = argsparser.parse_args()  # noqa

# mapper used for ../data/spk2id
dir_path = os.path.dirname(os.path.realpath(__file__))
mapper, spk_number = spkid_mapper(dir_path)

# input: Batch x Tmax X D
# Assuming 1024 dim per frame (T) (encoder projection)
net = Xtractor(spk_number, args.dropout, args.eproj)

#  Binary Cross Entropy
criterion = nn.CrossEntropyLoss()

# Optim
optimizer = torch.optim.Adam(net.parameters())
