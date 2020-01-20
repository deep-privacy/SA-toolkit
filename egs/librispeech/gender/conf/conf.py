#!/usr/bin/env python

import os
from damped.utils import gender_mapper
from damped.nets import BrijSpeakerXvector
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
argsparser.add("--hidden-units", default=512, type=int)  # noqa
argsparser.add("--rnn-layers", default=3, type=int)  # noqa
args = argsparser.parse_args()  # noqa


# input: Batch x Tmax X D
# Assuming 1024 dim per frame (T) (encoder projection)
class GenderNet(nn.Module):
    """ Gender classification net
    """

    def __init__(self):
        super(GenderNet, self).__init__()
        self.eproj = args.eproj
        self.hidden_size = args.hidden_units
        self.num_layers = args.rnn_layers

        self.lstm = nn.LSTM(
            self.eproj, self.hidden_size, self.num_layers, batch_first=True,
            dropout=args.dropout,
            bidirectional=False,
        )
        self.fc1 = nn.Linear(self.hidden_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2)

        self.dropout = nn.Dropout(p=args.dropout)
        self.activation = nn.ReLU()

    def zero_state(self, hs_pad):
        return hs_pad.new_zeros(self.num_layers, hs_pad.size(0), self.hidden_size)

    def forward(self, hs_pad):
        """Gender branch forward
        Args:
            hs_pad (torch.Tensor): batch of padded hidden state sequences (B, Tmax, D)
        """
        h_0 = self.zero_state(hs_pad)
        c_0 = self.zero_state(hs_pad)

        self.lstm.flatten_parameters()  # Memory: compact weights

        print(hs_pad.shape)
        out_lstm, (h_n, c_n) = self.lstm(hs_pad, (h_0, c_0))
        print(out_lstm.shape)

        out_fc1 = self.dropout(self.activation(self.fc1(out_lstm)))
        out_fc2 = self.activation(self.fc2(out_fc1))
        out_fc3 = self.fc3(out_fc2)
        return out_fc3


#  net = GenderNet()
net = BrijSpeakerXvector(2, args.eproj, args.hidden_units, args.rnn_layers, args.dropout)


#  Binary Cross Entropy
criterion = nn.CrossEntropyLoss()

# Optim
optimizer = torch.optim.Adam(net.parameters())

# mapper used for ../data/spk2gender
dir_path = os.path.dirname(os.path.realpath(__file__))
mapper = gender_mapper(dir_path)
