#!/usr/bin/env python

import os
from damped.utils import gender_mapper
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
argsparser.add("--dropout", default=0.5, type=float)  # noqa
args = argsparser.parse_args()  # noqa


# input: Batch x Tmax X D
# Assuming 1024 dim per frame (T) (encoder projection)
class GenderNet(nn.Module):
    """ Gender classification net
    """

    def __init__(self):
        super(GenderNet, self).__init__()
        self.eproj = args.eproj
        self.hidden_size = 1024
        self.num_layers = 1

        self.lstm = nn.LSTM(
            self.eproj, self.hidden_size, self.num_layers, batch_first=True,
        )
        self.fc1 = nn.Linear(self.hidden_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 2)

        self.dropout = nn.Dropout(p=args.dropout)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

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

        out_lstm, (h_0, c_0) = self.lstm(hs_pad, (h_0, c_0))

        h_0 = h_0[0]  # Take the last layer of the LSTM

        out_fc1 = self.dropout(self.activation(self.fc1(h_0)))
        out_fc2 = self.dropout(self.activation(self.fc2(out_fc1)))
        out_fc3 = self.dropout(self.activation(self.fc3(out_fc2)))
        out_fc4 = self.fc4(out_fc3)

        out = self.softmax(out_fc4)

        return out


net = GenderNet()


#  Binary Cross Entropy
criterion = nn.CrossEntropyLoss()

# Optim
optimizer = torch.optim.Adam(net.parameters())

# mapper used for ../data/spk2gender
dir_path = os.path.dirname(os.path.realpath(__file__))
mapper = gender_mapper(dir_path)
