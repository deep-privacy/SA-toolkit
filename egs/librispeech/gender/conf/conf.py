#!/usr/bin/env python

import os
from damped import nets
import damped
import torch
import torch.nn as nn

dir_path = os.path.dirname(os.path.realpath(__file__))

# This file will be used to define the domain branch.

# The folowing variable MUST be defined, it will be imported by trainer.py!
# - `net` for the network architecture
# - `criterion` for the loss
# - `mapper` to map the distirb y label to the domain task y label
# - `optimizer`


# Assuming 1024 dim per frame
frame1 = nets.TDNN(input_dim=50, output_dim=512, context_size=5, dilation=1)
frame2 = nets.TDNN(input_dim=512, output_dim=512, context_size=3, dilation=2)
frame3 = nets.TDNN(input_dim=512, output_dim=512, context_size=3, dilation=3)
frame4 = nets.TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1)
frame5 = nets.TDNN(input_dim=512, output_dim=1500, context_size=1, dilation=1)
# Input to frame1 is of shape (batch_size, T, 24)
# Output of frame5 will be (batch_size, T-14, 1500)

net = nn.Sequential(
    frame1,
    frame2,
    frame3,
    frame4,
    frame5,
    nets.StatsPooling(),  # mean + std (out_dim = 2x in_dim = 3000)
    nets.DenseEmbedding(in_dim=3000, mid_dim=512, out_dim=512),
    nets.DenseReLU(512, 2),
    nn.Softmax(dim=1),
)

#  Binary Cross Entropy
criterion = nn.BCELoss(reduction="mean")


# Domain label
spk2gender_lines = [
    line.rstrip("\n").split(" ")
    for line in open(os.path.join(dir_path, "..", "data", "spk2gender"))
]
spk2gender = dict(map(lambda x: (x[0], x[1]), spk2gender_lines))
print("Config: spk2gender: ", spk2gender)


# sent y_mapper to y label
def mapper(y_mapper):
    decoded_y_mapped_label = list(
        map(lambda x: damped.utils.codec.StrIntEncoder.decode(x), y_mapper.tolist())
    )
    label = torch.zeros((len(y_mapper), 2))  # gender 'f' for female, 'm' for male
    for i, x in enumerate(decoded_y_mapped_label):
        indice = {"f": 0, "m": 1}
        label[i][indice[spk2gender[x]]] = 1
    return label


optimizer = torch.optim.Adam(net.parameters())
