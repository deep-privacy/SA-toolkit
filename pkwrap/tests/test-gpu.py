#!/usr/bin/env python3

import torch
import pkwrap

x = torch.zeros(3, 3).uniform_()
x = x.cuda()
pkwrap.kaldi.InstantiateKaldiCuda()
x_kaldi = pkwrap.kaldi.matrix.TensorToKaldiCuSubMatrix(x)
print("CuSubMatrix created")
