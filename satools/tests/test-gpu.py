#!/usr/bin/env python3

import torch
import satools

x = torch.zeros(3, 3).uniform_()
x = x.cuda()
satools.kaldi.InstantiateKaldiCuda()
x_kaldi = satools.kaldi.matrix.TensorToKaldiCuSubMatrix(x)
print("CuSubMatrix created")
