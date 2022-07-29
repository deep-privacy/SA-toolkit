#!/usr/bin/env python3

import torch
import satools

t = satools.matrix.ReadKaldiMatrixFile("lfmmi_deriv_0")
print(t[0, :])
