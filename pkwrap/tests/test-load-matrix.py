#!/usr/bin/env python3

import torch
import pkwrap

t = pkwrap.matrix.ReadKaldiMatrixFile("lfmmi_deriv_0")
print(t[0, :])
