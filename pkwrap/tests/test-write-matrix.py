#!/usr/bin/env python3

import torch
import pkwrap

x = torch.randn(10, 10)

writer_spec = "ark,t:test"
writer = pkwrap.script_utils.feat_writer(writer_spec)
writer.Write("test", pkwrap.kaldi.matrix.TensorToKaldiMatrix(x))
