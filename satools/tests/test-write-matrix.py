#!/usr/bin/env python3

import torch
import satools

x = torch.randn(10, 10)

writer_spec = "ark,t:test"
writer = satools.script_utils.feat_writer(writer_spec)
writer.Write("test", satools.kaldi.matrix.TensorToKaldiMatrix(x))
