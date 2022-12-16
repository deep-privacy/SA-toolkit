#!/usr/bin/env python3

import os
import warnings

import sidekit

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

os.environ["SIDEKIT"] = "libsvm=false"

sidekit.nnet.xtrain(
    "local/spk/config/Librispeech.yaml",
    "local/spk/config/model_adv.yaml",
    "local/spk/config/training_adv.yaml",
)
