#!/usr/bin/env python3

import torch
import satools

satools.kaldi.chain.LoadDenominatorGraph("tests/data/den.fst", 3592)
