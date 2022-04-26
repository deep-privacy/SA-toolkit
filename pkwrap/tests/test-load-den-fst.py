#!/usr/bin/env python3

import torch
import pkwrap

pkwrap.kaldi.chain.LoadDenominatorGraph("tests/data/den.fst", 3592)
