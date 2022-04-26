#!/usr/bin/env python3

import torch
import pkwrap

egs_name = "ark:tests/data/cegs.ark"
egs = pkwrap.kaldi.chain.ReadChainEgsFile(egs_name, 0)
pkwrap.kaldi.chain.ShuffleEgs(egs)
egs = pkwrap.kaldi.chain.MergeChainEgs(egs, "8")
sup = pkwrap.kaldi.chain.GetSupervisionFromEgs(egs[0])
