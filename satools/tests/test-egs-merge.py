#!/usr/bin/env python3

import torch
import satools

egs_name = "ark:tests/data/cegs.ark"
egs = satools.kaldi.chain.ReadChainEgsFile(egs_name, 0)
satools.kaldi.chain.ShuffleEgs(egs)
egs = satools.kaldi.chain.MergeChainEgs(egs, "8")
sup = satools.kaldi.chain.GetSupervisionFromEgs(egs[0])
