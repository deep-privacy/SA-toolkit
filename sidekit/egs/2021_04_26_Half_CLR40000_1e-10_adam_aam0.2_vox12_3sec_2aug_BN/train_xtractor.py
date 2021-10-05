#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import torch
import sidekit
import yaml

from argparse import ArgumentParser

def main():
    parser = ArgumentParser('DDP usage example')
    parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.')
    args = parser.parse_args()

    # keep track of whether the current process is the `master` process (totally optional, but I find it useful for data laoding, logging, etc.)
    args.is_master = args.local_rank == 0

    sidekit.nnet.xvector.xtrain(dataset_description="cfg/VoxCeleb12_ssd.yaml",
                                model_description="cfg/model.yaml",
                                training_description="cfg/training.yaml")


if __name__ == '__main__':
    main()

