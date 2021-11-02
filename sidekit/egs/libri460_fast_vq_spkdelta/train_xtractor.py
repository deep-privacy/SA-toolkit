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
    parser.add_argument('--dataset', type=str, default="cfg/dataset.yaml")
    parser.add_argument('--model', type=str, default="cfg/model.yaml")
    parser.add_argument('--training', type=str, default="cfg/training.yaml")
    args = parser.parse_args()
    print(args)

    # keep track of whether the current process is the `master` process (totally optional, but I find it useful for data laoding, logging, etc.)
    args.is_master = args.local_rank == 0

    sidekit.nnet.xvector.xtrain(dataset_description=args.dataset,
                                model_description=args.model,
                                training_description=args.training)


if __name__ == '__main__':
    main()

