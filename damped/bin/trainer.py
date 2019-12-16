#!/usr/bin/env python

import torch
from damped import utils

import configargparse
import importlib.util
import sys


def get_parser(parser=None):
    """Get default arguments."""
    if parser is None:
        parser = configargparse.ArgumentParser(description="Train an domain branch")

    # general configuration
    parser.add(
        "--config", dest="config", type=str, help="config file path", required=True
    )

    return parser


def main(cmd_args):
    """Run the main training function."""
    parser = get_parser()
    args = parser.parse_args()

    # load the conf (net & criterion)
    spec = importlib.util.spec_from_file_location("config", args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    optimizer = config.optimizer
    net = config.net
    criterion = config.criterion

    while True:
        features, y_mapper = utils.fork_recv(rank=0, dtype=(torch.float32, torch.long))

        label = config.mapper(y_mapper)
        y_pred = net(features)

        if torch.any(torch.isnan(y_pred)):
            print(features)
            print("ERROR: ignoring this batch, prediction is NaN")
            continue

        loss = criterion(y_pred, label)
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    utils.init_distributedenv(rank=1)
    main(sys.argv[1:])
