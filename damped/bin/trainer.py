#!/usr/bin/env python

import torch
from damped import utils

import configargparse
import importlib.util

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def get_parser(parser=None):
    """Get default arguments."""
    if parser is None:
        parser = configargparse.ArgumentParser(description="Train an domain branch")

    # general configuration
    parser.add(
        "--config", dest="config", type=str, help="config file path", required=True
    )

    return parser


def main():
    """Run the main training function."""
    parser = get_parser()
    args = parser.parse_args()

    # load the conf
    spec = importlib.util.spec_from_file_location("config", args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    optimizer = config.optimizer
    net = config.net.to(device)
    criterion = config.criterion

    while True:
        features, y_mapper = utils.fork_recv(rank=0, dtype=(torch.float32, torch.long))

        label = config.mapper(y_mapper)

        optimizer.zero_grad()
        y_pred = net(features.to(device))
        loss = criterion(y_pred, label.to(device))
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    utils.init_distributedenv(rank=1)
    main()
