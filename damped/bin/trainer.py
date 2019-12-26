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

    accuracy = 0.0
    while True:
        features, y_mapper = utils.fork_recv(rank=0, dtype=(torch.float32, torch.long))

        target = config.mapper(y_mapper)

        optimizer.zero_grad()
        y_pred = net(features.to(device))

        if torch.any(torch.isnan(y_pred)):
            print(features)
            print("ERROR: ignoring this batch, prediction is NaN")
            continue

        loss = criterion(y_pred, target.to(device))
        loss.backward()
        optimizer.step()

        accuracy += (torch.argmax(y_pred.data, 1) == target.to(device)).sum()
        epoch = 1
        batch_idx = 1
        print(
            "Train Epoch: {} \tLoss: {:.6f}\tAccuracy: {:.3f}".format(
                epoch, batch_idx + 1, loss.item(), 100.0 * accuracy.item(),
            )
        )


if __name__ == "__main__":
    utils.init_distributedenv(rank=1)
    main()
