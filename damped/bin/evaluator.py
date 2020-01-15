#!/usr/bin/env python

import torch
from damped import utils
from damped.disturb import const

import configargparse
import importlib.util
import os


def save_model(model: torch.nn.Module, name: str, epoch: int):
    exp_dir = os.path.join("exp/", name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    torch.save(model.state_dict(), os.path.join(exp_dir, f"snapshot.ep.{epoch}"))
    f = open(os.path.join("exp/", name, "model.summary"), "w")
    f.write(str(model))
    f.close()


def get_parser(parser=None):
    """Get default arguments."""
    if parser is None:
        parser = configargparse.ArgumentParser(description="Train an domain branch")

    parser.add(
        "--config", dest="config", type=str, help="config file path", required=True
    )
    parser.add(
        "--task-rank",
        dest="task_rank",
        type=int,
        help="The rank of this task (torch.distributed)",
        required=True,
    )
    parser.add(
        "--snapshot",
        dest="snapshot",
        type=str,
        help="The model parameters to use for eval",
        required=True,
    )
    parser.add(
        "--label",
        dest="label",
        help="The label outputted by the net (0,1 in case of gender classification)",
        required=True,
        type=int,
        nargs="+",
    )
    parser.add(
        "--label-name",
        dest="label_name",
        help="The class name corresponding to the label of '--label'",
        required=True,
        type=str,
        nargs="+",
    )

    parser.add(
        "--gpu-device",
        dest="gpu_device",
        help="If the node has GPU accelerator, select the GPU to use",
        required=False,
        type=int,
        default=0,
    )

    return parser


def main():
    """Run the main training function."""
    parser = get_parser()
    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.gpu_device}" if torch.cuda.is_available() else "cpu"
    )

    assert len(args.label) == len(
        args.label_name
    ), "The size of '--label' must be the same as the size of '--label-name'"

    # init the rank of this task
    utils.init_distributedenv(rank=args.task_rank)

    # load the conf
    spec = importlib.util.spec_from_file_location("config", args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # create the net
    net = config.net.to(device)

    # load the snapshot
    net.load_state_dict(torch.load(args.snapshot)["model"])

    net.eval()
    total_labels = torch.LongTensor([])
    total_pred = torch.LongTensor([])
    with torch.no_grad():
        while True:
            features, y_mapper, is_meta_data = utils.fork_recv(
                rank=0, dtype=(torch.float32, torch.long)
            )

            if is_meta_data:
                meta_data = y_mapper

                if const.should_stop(meta_data):
                    break

            target = config.mapper(y_mapper)
            y_pred = net(features.to(device))

            _, predicted = torch.max(y_pred.data, dim=1)

            total_labels = torch.cat((total_labels, target.cpu()))
            total_pred = torch.cat((total_pred, predicted.cpu()))

    utils.display_evaluation_result(args, total_labels, total_pred)


if __name__ == "__main__":
    main()
