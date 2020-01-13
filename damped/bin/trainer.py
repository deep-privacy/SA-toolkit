#!/usr/bin/env python

import torch
from damped import utils
from damped.disturb import const
from sklearn.metrics import accuracy_score

import configargparse
import importlib.util
import os

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


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
        "--log-interval",
        dest="log_interval",
        type=int,
        help="Log training accuracy every X batch",
        required=False,
        default=10,
    )
    parser.add(
        "--task-rank",
        dest="task_rank",
        type=int,
        help="The rank of this task (torch.distributed)",
        required=True,
    )
    parser.add(
        "--label",
        dest="label",
        help="The label outputted by the net (0,1 in case of gender classification)",
        required=False,
        type=int,
        nargs="+",
    )
    parser.add(
        "--label-name",
        dest="label_name",
        help="The class name corresponding to the label of '--label'",
        required=False,
        type=str,
        nargs="+",
    )

    return parser


def main():
    """Run the main training function."""
    parser = get_parser()
    args = parser.parse_args()

    # init the rank of this task
    utils.init_distributedenv(rank=args.task_rank)

    # load the conf
    spec = importlib.util.spec_from_file_location("config", args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # create the net and training optim/criterion
    optimizer = config.optimizer
    net = config.net.to(device)
    criterion = config.criterion

    # keep track of some values while training
    total_correct = 0
    total_target = 0
    batch_idx = 0

    # Eval related
    eval_mode = False
    total_labels = torch.LongTensor([])
    total_pred = torch.LongTensor([])
    loss_batches = 0
    loss_batches_count = 0
    val_idx = 1

    while True:
        features, y_mapper, is_meta_data = utils.fork_recv(
            rank=0, dtype=(torch.float32, torch.long)
        )

        if is_meta_data:
            meta_data = y_mapper

            if const.should_stop(meta_data):
                break

            last_eval = eval_mode
            eval_mode = const.is_eval(meta_data)

            # detect changes from eval to train
            if not eval_mode and last_eval:
                if not args.label and not args.label_name:
                    print(
                        "Ignoring validation, no '--label-name' and '--label' parameters provided"
                    )
                else:
                    # display validation metics
                    print(
                        "Validation batch \tLoss: {:.6f}\tVal Accuracy: {:.3f}".format(
                            loss_batches / loss_batches_count,
                            (
                                accuracy_score(
                                    total_labels.flatten().numpy(),
                                    total_pred.flatten().numpy(),
                                )
                                * 100
                            ),
                        )
                    )
                    # clear for next eval
                    total_labels = torch.LongTensor([])
                    total_pred = torch.LongTensor([])
                    loss_batches = 0
                    loss_batches_count = 0

                    save_model(net, os.path.basename(args.config), val_idx)
                    val_idx += 1

            # When meta_data is shared, no features/label are sent
            continue

        target = config.mapper(y_mapper)

        # Eval
        if eval_mode:
            y_pred = net(features.to(device))

            _, predicted = torch.max(y_pred.data, dim=1)

            total_labels = torch.cat((total_labels, target.cpu()))
            total_pred = torch.cat((total_pred, predicted.cpu()))

            loss = criterion(y_pred, target.to(device))

            loss_batches += loss.cpu().detach().numpy()
            loss_batches_count += 1
            continue

        optimizer.zero_grad()
        y_pred = net(features.to(device))

        if torch.any(torch.isnan(y_pred)):
            print(features)
            print("ERROR: ignoring this batch, prediction is NaN")
            continue

        loss = criterion(y_pred, target.to(device))
        loss.backward()
        optimizer.step()

        correct = (torch.argmax(y_pred.data, 1) == target.to(device)).sum().item()
        total_correct += correct
        total_target += target.size(0)

        batch_idx += 1
        if batch_idx % args.log_interval == 0:
            print(
                "Train batch [{}]\tLoss: {:.6f}\tTrain Accuracy: {:.3f}".format(
                    batch_idx, loss.item(), (total_correct / total_target) * 100,
                )
            )
            total_correct = 0
            total_target = 0


if __name__ == "__main__":
    main()
