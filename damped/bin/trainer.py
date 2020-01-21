#!/usr/bin/env python3

import torch
import damped
from damped import utils
from damped.disturb import const
from sklearn.metrics import accuracy_score

import configargparse
import importlib.util
import os
import time


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
        nargs="?",
        required=False,
        default=10,
    )
    parser.add(
        "--exp-path",
        dest="exp_path",
        help="Path to save the exp model/results",
        required=True,
        type=str,
    )
    parser.add(
        "--task-rank",
        dest="task_rank",
        type=int,
        help="The rank of this task (torch.distributed)",
        required=True,
    )
    parser.add(
        "--n-checkpoint",
        dest="n_checkpoints",
        type=int,
        help="The number of checkpoints to keep (checkpoint frequency defined by log-interval)",
        nargs="?",
        required=False,
        default=3,
    )
    parser.add(
        "--gpu-device",
        dest="gpu_device",
        help="If the node has GPU accelerator, select the GPU to use",
        nargs="?",
        required=False,
        type=int,
        default=0,
    )
    parser.add(
        "--resume",
        dest="resume",
        help="Resume the training from a checkpoint",
        default="",
        nargs="?",
        required=False,
        type=str,
    )
    parser.add(
        "--world-size",
        dest="world_size",
        help="The number of expected TOTAL domain task (might be more than one)",
        required=True,
        type=int,
    )
    parser.add(
        "--master-ip",
        dest="master_ip",
        help="The ipv4 or ipv6 address of the master node. (The one that was damped.disturb-ed)",
        required=True,
        type=str,
    )

    return parser


def main():
    """Run the main training function."""
    parser = get_parser()
    args, _ = parser.parse_known_args()

    device = torch.device(
        f"cuda:{args.gpu_device}" if torch.cuda.is_available() else "cpu"
    )

    # load the conf
    spec = importlib.util.spec_from_file_location("config", args.config)
    config = importlib.util.module_from_spec(spec)
    config.argsparser = (
        parser  # Share the configargparse.ArgumentParser with the user defined module
    )
    spec.loader.exec_module(config)

    # create the net and training optim/criterion
    optimizer = config.optimizer
    net = config.net.to(device)
    criterion = config.criterion

    # keep track of some values while training
    total_correct = 0
    total_target = 0

    monitor = utils.Monitor(
        save_path=os.path.join("exp/", args.exp_path),
        exp_id=net.__class__.__name__,
        model=net,
        eval_metrics="acc, loss",  # First metric is considered to be early-stopping metric
        save_best_metrics=True,
        n_checkpoints=args.n_checkpoints,
    )
    monitor.set_optimizer(optimizer)
    monitor.save_model_summary()

    if args.resume:
        print("resumed from %s" % args.resume, flush=True)
        # load last checkpoint
        monitor.load_checkpoint(args.resume)

    # Eval related
    eval_mode = False
    total_labels = torch.LongTensor([])
    total_pred = torch.LongTensor([])
    loss_batches = 0
    loss_batches_count = 0

    # indicate if damped.disturb-ed toolkit wants the gradient form the DomainTask
    send_backward_grad = False

    # init the rank of this task
    utils.init_distributedenv(
        rank=args.task_rank, world_size=args.world_size, ip=args.master_ip
    )

    print("Training started on %s" % time.strftime("%d-%m-%Y %H:%M"), flush=True)

    # TODO(pchampio) refactor this training loop into sub-functions
    while True:
        features, y_mapper, is_meta_data = utils.fork_recv(
            rank=0, dtype=(torch.float32, torch.long)
        )

        if is_meta_data:
            meta_data = y_mapper

            if const.should_stop(meta_data):
                break

            if const.is_no_wait_backward(meta_data):
                print("Switch to NOT sending backward gradient", flush=True)
                send_backward_grad = False

            if const.is_wait_backward(meta_data):
                print("Switch to sending backward gradient", flush=True)
                send_backward_grad = True

            last_eval = eval_mode
            eval_mode = const.is_eval(meta_data)

            # detect changes from train to eval
            if eval_mode and not last_eval:
                print("Running evaluation on dev..", flush=True)
                net.eval()

            # detect changes from eval to train
            if not eval_mode and last_eval:
                net.train()
                # display validation metics
                accuracy = (
                    accuracy_score(
                        total_labels.flatten().numpy(), total_pred.flatten().numpy()
                    )
                    * 100
                )

                loss = 0
                if loss_batches_count != 0:
                    loss = loss_batches / loss_batches_count

                monitor.update_scores(
                    [
                        utils.Metric("acc", accuracy),
                        utils.Metric("loss", loss, higher_better=False,),
                    ]
                )
                monitor.save_models()
                monitor.vctr += 1

                # clear for next eval
                total_labels = torch.LongTensor([])
                total_pred = torch.LongTensor([])
                loss_batches = 0
                loss_batches_count = 0

            # When meta_data is shared, no features/label are sent
            continue

        target = config.mapper(y_mapper)

        input = features.to(device)
        input.requires_grad = True

        # Eval
        if eval_mode:
            y_pred = net(input)

            # send back the gradient if needed
            if send_backward_grad:
                 # backward will not be applied (eval), send 0 grad
                damped.disturb.DomainTask._isend(0, torch.zeros(*input.size())).wait()

            _, predicted = torch.max(y_pred.data, dim=1)

            total_labels = torch.cat((total_labels, target.cpu()))
            total_pred = torch.cat((total_pred, predicted.cpu()))

            loss = criterion(y_pred, target.to(device))

            loss_batches += loss.cpu().detach().numpy()
            loss_batches_count += 1
            continue

        optimizer.zero_grad()
        y_pred = net(input)

        if torch.any(torch.isnan(y_pred)):
            print(features)
            print("ERROR: ignoring this batch, prediction is NaN")
            continue

        loss = criterion(y_pred, target.to(device))
        loss.backward()
        # send back the gradient if needed
        req = None
        if send_backward_grad:
            req = damped.disturb.DomainTask._isend(0, input.grad.data.cpu())
        optimizer.step()

        correct = (torch.argmax(y_pred.data, 1) == target.to(device)).sum().item()
        total_correct += correct
        total_target += target.size(0)

        monitor.train_loss.append(loss.item())

        # wait back the gradient if needed
        if send_backward_grad:
            req.wait()

        monitor.uctr += 1
        if monitor.uctr % args.log_interval == 0:
            accuracy = (total_correct / total_target) * 100
            print(
                "Train batch [{}]\tLoss: {:.6f}\tTrain Accuracy: {:.3f}".format(
                    monitor.uctr, loss.item(), accuracy,
                ),
                flush=True,
            )
            total_correct = 0
            total_target = 0

            monitor.save_checkpoint()

    print("Training finished on %s" % time.strftime("%d-%m-%Y %H:%M"))


if __name__ == "__main__":
    main()
