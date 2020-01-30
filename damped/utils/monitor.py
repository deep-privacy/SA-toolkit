#!/usr/bin/env python
from collections import defaultdict
from collections import deque
from functools import total_ordering
import os
from pathlib import Path

import torch
from tensorboardX import SummaryWriter


class Monitor:
    """Class that tracks training progress. The following informations are
    kept as object attributes:
        self.uctr:       # of updates, i.e. mini-batches done so far
        self.vctr:       # of evaluations done on val_set so far
        self.train_loss: List of training losses
        self.val_scores: Dict of lists keeping tracking of validation metrics
    """

    # Variables to save
    VARS = ["uctr", "vctr", "train_loss", "val_scores"]

    def __init__(
        self,
        tensorboard_dir,
        save_path,
        exp_id,
        model,
        eval_metrics,
        history=None,
        save_best_metrics=False,
        n_checkpoints=3,
    ):
        self.tensorboard_writter = SummaryWriter(tensorboard_dir)
        self.save_path = save_path
        self.exp_id = exp_id
        self.model = model
        self.eval_metrics = eval_metrics.upper().replace(" ", "").split(",")
        self.save_best_metrics = save_best_metrics
        self.optimizer = None
        self.checkpoints = FileRotator(n_checkpoints)

        if history is None:
            history = {}

        self.uctr = history.pop("uctr", 0)
        self.vctr = history.pop("vctr", 1)
        self.train_loss = history.pop("train_loss", [])
        self.val_scores = history.pop("val_scores", defaultdict(list))

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        if len(self.eval_metrics) > 0:
            # To keep current best metric validation id and score
            self.cur_bests = {}

            # First metric is considered to be early-stopping metric
            self.early_metric = self.eval_metrics[0]

    @staticmethod
    def best_score(scores):
        """Returns the best validation id and score for that."""
        idx, score = sorted(
            enumerate(scores), key=lambda e: e[1], reverse=scores[0].higher_better
        )[0]
        return (idx + 1, score)

    def set_optimizer(self, optimizer):
        """Sets the optimizer to save its parameters."""
        self.optimizer = optimizer

    def state_dict(self):
        """Returns a dictionary of stateful variables."""
        return {k: getattr(self, k) for k in self.VARS}

    def save_checkpoint(self):
        """Saves a checkpoint by keeping track of file rotation."""
        self.checkpoints.push(self.save_model(suffix="update{}-val{}".format(self.uctr, self.vctr)))

    def save_model_summary(self):
        f = open(os.path.join(self.save_path, "model.summary"), "w")
        f.write(str(self.model))
        f.close()

    def save_model(self, metric=None, suffix="", do_symlink=False):
        """Saves a checkpoint with arbitrary suffix(es) appended."""
        # Construct file name
        fname = self.exp_id
        if metric:
            print("Saving best model based on {}".format(metric.name), flush=True)
            fname += "-val{:03d}.best.{}_{:.3f}".format(
                self.vctr, metric.name.lower(), metric.score
            )
        if suffix:
            fname += "-{}".format(suffix)
        fname = os.path.join(self.save_path, (fname + ".ckpt"))

        # Save the file
        model_dict = {
            "model": self.model.state_dict(),
            "history": self.state_dict(),
        }

        # Add optimizer states
        if self.optimizer is not None:
            model_dict["optimizer"] = self.optimizer.state_dict()

        torch.save(model_dict, fname)

        # Also create a symbolic link to the above checkpoint for the metric
        if metric and do_symlink:
            symlink = "{}.best.{}.ckpt".format(self.exp_id, metric.name.lower())
            symlink = Path(self.save_path) / Path(symlink)
            if symlink.exists():
                #  old_ckpt = symlink.resolve()
                symlink.unlink()
                #  old_ckpt.unlink()
            if symlink.is_symlink():
                symlink.unlink()
            symlink.symlink_to(fname)

        return fname

    def update_scores(self, results):
        """Updates score lists and current bests."""
        for metric in results:
            print("Validation {} -> {}".format(self.vctr, metric), flush=True)
            self.val_scores[metric.name].append(metric)
            self.cur_bests[metric.name] = self.best_score(self.val_scores[metric.name])

    def save_models(self):
        cur_bests = self.cur_bests.copy()

        # Let's start with early-stopping metric
        vctr, metric = cur_bests.pop(self.early_metric)
        if vctr == self.vctr:
            self.save_model(metric=metric, do_symlink=True)

        # If requested, save all best metric snapshots
        if self.save_best_metrics and cur_bests:
            for (vctr, metric) in cur_bests.values():
                if metric.name in self.eval_metrics and vctr == self.vctr:
                    self.save_model(metric=metric, do_symlink=True)

    def load_checkpoint(self, fname):
        data = self.load_pt_file(fname)
        self.model.load_state_dict(data["model"], strict=True)
        if data["optimizer"]:
            self.optimizer.load_state_dict(data["optimizer"])

        for k in self.VARS:
            if k in data["history"]:
                setattr(self, k, data["history"][k])

        self.vctr += 1

    @staticmethod
    def load_pt_file(fname):
        """Returns saved .(ck)pt file fields."""
        fname = str(Path(fname).expanduser())
        data = torch.load(fname)
        if "history" not in data:
            data["history"] = {}
        return data


class FileRotator:
    def __init__(self, maxlen):
        """A fixed queue with Path() elements where pushing a new element pops
        the oldest one and removes it from disk.

        Args:
            maxlen(int): The capacity of the queue.
        """
        self.maxlen = maxlen
        self.elems = deque(maxlen=self.maxlen)

    def push(self, elem):
        if len(self.elems) == self.maxlen:
            # Remove oldest item
            popped = self.elems.pop()
            if os.path.exists(popped):
                os.unlink(popped)

        # Add new item
        self.elems.appendleft(elem)

    def __repr__(self):
        return self.elems.__repr__()


@total_ordering
class Metric:
    """A Metric object to represent evaluation metrics.
    Args:
        name(str): A name for the metric that will be kept internally
            after upper-casing
        score(float): A floating point score
        detailed_score(str, optional): A custom, more detailed string
            representing the score given above (Default: "")
        higher_better(bool, optional): If ``False``, the smaller the better
            (Default: ``True``)
    """

    def __init__(self, name, score, detailed_score="", higher_better=True):
        self.name = name.upper()
        self.score = score
        self.detailed_score = detailed_score
        self.higher_better = higher_better

    def __eq__(self, other):
        return self.score == other.score

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        rhs = self.detailed_score if self.detailed_score else "%.2f" % self.score
        return self.name + " = " + rhs
