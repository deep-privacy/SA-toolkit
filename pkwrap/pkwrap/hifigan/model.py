from dataclasses import dataclass
import logging

import torch
import torch.nn.functional as F
from torch.distributed import init_process_group
from torch.utils.tensorboard import SummaryWriter
import os
from pathlib import Path
import glob
import copy
import time

from .. import utils
from .. import script_utils
from . import dataset

#  torch.set_num_threads(1)
#  torch.set_num_interop_threads(1)


@dataclass
class F0QuantModelOpts:
    mode: str = ""
    checkpoint_path: str = ""
    rank: int = 1
    lr: float = 0.0002
    minibatch_size: int = 32
    init_weight_model: str = ""
    adam_b1: float = 0.8
    adam_b2: float = 0.99
    lr_decay: float = 0.999
    segment_size: float = 16640
    training_epochs: int = 700
    checkpoint_interval: int = 1000  # in step (iter)
    quant_commit: float = 0.02
    f0_stats: str = "exp/f0_stats.pth"
    num_workers: int = 4
    train_utterances: list = "utt1.wav,utt2.wav"
    test_utterances: list = "utt1.wav,utt2.wav"

    def load_from_config(self, cfg):
        for key, value in cfg.items():
            if hasattr(self, key):
                type_of_value = type(getattr(self, key))
                setattr(self, key, type_of_value(value))
        return self


class F0QuantModel:
    def __init__(self, model_cls, **kwargs):
        self.opts = F0QuantModelOpts()
        self.opts.load_from_config(kwargs)

        self.Net = model_cls
        self.num_gpus = int(os.getenv("WORLD_SIZE", "1"))
        self.call_by_mode()

    def call_by_mode(self):
        if self.opts.mode in ["train", "training"]:
            self.train()

    def train(self):
        device = torch.device("cuda")
        if self.num_gpus > 1:
            device = torch.device("cuda:{:d}".format(self.opts.rank))
            logging.info(
                "Init from distributed training rank: {}".format(self.opts.rank)
            )
            init_process_group(
                backend="nccl",
                init_method="env://",
                rank=self.opts.rank,
            )

        generator = self.Net()

        if self.opts.init_weight_model == "last":
            pattern = os.path.join(self.opts.checkpoint_path, "g_" + "????????")
            cp_list = glob.glob(pattern)
            if len(cp_list) != 0:
                self.opts.init_weight_model = sorted(cp_list)[-1]
            else:
                self.opts.init_weight_model = ""
        if self.opts.init_weight_model != "":
            init_weight_provided = torch.load(self.opts.init_weight_model)
            init_weight_provided_matched, unmatch = utils.torch.match_state_dict(
                generator.state_dict(), init_weight_provided["generator"]
            )
            not_inited = generator.load_state_dict(
                init_weight_provided_matched, strict=False
            )
            logging.info(
                "Init from previous model {}, layers not initialized: {}: layers ommited (wrong shape): {}".format(
                    self.opts.init_weight_model,
                    str(not_inited),
                    str(unmatch.keys()),
                )
            )

        if self.opts.rank == 0:
            os.makedirs(self.opts.checkpoint_path, exist_ok=True)
            logging.info("checkpoints directory : " + str(self.opts.checkpoint_path))

        generator = generator.to(device)
        if self.num_gpus > 1:
            generator = torch.nn.parallel.DistributedDataParallel(
                generator, device_ids=[self.opts.rank]
            )
            copy.deepcopy(generator.module).validate_model(device=device)
        else:
            copy.deepcopy(generator).validate_model(device=device)

        optim_g = torch.optim.AdamW(
            generator.parameters(),
            self.opts.lr,
            betas=[self.opts.adam_b1, self.opts.adam_b2],
        )

        steps = 0
        best_val_err = 9999999
        last_epoch = -1
        if self.opts.init_weight_model != "":
            optim_g.load_state_dict(init_weight_provided["optim_g"])
            steps = init_weight_provided["steps"] + 1
            last_epoch = init_weight_provided["epoch"]
            best_val_err = init_weight_provided["best_val_err"]

        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optim_g, gamma=self.opts.lr_decay, last_epoch=last_epoch
        )

        def sample_interval(feats, lengths, filenames, f0s):
            acc_feats = []
            acc_f0s = []
            for batch_idx in range(feats.shape[0]):
                _feats, _f0s = dataset.sample_interval(
                    [feats[batch_idx], f0s[batch_idx]],
                    seq_len=self.opts.segment_size,
                    max_len=lengths[batch_idx].item(),
                )
                acc_feats.append(_feats)
                acc_f0s.append(_f0s)
            feats = torch.stack(acc_feats)
            f0s = torch.stack(acc_f0s)
            lengths = torch.zeros_like(lengths) + self.opts.segment_size
            return feats, lengths, filenames, f0s

        trainset = dataset.WavList(self.opts.train_utterances)

        train_sampler = (
            torch.utils.data.DistributedSampler(trainset) if self.num_gpus > 1 else None
        )

        dataloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.opts.minibatch_size,
            shuffle=False,
            num_workers=self.opts.num_workers,
            collate_fn=dataset.collate_fn_padd(
                self.opts.f0_stats, get_func=sample_interval
            ),
            sampler=train_sampler,
            persistent_workers=True,
        )

        if self.opts.rank == 0:
            dataloader_test = torch.utils.data.DataLoader(
                dataset.WavList(self.opts.test_utterances),
                batch_size=self.opts.minibatch_size,
                shuffle=False,
                num_workers=self.opts.num_workers,
                collate_fn=dataset.collate_fn_padd(self.opts.f0_stats),
                persistent_workers=True,
            )
            sw = SummaryWriter(os.path.join(self.opts.checkpoint_path, "logs"))

        if self.opts.rank == 0:
            logging.info(
                "Caching f0 (setting a higher 'num_workers' can speed-up this part)"
            )
            for i, batch in enumerate(dataloader):
                pass
            for i, batch in enumerate(dataloader_test):
                pass

        generator.train()
        logging.info("Starting training..")
        for epoch in range(max(0, last_epoch), self.opts.training_epochs):
            if self.opts.rank == 0:
                start = time.time()
                logging.info("Epoch: {}".format(epoch + 1))
            if self.num_gpus > 1:
                train_sampler.set_epoch(epoch)

            for i, batch in enumerate(dataloader):
                if self.opts.rank == 0:
                    start_b = time.time()

                feats, lengths, filenames, f0s = batch
                f0s = f0s.to(device)
                y_g_hat, commit_loss, metrics = generator(f0=f0s)
                f0_commit_loss = commit_loss[0]
                f0_metrics = metrics[0]

                optim_g.zero_grad()

                # L2 Reconstruction Loss
                loss_recons = F.mse_loss(y_g_hat, f0s)

                loss_quant = f0_commit_loss * self.opts.quant_commit
                loss = loss_recons + loss_quant

                loss.backward()
                optim_g.step()

                if self.opts.rank == 0:
                    if steps % 20 == 0:
                        logging.info(
                            "Steps: {:d}, Gen Loss Total: {:4.3f}, Loss MSE: {:4.3f}, Loss quant: {:4.3f}, s/b: {:4.3f}".format(
                                steps,
                                loss,
                                loss_recons,
                                loss_quant,
                                time.time() - start_b,
                            )
                        )
                    if steps % self.opts.checkpoint_interval == 0 and steps != 0:
                        checkpoint_path = "{}/g_{:08d}".format(
                            self.opts.checkpoint_path, steps
                        )
                        sw.add_scalar("training/gen_loss_total", loss_recons, steps)
                        sw.add_scalar("training/commit_error", f0_commit_loss, steps)
                        sw.add_scalar(
                            "training/used_curr", f0_metrics["used_curr"].item(), steps
                        )
                        sw.add_scalar(
                            "training/entropy", f0_metrics["entropy"].item(), steps
                        )
                        sw.add_scalar(
                            "training/usage", f0_metrics["usage"].item(), steps
                        )
                        generator.eval()
                        torch.cuda.empty_cache()
                        val_err_tot = 0
                        val_quant_tot = 0
                        with torch.no_grad():
                            for j, batch in enumerate(dataloader_test):
                                feats, lengths, filenames, f0s = batch

                                f0s = f0s.to(device)
                                y_g_hat, commit_loss, _ = generator(f0=f0s)

                                val_quant_tot += commit_loss[0]
                                val_err_tot += F.mse_loss(y_g_hat, f0s).item()

                            val_err = val_err_tot / (j + 1)
                            val_quant = val_quant_tot / (j + 1)
                            sw.add_scalar(
                                "validation/total_loss",
                                val_err + val_quant * self.opts.quant_commit,
                                steps,
                            )
                            sw.add_scalar("validation/commit_error", val_quant, steps)
                            sw.add_scalar("validation/commit_error", val_quant, steps)
                            logging.info(
                                "\nValidation: {:d}, Loss Total: {:4.3f}, Loss MSE: {:4.3f}, Loss quant: {:4.3f}\n".format(
                                    steps,
                                    val_err + val_quant * self.opts.quant_commit,
                                    val_err,
                                    val_quant,
                                )
                            )

                        _g = (
                            copy.deepcopy(
                                generator.module if self.num_gpus > 1 else generator
                            )
                            .cpu()
                            .state_dict()
                        )
                        _is_new_best = False
                        if val_err + val_quant * self.opts.quant_commit < best_val_err:
                            best_val_err = val_err + val_quant * self.opts.quant_commit
                            _is_new_best = True

                        torch.save(
                            {
                                "generator": _g,
                                "optim_g": optim_g.state_dict(),
                                "steps": steps,
                                "epoch": epoch,
                                "best_val_err": best_val_err,
                            },
                            checkpoint_path,
                        )

                        if _is_new_best:
                            symlink = Path(self.opts.checkpoint_path + "/g_best")
                            if symlink.is_symlink():
                                symlink.unlink()
                            symlink.symlink_to(os.path.basename(checkpoint_path))

                        if steps >= 10000 and (steps - 10000) % 10000 != 0:
                            mdl = "{}/g_{:08d}".format(
                                self.opts.checkpoint_path,
                                steps - 10000,
                            )
                            if os.path.isfile(mdl) and os.path.basename(
                                os.path.realpath(self.opts.checkpoint_path + "/g_best")
                            ) != "g_{:08d}".format(steps - 10000):
                                script_utils.run(["rm", mdl])

                    generator.train()
                steps += 1

            scheduler_g.step()

            if self.opts.rank == 0:
                logging.info(
                    "Time taken for epoch {} is {} sec\n".format(
                        epoch + 1, int(time.time() - start)
                    )
                )

        if self.opts.rank == 0:
            logging.info("Finished training")
