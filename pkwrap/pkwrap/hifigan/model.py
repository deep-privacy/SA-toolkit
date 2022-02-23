from dataclasses import dataclass
import logging

import torch
import torch.nn.functional as F
from torch.distributed import init_process_group
from torch.utils.tensorboard import SummaryWriter
import os
import json
from pathlib import Path
import glob
import copy
import time
import itertools
from carbontracker.tracker import CarbonTracker


from .. import utils
from .. import script_utils
from . import dataset
from . import nn_speech

#  torch.set_num_threads(1)
#  torch.set_num_interop_threads(1)

#############################
#  F0 Quantification model  #
#############################


@dataclass
class ModelOpts:
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
    training_epochs: int = 1500
    checkpoint_interval: int = 1000  # in step (iter)
    quant_commit: float = 0.02
    f0_stats: str = "{'f0_mean': 209.04119886766213, 'f0_std': 58.75603900262766}"  # single speaker f0 adaptation
    num_workers: int = 4
    train_utterances: list = "utt1.wav,utt2.wav"
    test_utterances: list = "utt1.wav,utt2.wav"
    cold_restart: bool = False  # restart training at the 0 epoch instead of the one provided by "init_weight_model"

    def load_from_config(self, cfg):
        for key, value in cfg.items():
            if hasattr(self, key):
                type_of_value = type(getattr(self, key))
                setattr(self, key, type_of_value(value))
        return self


class _AbstractModel:
    def __init__(self):
        pass

    def sample_interval(self, feats, lengths, filenames, f0s, ys):
        """
        Sample from a batch given a opts.segment_size
        return:
            feats: torchaudio.load() features
            lengths: lengths of each source audio file (not equal to feats since it has some padding)
            filenames: filename of the audios
            f0s: the F0 features
            ys: the torchaudio.load()+librosa normalized audio files ready for voice conversion training
            tuple: tuple of above features that where not sample at the interval
        """
        ori_feats, ori_lengths, ori_f0s, ori_ys = (
            feats.clone(),
            lengths.clone(),
            f0s.clone(),
            ys.clone(),
        )
        acc_feats, acc_f0s, acc_ys, intervals = [], [], [], []
        for batch_idx in range(feats.shape[0]):
            new_seqs, iterval_idx = dataset.sample_interval(
                [feats[batch_idx], ys[batch_idx], f0s[batch_idx]],
                seq_len=self.opts.segment_size,
                max_len=lengths[batch_idx].item(),
            )
            _feats, _ys, _f0s = new_seqs
            acc_feats.append(_feats)
            acc_f0s.append(_f0s)
            acc_ys.append(_ys)
            intervals.append((iterval_idx[0], iterval_idx[2]))
        feats = torch.stack(acc_feats)
        f0s = torch.stack(acc_f0s)
        ys = torch.stack(acc_ys)
        lengths = torch.zeros_like(lengths) + self.opts.segment_size
        return (
            feats,
            lengths,
            filenames,
            f0s,
            ys,
            (ori_feats, ori_lengths, ori_f0s, ori_ys),
            intervals,
        )

    def init_cuda_model_distributed(self):
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
        init_weight_provided = None
        if self.opts.init_weight_model != "":
            if os.path.islink(self.opts.init_weight_model):
                self.opts.init_weight_model = os.path.realpath(
                    self.opts.init_weight_model
                )
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

        if hasattr(generator, "after_load_hook"):
            generator.after_load_hook()

        generator = generator.to(device)
        if self.num_gpus > 1:
            generator = torch.nn.parallel.DistributedDataParallel(
                generator,
                device_ids=[self.opts.rank],
                find_unused_parameters=True,
            )
        self.Net().to(device).validate_model(device=device)

        return generator, device, init_weight_provided


class F0QuantModel(_AbstractModel):
    def __init__(self, model_cls, **kwargs):
        super().__init__()
        self.opts = ModelOpts()
        self.opts.load_from_config(kwargs)

        self.Net = model_cls
        self.num_gpus = int(os.getenv("WORLD_SIZE", "1"))

        self.opts.f0_stats = json.loads(self.opts.f0_stats.replace("'", '"'))

        self.call_by_mode()

    def call_by_mode(self):
        if self.opts.mode in ["train", "training"]:
            self.train()

    def train(self):
        generator, device, init_weight_provided = self.init_cuda_model_distributed()
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
                self.opts.f0_stats, get_func=self.sample_interval
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
                "Caching f0 (might be slow: setting a higher 'num_workers' can speed-up this part (32 for f0 extration is good, while 4 for training is enough))"
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

                feats, lengths, filenames, f0s, _, _, _ = batch
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
                                feats, lengths, filenames, f0s, _ = batch

                                f0s = f0s.to(device)
                                y_g_hat, commit_loss, _ = generator(f0=f0s)

                                val_quant_tot += commit_loss[0]
                                val_err_tot += F.mse_loss(y_g_hat, f0s).item()

                            val_err = val_err_tot / (j + 1)
                            val_quant = val_quant_tot / (j + 1)
                            sw.add_scalar(
                                "validation/gen_loss_total",
                                val_err + val_quant * self.opts.quant_commit,
                                steps,
                            )
                            sw.add_scalar(
                                "validation/mse_loss",
                                val_err,
                                steps,
                            )
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


#####################
#  Hifi Gan Model   #
#####################


class HifiGanModel(_AbstractModel):
    def __init__(self, model_cls, **kwargs):
        super().__init__()
        self.opts = ModelOpts()
        self.opts.load_from_config(kwargs)

        self.Net = model_cls
        self.num_gpus = int(os.getenv("WORLD_SIZE", "1"))

        self.opts.f0_stats = json.loads(self.opts.f0_stats.replace("'", '"'))

        self.call_by_mode()

    def call_by_mode(self):
        if self.opts.mode in ["train", "training"]:
            self.train()

    def train(self):
        generator, device, init_weight_provided = self.init_cuda_model_distributed()

        mpd = nn_speech.MultiPeriodDiscriminator().to(device)
        msd = nn_speech.MultiScaleDiscriminator().to(device)

        if init_weight_provided != None:
            mpd.load_state_dict(
                torch.load(self.opts.init_weight_model.replace("g_", "d_"))["mpd"]
            )
            msd.load_state_dict(
                torch.load(self.opts.init_weight_model.replace("g_", "d_"))["msd"]
            )

        if self.num_gpus > 1:
            mpd = torch.nn.parallel.DistributedDataParallel(
                mpd,
                device_ids=[self.opts.rank],
                find_unused_parameters=True,
            ).to(device)
            msd = torch.nn.parallel.DistributedDataParallel(
                msd,
                device_ids=[self.opts.rank],
                find_unused_parameters=True,
            ).to(device)

        optim_g = torch.optim.AdamW(
            generator.parameters(),
            self.opts.lr,
            betas=[self.opts.adam_b1, self.opts.adam_b2],
        )
        optim_d = torch.optim.AdamW(
            itertools.chain(msd.parameters(), mpd.parameters()),
            self.opts.lr,
            betas=[self.opts.adam_b1, self.opts.adam_b2],
        )

        steps = 0
        best_val_err = 9999999
        last_epoch = -1
        if self.opts.init_weight_model != "":
            if not self.opts.cold_restart:
                optim_g.load_state_dict(init_weight_provided["optim_g"])
                optim_d.load_state_dict(init_weight_provided["optim_d"])
                steps = init_weight_provided["steps"] + 1
                last_epoch = init_weight_provided["epoch"]
                best_val_err = init_weight_provided["best_val_err"]

        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optim_g, gamma=self.opts.lr_decay, last_epoch=last_epoch
        )
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            optim_d, gamma=self.opts.lr_decay, last_epoch=last_epoch
        )

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
                self.opts.f0_stats, get_func=self.sample_interval
            ),
            sampler=train_sampler,
            persistent_workers=True,
        )

        if self.opts.rank == 0:
            dataset_test = dataset.WavList(self.opts.test_utterances)
            dataloader_test = torch.utils.data.DataLoader(
                dataset_test,
                batch_size=4,
                shuffle=False,
                num_workers=self.opts.num_workers,
                collate_fn=dataset.collate_fn_padd(self.opts.f0_stats),
                persistent_workers=True,
            )
            sw = SummaryWriter(os.path.join(self.opts.checkpoint_path, "logs"))

        generator.train()
        mpd.train()
        msd.train()

        if self.opts.rank == 0:
            logging.info(f"Starting training from epoch: {max(0, last_epoch)}")
            logging.info(
                f"Logging:\n\ttensorboard --logdir {self.opts.checkpoint_path} --samples_per_plugin=images=100000,audio=100000"
            )

        if self.opts.rank == 0:
            carbonTracker = CarbonTracker(
                epochs=self.opts.training_epochs - max(0, last_epoch),
                components="gpu",
                verbose=2,
            )
        for epoch in range(max(0, last_epoch), self.opts.training_epochs):
            if self.opts.rank == 0:
                carbonTracker.epoch_start()
                start = time.time()
                logging.info("Epoch: {}".format(epoch + 1))
            if self.num_gpus > 1:
                train_sampler.set_epoch(epoch)

            for i, batch in enumerate(dataloader):
                if self.opts.rank == 0:
                    start_b = time.time()

                (
                    feats,
                    lengths,
                    filenames,
                    f0s,
                    ys,
                    no_sampling_inter,
                    sampling_iter_idx,
                ) = batch
                f0s, feats, ys = f0s.to(device), feats.to(device), ys.to(device)
                ori_feats, ori_lengths, ori_f0s, ori_ys = no_sampling_inter
                no_sampling_inter = (
                    ori_feats.to(device),
                    ori_lengths.to(device),
                    ori_f0s.to(device),
                    ori_ys.to(device),
                )
                y_g_hat = generator(
                    f0=f0s,
                    audio=feats,
                    full_audio_to_cache=no_sampling_inter,
                    filenames=filenames,
                    sampling_iter_idx=sampling_iter_idx,
                )

                assert (
                    y_g_hat.shape[2] >= ys.shape[-1] - 4000
                    and y_g_hat.shape[2] <= ys.shape[-1] + 4000
                ), f"Mismatch too high in vocoder output shape - {y_g_hat.shape} != {ys.shape}"

                # Trucate vocoder output
                if y_g_hat.shape[-1] > ys.shape[-1]:
                    y_g_hat = y_g_hat[:, :, : feats.shape[-1]]
                if y_g_hat.shape[-1] < ys.shape[-1]:
                    ys = ys[:, :, : y_g_hat.shape[-1]]

                y_g_hat_mel = dataset.mel_spectrogram(
                    y=y_g_hat.squeeze(1),
                    n_fft=1024,
                    num_mels=80,
                    sampling_rate=16000,
                    hop_size=256,
                    win_size=1024,
                    fmin=0,
                    fmax=8000,
                )

                y_mel = dataset.mel_spectrogram(
                    y=ys.squeeze(1),
                    n_fft=1024,
                    num_mels=80,
                    sampling_rate=16000,
                    hop_size=256,
                    win_size=1024,
                    fmin=0,
                    fmax=8000,
                )

                optim_d.zero_grad()

                def loss_discriminators(ys, y_g_hat):

                    # MPD
                    y_df_hat_r, y_df_hat_g, _, _ = mpd(ys, y_g_hat.detach())
                    (
                        loss_disc_f,
                        losses_disc_f_r,
                        losses_disc_f_g,
                    ) = nn_speech.discriminator_loss(y_df_hat_r, y_df_hat_g)

                    # MSD
                    y_ds_hat_r, y_ds_hat_g, _, _ = msd(ys, y_g_hat.detach())
                    (
                        loss_disc_s,
                        losses_disc_s_r,
                        losses_disc_s_g,
                    ) = nn_speech.discriminator_loss(y_ds_hat_r, y_ds_hat_g)

                    loss_disc_all = loss_disc_s + loss_disc_f
                    return loss_disc_all

                loss_discriminators(ys, y_g_hat).backward()
                optim_d.step()

                # Generator
                optim_g.zero_grad()

                # L1 Mel-Spectrogram Loss
                loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(ys, y_g_hat)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(ys, y_g_hat)
                loss_fm_f = nn_speech.feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = nn_speech.feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_f, losses_gen_f = nn_speech.generator_loss(y_df_hat_g)
                loss_gen_s, losses_gen_s = nn_speech.generator_loss(y_ds_hat_g)
                loss_gen_all = (
                    loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
                )

                loss_gen_all.backward()
                optim_g.step()

                if self.opts.rank == 0:
                    if steps % 20 == 0:
                        torch.cuda.empty_cache()
                        logging.info(
                            "Steps: {:d}, Gen Loss Total: {:4.3f}, Mel-Spec. Error: {:4.3f}, s/b: {:4.3f}".format(
                                steps,
                                loss_gen_all,
                                loss_mel / 45,
                                time.time() - start_b,
                            )
                        )
                    if steps % self.opts.checkpoint_interval == 0:
                        checkpoint_path = "{}/g_{:08d}".format(
                            self.opts.checkpoint_path, steps
                        )
                        checkpoint_path_d = "{}/d_{:08d}".format(
                            self.opts.checkpoint_path, steps
                        )

                        sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                        sw.add_scalar("training/mel_spec_error", loss_mel / 45, steps)

                        generator.eval()
                        val_err_tot = 0
                        gen_loss_tot = 0
                        with torch.no_grad():
                            for j, batch in enumerate(dataloader_test):
                                feats, lengths, filenames, f0s, ys = batch

                                f0s = f0s.to(device)
                                feats = feats.to(device)
                                ys = ys.to(device)
                                y_g_hat = generator(f0=f0s, audio=feats, filenames=filenames)

                                assert (
                                    y_g_hat.shape[2] >= ys.shape[-1] - 8000
                                    and y_g_hat.shape[2] <= ys.shape[-1] + 8000
                                ), f"Mismatch too high in vocoder output shape - {y_g_hat.shape} != {ys.shape}"

                                # Trucate vocoder output
                                if y_g_hat.shape[-1] > ys.shape[-1]:
                                    y_g_hat = y_g_hat[:, :, : feats.shape[-1]]
                                if y_g_hat.shape[-1] < ys.shape[-1]:
                                    ys = ys[:, :, : y_g_hat.shape[-1]]

                                y_g_hat_mel = dataset.mel_spectrogram(
                                    y=y_g_hat.squeeze(1),
                                    n_fft=1024,
                                    num_mels=80,
                                    sampling_rate=16000,
                                    hop_size=256,
                                    win_size=1024,
                                    fmin=0,
                                    fmax=8000,
                                )
                                y_mel = dataset.mel_spectrogram(
                                    y=ys.squeeze(1),
                                    n_fft=1024,
                                    num_mels=80,
                                    sampling_rate=16000,
                                    hop_size=256,
                                    win_size=1024,
                                    fmin=0,
                                    fmax=8000,
                                )

                                val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()
                                gen_loss_tot += (
                                    loss_discriminators(ys, y_g_hat).sum().item()
                                ) + val_err_tot

                                if j <= 4:
                                    (
                                        feats,
                                        lengths,
                                        filenames,
                                        f0s,
                                        ys,
                                    ) = dataset.collate_fn_padd(self.opts.f0_stats)(
                                        [dataset_test[j]]
                                    )
                                    f0s = f0s.to(device)
                                    feats = feats.to(device)
                                    ys = ys.to(device)
                                    y_g_hat = generator(f0=f0s, audio=feats, filenames=filenames)

                                    if steps == 0:
                                        logging.info(
                                            "Len generated audio: "
                                            + str(y_g_hat.shape)
                                            + " - Len ground truth audio: "
                                            + str(ys.shape)
                                        )
                                        sw.add_audio(
                                            "gt/y_{}".format(j),
                                            ys,
                                            steps,
                                            16000,
                                        )
                                        sw.add_figure(
                                            "gt/y_spec_{}".format(j),
                                            dataset.plot_spectrogram(ys.squeeze(1)),
                                            steps,
                                        )

                                    sw.add_audio(
                                        "generated/y_hat_{}".format(j),
                                        y_g_hat[0],
                                        steps,
                                        16000,
                                    )
                                    sw.add_figure(
                                        "generated/y_hat_spec_{}".format(j),
                                        dataset.plot_spectrogram(y_g_hat.squeeze(1)),
                                        steps,
                                    )

                            val_err = val_err_tot / (j + 1)
                            gen_err = gen_loss_tot / (j + 1)
                            logging.info(
                                "\nValidation: {:d}, Gen Loss Total: {:4.3f}, Mel-Spec. Error: {:4.3f}\n".format(
                                    steps,
                                    gen_err,
                                    val_err,
                                )
                            )
                            sw.add_scalar("validation/mel_spec_error", val_err, steps)
                            sw.add_scalar("validation/gen_loss_total", gen_err, steps)

                        _g = (
                            generator.module if self.num_gpus > 1 else generator
                        ).state_dict()
                        _is_new_best = False
                        if gen_err < best_val_err:
                            best_val_err = gen_err
                            _is_new_best = True

                        torch.save(
                            {
                                "generator": _g,
                                "optim_g": optim_g.state_dict(),
                                "optim_d": optim_d.state_dict(),
                                "steps": steps,
                                "epoch": epoch,
                                "best_val_err": best_val_err,
                            },
                            checkpoint_path,
                        )
                        torch.save(
                            {
                                "mpd": (
                                    mpd.module if self.num_gpus > 1 else mpd
                                ).state_dict(),
                                "msd": (
                                    msd.module if self.num_gpus > 1 else msd
                                ).state_dict(),
                            },
                            checkpoint_path_d,
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
                                script_utils.run(["rm", mdl.replace("g_", "d_")])

                    torch.cuda.empty_cache()
                    generator.train()
                steps += 1

            scheduler_g.step()
            scheduler_d.step()

            if self.opts.rank == 0:
                carbonTracker.epoch_end()
                logging.info(
                    "Time taken for epoch {} is {} sec\n".format(
                        epoch + 1, int(time.time() - start)
                    )
                )

        if self.opts.rank == 0:
            logging.info("Finished training")
            carbonTracker.stop()
