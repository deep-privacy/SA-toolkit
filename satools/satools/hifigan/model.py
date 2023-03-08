import argparse
import itertools
import json
import logging
import sys
import os
import time
from dataclasses import dataclass, fields
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.distributed import init_process_group
from torch.utils.tensorboard import SummaryWriter

from . import dataset
from . import nn
from .. import script_utils
from .. import utils
import satools


#  torch.set_num_threads(1)
#  torch.set_num_interop_threads(1)

@dataclass
class ModelOpts:
    mode: str = ""
    base_model: str = ""
    base_model_args: str = "{}"
    dirname: str = ""
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
    num_workers: int = 4
    train_set: str = "./data/part"
    dev_set: list = "./data/part"
    cold_restart: bool = False  # restart training at the 0 epoch instead of the one provided by "init_weight_model"
    new_model: str = ""

    # dataset mel_spectrogram config
    n_fft:int = 1024
    num_mels:int = 80
    sampling_rate:int = 16000
    hop_size:int = 256
    win_size:int = 1024
    fmin:int = 0
    fmax:int = 8000

    def dataset_conf(self):
        return {"n_fft": self.n_fft, "num_mels": self.num_mels, "sampling_rate": self.sampling_rate, "hop_size": self.hop_size, "win_size": self.win_size, "fmin": self.fmin, "fmax": self.fmax}

    def load_from_args(self):
        parser = argparse.ArgumentParser(description="")
        for field in fields(self):
            if field.name == "base_model":
                parser.add_argument("base_model", nargs='?') # $1 / nargs=? reutrns ChainE2EModel for testing
                continue
            parser.add_argument(f"--{field.name.replace('_', '-')}", default=field.default, type=field.type)
        args = parser.parse_args()
        return self.load_from_config(vars(args))

    def load_from_config(self, cfg):
        for key, value in cfg.items():
            if hasattr(self, key):
                type_of_value = type(getattr(self, key))
                setattr(self, key, type_of_value(value))
        return self


#####################
#  Hifi Gan Model   #
#####################


class HifiGanModel():
    def __init__(self, model_cls, cmd_line=False, **kwargs):
        super().__init__()
        self.opts = ModelOpts()
        if cmd_line:
            self.opts.load_from_args()
        else:
            self.opts.load_from_config(kwargs)

        self.Net = model_cls
        self.num_gpus = int(os.getenv("WORLD_SIZE", "1"))

        self.call_by_mode()

    def call_by_mode(self):
        if self.opts.mode in ["init"]:
            self.init()
        elif self.opts.mode == "jit_save":
            self.jit_save()
        elif self.opts.mode in ["train", "training"]:
            self.train()
        else:
            logging.critical(f"Mode '{self.opts.mode}' not defined")

    def load_state_model(self, file):
        m = torch.load(file)
        if "base_model_state_dict" in m:
            return m["base_model_state_dict"]
        return m

    def save_model(self, model, file=None):
        file = self.opts.new_model if file==None else file
        install_path = os.path.dirname(os.path.dirname(satools.__path__[0])) # dir to git clone

        torch.save({"base_model_state_dict": model.state_dict(),
                    "task_path": os.getcwd().replace(install_path, ""),
                    "install_path": install_path,
                    "base_model_path": sys.argv[0],
                    "base_model_params": {},
                    "base_model_args": json.loads(self.opts.base_model_args),
                    }, file)

    def jit_save(self):
        file = self.opts.new_model
        model = self.Net()
        model.load_state_dict(self.load_state_model(self.opts.base_model))
        model.remove_weight_norm()
        model = torch.jit.script(model)
        torch.jit.save(model, file)
        logging.info("Saved to: " + str(file))
        self.save_model(model, self.opts.base_model) # re-save old model (update dirs/exp keys)

    def init(self):
        model = self.Net()
        logging.info(str(model))
        if self.opts.init_weight_model != "":
            init_weight_provided = self.load_state_model(self.opts.init_weight_model)

            init_weight_provided_matched, unmatch = satools.utils.torch.match_state_dict(
                model.state_dict(), init_weight_provided
            )
            not_inited = model.load_state_dict(
                init_weight_provided_matched, strict=False
            )
            logging.info(
                "Init from previous model {}, layers not initialized: {}: layers ommited (wrong shape): {}".format(
                    self.opts.init_weight_model,
                    str(not_inited),
                    str(unmatch.keys()),
                )
            )

        if hasattr(model, "init"):
            if self.opts.init_weight_model:
                logging.warning("'init_weight_model' in config and model.init() in pytorch model may cancel eachother")
            model.init()

        self.save_model(model, self.opts.base_model)

        mpd = nn.MultiPeriodDiscriminator()
        msd = nn.MultiScaleDiscriminator()
        file = self.opts.base_model.replace("g_", "d_")
        torch.save({ "mpd": ( mpd.module if self.num_gpus > 1 else mpd ).state_dict(), "msd": ( msd.module if self.num_gpus > 1 else msd ).state_dict(), }, file)

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
        generator.load_state_dict(self.load_state_model(self.opts.base_model))
        generator = generator.to(device)

        mpd = nn.MultiPeriodDiscriminator()
        msd = nn.MultiScaleDiscriminator()
        mpd.load_state_dict(torch.load(self.opts.base_model.replace("g_", "d_"))["mpd"])
        msd.load_state_dict(torch.load(self.opts.base_model.replace("g_", "d_"))["msd"])
        mpd.to(device)
        msd.to(device)

        _networks = [generator, mpd, msd]
        if self.num_gpus > 1:
            for i in range(len(_networks)):
                _networks[i] = torch.nn.parallel.DistributedDataParallel(
                    _networks[i], device_ids=[self.opts.rank],
                )

        return tuple(_networks), device

    def init_train_optims(self, optim_g, optim_d):
        steps = 0
        best_val_err = 9999999
        last_epoch = -1

        f = self.opts.base_model.replace("g_", "optim_")
        if Path(f).is_file():
            sd = torch.load(f)
            optim_g.load_state_dict(sd["optim_g"])
            optim_d.load_state_dict(sd["optim_d"])
            steps = sd["steps"] + 1
            last_epoch = sd["epoch"]
            best_val_err = sd["best_val_err"]

        return optim_g, optim_d, steps, best_val_err, last_epoch

    def train(self):
        (generator, mpd, msd), device = self.init_cuda_model_distributed()

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

        optim_g, optim_d, steps, best_val_err, last_epoch = self.init_train_optims(optim_g, optim_d)
        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=self.opts.lr_decay, last_epoch=last_epoch)
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=self.opts.lr_decay, last_epoch=last_epoch)

        wavs_scp = utils.kaldi.read_wav_scp(self.opts.train_set + "/wav.scp")
        trainset = dataset.WavList(list(wavs_scp.values()), list(wavs_scp.keys()), load_func=utils.kaldi.load_wav_from_scp)

        train_sampler = (torch.utils.data.DistributedSampler(trainset) if self.num_gpus > 1 else None)

        dataloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.opts.minibatch_size,
            shuffle=False,
            num_workers=self.opts.num_workers,
            collate_fn=dataset.collate_fn_padd(),
            sampler=train_sampler,
            persistent_workers=True,
        )

        if self.opts.rank == 0:
            wavs_scp = utils.kaldi.read_wav_scp(self.opts.dev_set + "/wav.scp")
            dataset_test = dataset.WavList(list(wavs_scp.values()), list(wavs_scp.keys()), load_func=utils.kaldi.load_wav_from_scp)
            dataloader_test = torch.utils.data.DataLoader(
                dataset_test,
                batch_size=4,
                shuffle=False,
                num_workers=self.opts.num_workers,
                collate_fn=dataset.collate_fn_padd(),
                persistent_workers=True,
            )
            sw = SummaryWriter(os.path.join(self.opts.dirname, "logs"))

        generator.train()
        mpd.train()
        msd.train()

        if self.opts.rank == 0:
            logging.info(f"Starting training from epoch: {max(0, last_epoch)}")
            logging.info(
                f"Logging:\n\ttensorboard --logdir {self.opts.dirname} --samples_per_plugin=images=100000,audio=100000"
            )

        for epoch in range(max(0, last_epoch), self.opts.training_epochs):
            if self.opts.rank == 0:
                start = time.time()
                logging.info("Epoch: {}".format(epoch + 1))
            if self.num_gpus > 1:
                train_sampler.set_epoch(epoch)

            optimizer_was_run = False
            for i, batch in enumerate(dataloader):
                if self.opts.rank == 0:
                    start_b = time.time()

                (audio, lengths, filenames, ys) = batch
                audio, ys = audio.to(device), ys.to(device)
                y_g_hat = generator(audio)

                assert (
                    y_g_hat.shape[2] >= ys.shape[-1] - 4000
                    and y_g_hat.shape[2] <= ys.shape[-1] + 4000
                ), f"Mismatch too high in vocoder output shape - {y_g_hat.shape} != {ys.shape}"

                # Trucate vocoder output
                if y_g_hat.shape[-1] > ys.shape[-1]:
                    y_g_hat = y_g_hat[:, :, : feats.shape[-1]]
                if y_g_hat.shape[-1] < ys.shape[-1]:
                    ys = ys[:, :, : y_g_hat.shape[-1]]

                y_g_hat_mel = dataset.mel_spectrogram(y=y_g_hat.squeeze(1), *self.opts.dataset_conf())

                y_mel = dataset.mel_spectrogram(y=ys.squeeze(1), *self.opts.dataset_conf())

                optim_d.zero_grad()

                def loss_discriminators(ys, y_g_hat):

                    # MPD
                    y_df_hat_r, y_df_hat_g, _, _ = mpd(ys, y_g_hat.detach())
                    (
                        loss_disc_f,
                        losses_disc_f_r,
                        losses_disc_f_g,
                    ) = nn.discriminator_loss(y_df_hat_r, y_df_hat_g)

                    # MSD
                    y_ds_hat_r, y_ds_hat_g, _, _ = msd(ys, y_g_hat.detach())
                    (
                        loss_disc_s,
                        losses_disc_s_r,
                        losses_disc_s_g,
                    ) = nn.discriminator_loss(y_ds_hat_r, y_ds_hat_g)

                    loss_disc_all = loss_disc_s + loss_disc_f
                    return loss_disc_all

                loss_discriminators(ys, y_g_hat).backward()
                optim_d.step()

                # Generator
                optimizer_was_run = True
                optim_g.zero_grad()

                # L1 Mel-Spectrogram Loss
                loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(ys, y_g_hat)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(ys, y_g_hat)
                loss_fm_f = nn.feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = nn.feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_f, losses_gen_f = nn.generator_loss(y_df_hat_g)
                loss_gen_s, losses_gen_s = nn.generator_loss(y_ds_hat_g)
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
                        dirname = "{}/g_{:08d}".format(self.opts.dirname, steps)
                        checkpoint_path_d = "{}/d_{:08d}".format(self.opts.dirname, steps)

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

                                y_g_hat_mel = dataset.mel_spectrogram(y=y_g_hat.squeeze(1), *self.opts.dataset_conf())
                                y_mel = dataset.mel_spectrogram(y=ys.squeeze(1), *self.opts.dataset_conf())

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
                                    ) = dataset.collate_fn_padd()([dataset_test[j]])
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
                                        sw.add_audio("gt/y_{}".format(j), ys, steps, self.opts.sampling_rate)
                                        sw.add_figure(
                                            "gt/y_spec_{}".format(j),
                                            dataset.plot_spectrogram(ys.squeeze(1)),
                                            steps,
                                        )

                                    sw.add_audio("generated/y_hat_{}".format(j), y_g_hat[0], steps, self.opts.sampling_rate)
                                    sw.add_figure("generated/y_hat_spec_{}".format(j), dataset.plot_spectrogram(y_g_hat.squeeze(1)), steps)

                            val_err = val_err_tot / (j + 1)
                            gen_err = gen_loss_tot / (j + 1)
                            logging.info("\nValidation: {:d}, Gen Loss Total: {:4.3f}, Mel-Spec. Error: {:4.3f}\n".format( steps, gen_err, val_err, ))
                            sw.add_scalar("validation/mel_spec_error", val_err, steps)
                            sw.add_scalar("validation/gen_loss_total", gen_err, steps)

                        _g = (generator.module if self.num_gpus > 1 else generator).state_dict()
                        _is_new_best = False
                        if gen_err < best_val_err:
                            best_val_err = gen_err
                            _is_new_best = True

                        torch.save({ "generator": _g, "optim_g": optim_g.state_dict(), "optim_d": optim_d.state_dict(), "steps": steps, "epoch": epoch, "best_val_err": best_val_err, }, dirname)
                        torch.save({ "mpd": ( mpd.module if self.num_gpus > 1 else mpd ).state_dict(), "msd": ( msd.module if self.num_gpus > 1 else msd ).state_dict(), }, checkpoint_path_d)

                        if _is_new_best:
                            symlink = Path(self.opts.dirname + "/g_best")
                            if symlink.is_symlink():
                                symlink.unlink()
                            symlink.symlink_to(os.path.basename(dirname))

                        if steps >= 10000 and (steps - 10000) % 10000 != 0:
                            mdl = "{}/g_{:08d}".format(self.opts.dirname, steps - 10000)
                            if os.path.isfile(mdl) and os.path.basename(os.path.realpath(self.opts.dirname + "/g_best")) != "g_{:08d}".format(steps - 10000):
                                script_utils.run(["rm", mdl])
                                script_utils.run(["rm", mdl.replace("g_", "d_")])

                    torch.cuda.empty_cache()
                    generator.train()
                steps += 1

            if optimizer_was_run:
                scheduler_g.step()
                scheduler_d.step()

            if self.opts.rank == 0:
                logging.info("Time taken for epoch {} is {} sec\n".format( epoch + 1, int(time.time() - start) ))

        if self.opts.rank == 0:
            logging.info("Finished training")
