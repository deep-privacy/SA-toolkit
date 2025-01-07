import argparse
import itertools
import logging
import sys
import os
import time
import datetime
from dataclasses import dataclass, fields
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.distributed import init_process_group

from . import dataset
from . import nn
from .. import script_utils
from .. import utils
import satools

logging.basicConfig(level=logging.INFO, format="satools %(levelname)s: %(message)s")
logging.getLogger("geocoder").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

import warnings
warnings.filterwarnings(
    "ignore", message=r'.*but did not find any unused parameters.*'
)


@dataclass
class DataloadingOpts:
    cache_path: str = "./data/cache"
    cache_functions: str = "[]"
    cache_worker_name: str = ""
    minibatch_size: int = 32
    num_worker_dataloader: int = 4
    rank: int = 0

@dataclass
class ModelOpts(DataloadingOpts):
    mode: str = ""
    base_model: str = ""
    base_model_args: str = "{}"
    dirname: str = ""
    num_gpus: int = 1
    lr: float = 0.0002
    init_weight_model: str = ""
    adam_b1: float = 0.8
    adam_b2: float = 0.99
    lr_decay: float = 0.999
    segment_size: float = 16640
    training_epochs: int = 1500
    checkpoint_interval: int = 1000  # in step (iter)
    train_set: str = "./data/part"
    dev_set: str = "./data/part"
    new_model: str = ""
    max_len_missmatch: int = 200 # allow some missmatch between original an converted speech
    logging_interval: int = 20
    safe_gpu: str = "false"

    torch_compile: str = "false"

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
        self.opts.num_gpus = int(os.environ.get("WORLD_SIZE", "1"))
        self.opts.rank = int(os.environ.get("LOCAL_RANK", "0"))

        if self.opts.rank != 0:
            logging.disable(logging.INFO)

        self.call_by_mode()

    def call_by_mode(self):
        self.reset_utt2spk()
        if self.opts.mode in ["init"]:
            self.init()
        elif self.opts.mode == "jit_save":
            self.jit_save()
        elif self.opts.mode in ["train", "training"]:
            if self.opts.safe_gpu.lower() == "lower":
                utils.safe_gpu()
            self.train()
        else:
            logging.critical(f"Mode '{self.opts.mode}' not defined")

    def reset_utt2spk(self):
        # what if the user wants to pass it? Just override this function
        utt2spk_scp = utils.kaldi.read_wav_scp(self.opts.train_set + "/utt2spk")
        self.utt2spk = utt2spk_scp

    def load_state_model(self, file):
        m = torch.load(file, weights_only=False)
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
                    "base_model_params": {"utt2spk": self.utt2spk},
                    "base_model_args": utils.fix_json(self.opts.base_model_args),
                    }, file)

        torch.save({
                    "task_path": os.getcwd().replace(install_path, ""),
                    "install_path": install_path,
                    "base_model_path": sys.argv[0],
                    "base_model_params": {"utt2spk": self.utt2spk},
                    "base_model_args": utils.fix_json(self.opts.base_model_args),
                    }, os.path.join(self.opts.dirname, "conf.pt"))

    def jit_save(self):
        logging.info("Creating a JIT model for easy sharing")
        file = self.opts.new_model
        model = self.Net(utt2spk=self.utt2spk)
        model.load_state_dict(self.load_state_model(self.opts.base_model))
        self.save_model(model, self.opts.base_model) # re-save old model (update install_path/base_model_args/.. keys)
        model.remove_weight_norm()
        model = torch.jit.script(model)
        torch.jit.save(model, file)
        logging.info("Saved to: " + str(file))

    def init(self):
        model = self.Net(utt2spk=self.utt2spk)
        logging.info(str(model))
        if self.opts.init_weight_model != "" and os.path.exists(self.opts.init_weight_model):
            init_weight_provided = self.load_state_model(self.opts.init_weight_model)

            init_weight_provided_matched, unmatch = utils.torch.match_state_dict(
                model.state_dict(), init_weight_provided
            )
            not_inited = model.load_state_dict(
                init_weight_provided_matched, strict=False
            )
            logging.info(
                "Init generator from previous model {}, layers not initialized: {}: layers ommited (wrong shape): {}".format(
                    self.opts.init_weight_model,
                    str(not_inited),
                    str(unmatch.keys()),
                )
            )

        if hasattr(model, "init"):
            if self.opts.init_weight_model:
                logging.warning("'init_weight_model' in config and model.init() in pytorch model may cancel eachother")
            model.init()

        trainset = utils.WavScpDataset.from_wav_scpfile(self.opts.train_set + "/wav.scp")
        if hasattr(model, "fake_epoch"):
            logging.info("Running a fake first epoch")
            device = torch.device("cuda")
            model.to(device)
            self.opts.cache_worker_name = "train_set"
            for j in tqdm(range(len(trainset))):
                batch = dataset.model_collate(model, self.opts, auto_sample=False)([trainset[j]])
                batch.compute_cuda_extract_feat(model, self.opts, "train_set", device)
                batch.to(device)
                run_after = model.fake_epoch(batch)
            utils.feature_extractor_decorator.merge_cache(model, self.opts.cache_path)
            if hasattr(model, "fake_end_after"):
                model.fake_end_after()

        model.to("cpu")
        self.save_model(model, self.opts.base_model)

        mpd = nn.MultiPeriodDiscriminator()
        msd = nn.MultiScaleDiscriminator()
        if self.opts.init_weight_model != "":
            discriminators = os.path.realpath(self.opts.init_weight_model).replace("g_", "d_")
            if os.path.exists(discriminators):
                logging.info("Init discriminators from previous model")
                try:
                    mpd.load_state_dict(torch.load(discriminators, weights_only=False)["mpd"])
                    msd.load_state_dict(torch.load(discriminators, weights_only=False)["msd"])
                except:
                    logging.warning(f"Failed to load discriminators from path: {discriminators}")

        file = self.opts.base_model.replace("g_", "d_")
        torch.save({ "mpd":  mpd.state_dict(), "msd": msd.state_dict() }, file)

    def init_cuda_model_distributed(self):
        device = torch.device("cuda")
        if self.opts.num_gpus > 1:
            device = torch.device("cuda:{:d}".format(self.opts.rank))
            logging.warning(
                "Init from distributed training rank: {}".format(self.opts.rank)
            )
            init_process_group(
                backend="nccl",
                init_method="env://",
                rank=self.opts.rank,
            )

        generator = self.Net(utt2spk=self.utt2spk)
        generator.load_state_dict(self.load_state_model(self.opts.base_model))
        generator = generator.to(device)

        mpd = nn.MultiPeriodDiscriminator()
        mpd.load_state_dict(torch.load(self.opts.base_model.replace("g_", "d_"), weights_only=False)["mpd"])
        mpd.to(device)
        msd = nn.MultiScaleDiscriminator()
        msd.load_state_dict(torch.load(self.opts.base_model.replace("g_", "d_"), weights_only=False)["msd"])
        msd.to(device)

        _networks = [generator, mpd, msd]
        if self.opts.num_gpus > 1:
            for i in range(len(_networks)):
                _networks[i] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(_networks[i])
                _networks[i] = torch.nn.parallel.DistributedDataParallel(
                    _networks[i], device_ids=[self.opts.rank], find_unused_parameters=True
                )
                _networks[i] = utils.torch.WrappedTorchDDP(_networks[i])

        return tuple(_networks), device

    def init_train_optims(self, optim_g, optim_d):
        steps = 0
        best_val_err = 9999999
        last_epoch = -1

        f = os.path.realpath(self.opts.base_model).replace("g_", "trainer_")
        if Path(f).is_file():
            logging.info(f"Loading trainer from: {f}")
            sd = torch.load(f, weights_only=False)
            optim_g.load_state_dict(sd["optim_g"])
            optim_d.load_state_dict(sd["optim_d"])
            steps = sd["steps"] + 1
            last_epoch = sd["epoch"]
            best_val_err = sd["best_val_err"]

        return optim_g, optim_d, steps, best_val_err, last_epoch

    def truncate(self, ys, y_gen, mult=1):
        assert (
            y_gen.shape[-1] >= ys.shape[-1] - (self.opts.max_len_missmatch * mult)
            and y_gen.shape[-1] <= ys.shape[-1] + (self.opts.max_len_missmatch * mult)
        ), f"Mismatch too high in vocoder output shape '{y_gen.shape} != {ys.shape}' max miss match = {self.opts.max_len_missmatch} * {mult}"

        # Trucate vocoder output
        if y_gen.shape[-1] > ys.shape[-1]:
            y_gen = y_gen[:, :, : ys.shape[-1]]
        if y_gen.shape[-1] < ys.shape[-1]:
            ys = ys[:, :, : y_gen.shape[-1]]
        return ys, y_gen

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

        trainset = utils.WavScpDataset.from_wav_scpfile(self.opts.train_set + "/wav.scp")

        train_sampler = (torch.utils.data.DistributedSampler(trainset, shuffle=True) if self.opts.num_gpus > 1 else None)

        self.opts.cache_worker_name = "train_set"
        dataloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.opts.minibatch_size,
            shuffle=train_sampler==None,
            num_workers=self.opts.num_worker_dataloader,
            collate_fn=dataset.model_collate(generator, self.opts),
            persistent_workers=True,
            sampler=train_sampler,
        )

        if self.opts.rank == 0:
            dataset_test = utils.WavScpDataset.from_wav_scpfile(self.opts.dev_set + "/wav.scp")
            dataset_test.wavs_idx = trainset.wavs_idx[:len(dataset_test.wavs_idx)] # Convert to speaker in the trainset for dev the set
            self.opts.cache_worker_name = "dev_set"
            dataloader_test = torch.utils.data.DataLoader(
                dataset_test,
                batch_size=4,
                shuffle=False,
                num_workers=self.opts.num_worker_dataloader,
                collate_fn=dataset.model_collate(generator, self.opts),
                persistent_workers=True,
            )
            from torch.utils.tensorboard import SummaryWriter
            sw = SummaryWriter(os.path.join(self.opts.dirname, "runs"))
            handler = utils.LogHandlerSummaryWriter(sw)
            handler.setFormatter(logging.Formatter("`" + logging.root.handlers[0].formatter._fmt + "`"))
            logging.getLogger().addHandler(handler)


        generator.train()
        mpd.train()
        msd.train()

        if self.opts.torch_compile.lower() == "true":
            logging.info(f"torch.compile network..")
            generator = torch.compile(generator)
            mpd = torch.compile(mpd)
            msd = torch.compile(msd)

        logging.info(
            f"Logging:\n\ttensorboard --logdir {self.opts.dirname} --samples_per_plugin=images=100000,audio=100000"
        )

        dataset.setup_extractor(generator, self.opts)

        for epoch in range(last_epoch+1, self.opts.training_epochs):
            if self.opts.num_gpus > 1:
                train_sampler.set_epoch(epoch)

            optimizer_was_run = False
            for i, batch in enumerate(dataloader):
                start_b = time.time()

                try:
                    batch.compute_cuda_extract_feat(generator, self.opts, "train_set", device)
                    batch.to(device)
                    batch.sample(self.opts.segment_size)
                    y_gen = generator(batch)

                    ys, y_gen = self.truncate(batch.yss, y_gen)

                    y_g_gen_mel = dataset.mel_spectrogram(y=y_gen.squeeze(1), **self.opts.dataset_conf())

                    y_mel = dataset.mel_spectrogram(y=ys.squeeze(1), **self.opts.dataset_conf())

                    optim_d.zero_grad()
                except Exception as e:
                    logging.warning("Error with: "+ str(batch))
                    raise e

                def loss_discriminators(ys, y_gen):

                    # MPD
                    y_df_gen_r, y_df_gen_g, _, _ = mpd(ys, y_gen.detach())
                    (
                        loss_disc_f,
                        losses_disc_f_r,
                        losses_disc_f_g,
                    ) = nn.discriminator_loss(y_df_gen_r, y_df_gen_g)

                    # MSD
                    y_ds_gen_r, y_ds_gen_g, _, _ = msd(ys, y_gen.detach())
                    (
                        loss_disc_s,
                        losses_disc_s_r,
                        losses_disc_s_g,
                    ) = nn.discriminator_loss(y_ds_gen_r, y_ds_gen_g)

                    loss_disc_all = loss_disc_s + loss_disc_f
                    return loss_disc_all

                loss_discriminators(ys, y_gen).backward()
                optim_d.step()

                # Generator
                optimizer_was_run = True
                optim_g.zero_grad()

                # L1 Mel-Spectrogram Loss
                loss_mel = F.l1_loss(y_mel, y_g_gen_mel) * 45

                y_df_gen_r, y_df_gen_g, fmap_f_r, fmap_f_g = mpd(ys, y_gen)
                y_ds_gen_r, y_ds_gen_g, fmap_s_r, fmap_s_g = msd(ys, y_gen)
                loss_fm_f = nn.feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = nn.feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_f, losses_gen_f = nn.generator_loss(y_df_gen_g)
                loss_gen_s, losses_gen_s = nn.generator_loss(y_ds_gen_g)
                loss_gen_all = (
                    loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
                )

                loss_gen_all.backward()
                optim_g.step()

                if self.opts.rank == 0:
                    if steps % self.opts.logging_interval == 0:
                        torch.cuda.empty_cache()
                        logging.info(
                            "{} Epoch: {:>3d}, Steps: {:>5d}, Gen Loss Total: {:4.3f}, Mel-Spec. Error: {:4.3f}, s/b: {:4.3f}".format(
                                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                epoch+1,
                                steps,
                                loss_gen_all,
                                loss_mel / 45,
                                time.time() - start_b,
                            )
                        )
                    if steps % self.opts.checkpoint_interval == 0:
                        checkpoint_path_g = "{}/g_{:08d}.pt".format(self.opts.dirname, steps)
                        checkpoint_path_trainer = "{}/trainer_{:08d}.pt".format(self.opts.dirname, steps)
                        checkpoint_path_d = "{}/d_{:08d}.pt".format(self.opts.dirname, steps)

                        sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                        sw.add_scalar("training/mel_spec_error", loss_mel / 45, steps)

                        generator.eval()
                        val_err_tot = 0
                        gen_loss_tot = 0
                        with torch.no_grad():
                            torch.cuda.empty_cache()
                            for j, batch_test in enumerate(dataloader_test):

                                try:
                                    batch_test.compute_cuda_extract_feat(generator, self.opts, "dev_set", device)
                                    batch_test.to(device)
                                    y_gen = generator(batch_test)

                                    ys, y_gen = self.truncate(batch_test.yss, y_gen, mult=batch_test.yss.shape[-1]//self.opts.segment_size) # truncate high because of batch padding on full audio

                                    y_g_gen_mel = dataset.mel_spectrogram(y=y_gen.squeeze(1), **self.opts.dataset_conf())
                                    y_mel = dataset.mel_spectrogram(y=ys.squeeze(1), **self.opts.dataset_conf())

                                    val_err_tot += F.l1_loss(y_mel, y_g_gen_mel).item()
                                    gen_loss_tot += (
                                        loss_discriminators(ys, y_gen).sum().item()
                                    ) + val_err_tot

                                    if j <= 4:
                                        batch_test = dataset.model_collate(generator, self.opts, auto_sample=False)([dataset_test[j]])
                                        batch_test.compute_cuda_extract_feat(generator, self.opts, "dev_set", device)
                                        batch_test.to(device)
                                        y_gen = generator(batch_test)

                                        if steps == 0:
                                            logging.info("Len generated audio: " + str(y_gen.shape) + " - Len ground truth audio: " + str(batch_test.yss.shape))
                                            sw.add_audio("gt/y_{}".format(j), batch_test.yss, steps, self.opts.sampling_rate)
                                            sw.add_figure(
                                                "gt/y_spec_{}".format(j),
                                                dataset.plot_spectrogram(batch_test.yss.squeeze(1)),
                                                steps,
                                            )

                                        sw.add_audio("generated/y_gen_{}".format(j), y_gen[0], steps, self.opts.sampling_rate)
                                        sw.add_figure("generated/y_gen_spec_{}".format(j), dataset.plot_spectrogram(y_gen.squeeze(1)), steps)

                                except Exception as e:
                                    logging.warning("Error with: "+ str(batch_test))
                                    logging.warning(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")

                            val_err = val_err_tot / (j + 1)
                            gen_err = gen_loss_tot / (j + 1)
                            logging.info("Validation: {:d}, Gen Loss Total: {:4.3f}".format( steps, gen_err ))
                            sw.add_scalar("validation/mel_spec_error", val_err, steps)
                            sw.add_scalar("validation/gen_loss_total", gen_err, steps)

                        if steps != 0:
                            _is_new_best = False
                            if gen_err < best_val_err:
                                best_val_err = gen_err
                                _is_new_best = True


                            self.save_model(generator, checkpoint_path_g)
                            torch.save({"optim_g": optim_g.state_dict(), "optim_d": optim_d.state_dict(), "steps": steps, "epoch": epoch, "best_val_err": best_val_err }, checkpoint_path_trainer)
                            torch.save({ "mpd": mpd.state_dict(), "msd": msd.state_dict() }, checkpoint_path_d)

                            if _is_new_best:
                                symlink = Path(self.opts.dirname + "/g_best.pt")
                                if symlink.is_symlink():
                                    symlink.unlink()
                                symlink.symlink_to(os.path.basename(checkpoint_path_g))

                            max_last = 10
                            keep_every = self.opts.checkpoint_interval*10
                            rm_step = steps - (self.opts.checkpoint_interval*max_last)
                            mdl = "{}/g_{:08d}.pt".format(self.opts.dirname, rm_step)
                            if os.path.isfile(mdl) and (rm_step) % keep_every !=0 and os.path.basename(os.path.realpath(self.opts.dirname + "/g_best.pt")) != "g_{:08d}.pt".format(rm_step):
                                script_utils.run(["rm", mdl], quit_on_error=False)
                                script_utils.run(["rm", mdl.replace("g_", "d_")], quit_on_error=False)
                                script_utils.run(["rm", mdl.replace("g_", "trainer_")], quit_on_error=False)

                    torch.cuda.empty_cache()
                    generator.train()
                steps += 1

            if optimizer_was_run:
                scheduler_g.step()
                scheduler_d.step()
            if self.opts.rank == 0:
                utils.feature_extractor_decorator.merge_cache(generator, self.opts.cache_path)
        logging.info("Finished training")
