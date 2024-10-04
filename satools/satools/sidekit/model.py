import argparse
import pandas
import numpy
import logging
import sys
import re
import csv
import os
import time
import datetime
from dataclasses import dataclass, fields
from pathlib import Path
import json
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group

from .. import script_utils
from .. import utils
from . import dataset
from . import objf
from .monitor import TrainingMonitor
import satools

logging.basicConfig(level=logging.INFO, format="satools %(levelname)s: %(message)s")
logging.getLogger("geocoder").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

@dataclass
class DecodeOpts:
    use_gpu: bool = False
    enrolls_wav_scp: str = "./path/wav.scp"
    trails_wav_scp: str = "./path/wav.scp"
    enroll_utt2spk: str = "./path/utt2spk" # for x-vector mean before scoring
    trials: str = "./path/trails"
    decode_output: str = ""

@dataclass
class DataloadingOpts:
    minibatch_size: int = 32
    overlap: int = -1
    examples_per_speaker_in_batch: str = "batch_size/10"
    samples_per_speaker_in_epoch: str = "num_spk*1"
    num_worker_dataloader: int = 4
    rank: int = 0

@dataclass
class ModelOpts(DataloadingOpts, DecodeOpts):
    mode: str = ""
    base_model: str = ""
    base_model_args: str = "{}"

    init_weight_model: str = ""

    dirname: str = ""
    train_set: str = "./data/part"
    dev_ratio: float = 0.005

    test_set: str = "./data/part"
    compute_test_set_eer: str = "false"

    augmentation: str = "{}"
    optim: str = "{}"
    mixed_precision: str = "true"
    torch_compile: str = "false"

    num_gpus: int = 1
    segment_size: float = 16640
    training_epochs: int = 1500
    patience: int = 50

    checkpoint_interval: int = 1000  # in epoch
    logging_interval: int = 20

    new_model: str = ""

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


#######################
#  ASV sidekit Model  #
#######################


class SidekitModel():
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
        self.reset_num_speakers()
        if self.opts.mode in ["init"]:
            self.init()
        elif self.opts.mode == "jit_save":
            self.jit_save()
        elif self.opts.mode in ["decode", "infer", "eval"]:
            self.infer()
        elif self.opts.mode in ["train", "training"]:
            self.train()
        else:
            logging.critical(f"Mode '{self.opts.mode}' not defined")

    def reset_num_speakers(self):
        self.num_speakers = 1
        train_csv = os.path.join(self.opts.dirname, "train.csv")
        if os.path.exists(train_csv):
            with open(train_csv) as csvfile:
                read = csv.DictReader(csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
                total_speaker_idx = set()
                for i, row in enumerate(read):
                    spk_id = int(row["speaker_idx"])
                    total_speaker_idx.add(spk_id)
                assert max(total_speaker_idx)+1 == len(total_speaker_idx), f"{max(total_speaker_idx)+1} != {len(total_speaker_idx)}"
            self.num_speakers = max(total_speaker_idx)+1
        else:
            self.num_speakers = torch.load(self.opts.base_model, weights_only=False)["base_model_params"]["num_speakers"]

        if self.num_speakers == 1:
            logging.critical(f"Could not find file {train_csv} or key 'base_model_params' in model file to know the number of speaker of the model")
            sys.exit(1)

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
                    "base_model_params": {"num_speakers": self.num_speakers},
                    "base_model_args": utils.fix_json(self.opts.base_model_args),
                    }, file)

    def jit_save(self):
        logging.info("Creating a JIT model for easy sharing")
        file = self.opts.new_model
        model = self.Net(num_speakers=self.num_speakers)
        model.load_state_dict(self.load_state_model(self.opts.base_model))
        self.save_model(model, self.opts.base_model) # re-save old model (update install_path/base_model_args/.. keys)
        model = torch.jit.script(model)
        torch.jit.save(model, file)
        logging.info("Saved to: " + str(file))

    def init(self):
        model = self.Net(num_speakers=self.num_speakers)
        logging.info(str(model))
        if self.opts.init_weight_model != "":
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

        self.save_model(model, self.opts.base_model)


    @torch.no_grad()
    def infer(self):
        device = torch.device("cpu")
        if self.opts.use_gpu:
            device = torch.device("cuda")
            logging.info("Using GPU")

        xtractor = self.Net(num_speakers=self.num_speakers)
        xtractor.load_state_dict(self.load_state_model(self.opts.base_model))
        xtractor = xtractor.to(device)

        os.makedirs(self.opts.dirname, exist_ok=True) # for tqdm
        tqdm_file = open(self.opts.dirname+ "/log/tqdm", "w")

        os.makedirs(self.opts.decode_output, exist_ok=True)
        metrics = objf.test(
            xtractor,
            device,
            self.opts.enrolls_wav_scp,
            self.opts.trails_wav_scp,
            self.opts.enroll_utt2spk,
            self.opts.trials,
            self.opts.decode_output,
            self.opts.mixed_precision.lower()=="true",
            tqdm_file=tqdm_file,
        )
        del metrics["score"]

        tqdm_file.seek(0)
        tqdm_file.truncate()
        with open(self.opts.decode_output + "/metric.json", 'w+') as out:
            json.dump(metrics, out)

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

        net = self.Net(num_speakers=self.num_speakers)
        net.load_state_dict(self.load_state_model(self.opts.base_model))
        net = net.to(device)

        if self.opts.num_gpus > 1:
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
            net = torch.nn.parallel.DistributedDataParallel(
                net, device_ids=[self.opts.rank]
            )
            net = utils.torch.WrappedTorchDDP(net)

        return net, device

    def init_monitor_train_optims(self, optim):
        monitor = TrainingMonitor(
            patience=self.opts.patience,
            best_accuracy=0.0,
            best_eer_epoch=1,
            best_eer=999,
            compute_test_eer=self.opts.compute_test_set_eer.lower()=="true"
        )
        last_epoch = -1
        scheduler_epoch = -1

        f = os.path.dirname(self.opts.base_model) + "/" + "trainer_" + os.path.basename(os.path.realpath(self.opts.base_model))
        if Path(f).is_file():
            logging.info(f"Loading trainer from: {f}")
            sd = torch.load(f, weights_only=False)
            optim.load_state_dict(sd["optim"])
            last_epoch = sd["epoch"]
            scheduler_epoch = sd["scheduler_epoch"]
            monitor = sd["monitor"]

        scheduler = self.get_scheduler(optim, scheduler_epoch)

        if self.opts.rank == 0:
            sw = monitor.add_tensorboard(os.path.join(self.opts.dirname, "runs"))
            handler = utils.LogHandlerSummaryWriter(sw)
            handler.setFormatter(logging.Formatter("`" + logging.root.handlers[0].formatter._fmt + "`"))
            logging.getLogger().addHandler(handler)
            logging.info(sys.argv)


        if last_epoch != -1:
            logging.info(f"Loaded last metrics from trainer:")
            monitor.display(add_to_tensorboard=False)
            monitor.display_final()
            logging.info("")

        return optim, scheduler, monitor, last_epoch

    def get_optim(self, net):
        otpim_dict = utils.fix_json(self.opts.optim)
        _optimizer = eval(otpim_dict["optimizer"]["type"])
        _options = {**otpim_dict["optimizer"]["opts"]}
        if hasattr(net, 'set_lr_weight_decay_layers_for_optim'):
            optimizer = net.set_lr_weight_decay_layers_for_optim(_optimizer, _options)
            logging.info("Set lr weight decay for optim in model definition")
        else:
            optimizer = _optimizer(net.parameters(), **_options)
            logging.info("Set lr weight for optim in traininer")
        return optimizer

    def get_scheduler(self, optimizer, last_epoch):
        otpim_dict = utils.fix_json(self.opts.optim)
        if "last_epoch" in otpim_dict["scheduler"]["opts"]:
            scheduler = eval(otpim_dict["scheduler"]["type"])(optimizer=optimizer, **otpim_dict["scheduler"]["opts"])
        else:
            scheduler = eval(otpim_dict["scheduler"]["type"])(optimizer=optimizer, **otpim_dict["scheduler"]["opts"], last_epoch=last_epoch)
        return scheduler

    def train(self):
        xtractor, device = self.init_cuda_model_distributed()
        optim = self.get_optim(xtractor)
        optim, scheduler, monitor, last_epoch = self.init_monitor_train_optims(optim)

        df = pandas.read_csv(os.path.join(self.opts.dirname, "train.csv"))
        numpy.random.seed(42)
        training_df, validation_df = train_test_split(df, test_size=self.opts.dev_ratio if self.opts.dev_ratio != 0 else 0.005)

         # In case validation_ratio equals to zero, we sample validation from the train set
        if self.opts.dev_ratio == 0:
            training_df = df

        training_set = dataset.SideSet(
            dataset_df=training_df,
            segment_size=self.opts.segment_size,
            set_type="train",
            tqdm_display=self.opts.rank==0,
            overlap=self.opts.overlap,
            augmentation=utils.fix_json(self.opts.augmentation),
        )

        match = re.match(r"batch_size/(\d+)", self.opts.examples_per_speaker_in_batch)
        if self.opts.examples_per_speaker_in_batch == "batch_size":
            examples_per_speaker_in_batch = self.opts.minibatch_size
        elif match:
            examples_per_speaker_in_batch = self.opts.minibatch_size // int(match.group(1))
        else:
            examples_per_speaker_in_batch = int(self.opts.examples_per_speaker_in_batch)

        match = re.match(r"num_spk\*(\d+)", self.opts.samples_per_speaker_in_epoch)
        match_2 = re.match(r"num_spk/(\d+)", self.opts.samples_per_speaker_in_epoch)
        if self.opts.samples_per_speaker_in_epoch == "num_spk":
            samples_per_speaker_in_epoch = self.num_speakers
        elif match:
            samples_per_speaker_in_epoch = self.num_speakers // int(match.group(1))
        elif match_2:
            samples_per_speaker_in_epoch = self.num_speakers * int(match_2.group(1))
        else:
            samples_per_speaker_in_epoch = int(self.opts.samples_per_speaker_in_epoch)

        #  samples_per_speaker_in_epoch=1

        train_sampler = dataset.SideSampler(
            data_source=training_set.sessions['speaker_idx'],
            spk_count=self.num_speakers,
            examples_per_speaker=examples_per_speaker_in_batch,
            samples_per_speaker=samples_per_speaker_in_epoch,
            batch_size=self.opts.minibatch_size,
            rank=self.opts.rank,
            num_process=self.opts.num_gpus,
        )

        training_loader = torch.utils.data.DataLoader(
            training_set,
            batch_size=self.opts.minibatch_size,
            pin_memory=True,
            num_workers=self.opts.num_worker_dataloader,
            worker_init_fn=utils.torch.seed_worker,
            sampler=train_sampler,
        )

        if self.opts.rank == 0:
            validation_set = dataset.SideSet(
                dataset_df=validation_df,
                set_type="validation",
                overlap=0,
            )

            validation_loader = torch.utils.data.DataLoader(
                validation_set,
                batch_size=self.opts.minibatch_size,
                pin_memory=True,
                num_workers=self.opts.num_worker_dataloader,
                worker_init_fn=utils.torch.seed_worker,
                shuffle=False,
            )

        scaler = torch.cuda.amp.GradScaler()

        if self.opts.torch_compile.lower() == "true":
            logging.info(f"torch.compile network..")
            xtractor = torch.compile(xtractor)

        if self.opts.mixed_precision.lower(): logging.info(f"Using mixed_precision")

        logging.info(f"Logging:\n\ttensorboard --logdir {self.opts.dirname}")

        for epoch in range(last_epoch+1, self.opts.training_epochs):

            # Process one epoch and return the current model
            if monitor.current_patience == 0:
                logging.info(f"Stopping at epoch {epoch} for cause of patience")
                break

            train_sampler.set_epoch(epoch)
            monitor.update(epoch=epoch)

            xtractor = objf.train_epoch(
                xtractor,
                self.opts,
                monitor,
                training_loader,
                optim,
                scheduler,
                device,
                scaler=scaler,
                mixed_precision=self.opts.mixed_precision.lower()=="true",
            )

            if self.opts.rank == 0 and epoch % self.opts.checkpoint_interval == 0:

                val_acc, val_loss, val_eer, val_eer_ci_lower, val_eer_ci_upper = objf.validation(
                    xtractor,
                    validation_loader,
                    device,
                    self.opts.mixed_precision.lower()=="true",
                )

                if self.opts.compute_test_set_eer.lower()=="false":
                    monitor.update(val_eer=val_eer,
                                   val_loss=val_loss,
                                   val_acc=val_acc)
                    monitor.display()
                else:
                    metrics = objf.test(
                        xtractor,
                        device,
                        "{}/enroll.wav.scp".format(self.opts.test_set),
                        "{}/trials.wav.scp".format(self.opts.test_set),
                        "{}/enroll.utt2spk".format(self.opts.test_set),
                        "{}/trials".format(self.opts.test_set),
                        self.opts.test_set,
                        self.opts.mixed_precision.lower()=="true",
                    )

                    monitor.update(test_eer=min(metrics["eer"], metrics["asnorm"]["eer"]),
                                   test_metric=metrics,
                                   val_eer=val_eer,
                                   val_loss=val_loss,
                                   val_acc=val_acc)
                    monitor.display()

                    checkpoint_path = "{}/{}.pt".format(self.opts.dirname, epoch+1)
                    checkpoint_path_trainer = "{}/trainer_{}.pt".format(self.opts.dirname, epoch+1)

                    self.save_model(xtractor, checkpoint_path)
                    torch.save({"optim": optim.state_dict(),"epoch": epoch, "scheduler_epoch": scheduler.last_epoch,  "monitor": monitor}, checkpoint_path_trainer)

                    if monitor.is_best:
                        symlink = Path(self.opts.dirname + "/best.pt")
                        if symlink.is_symlink():
                            symlink.unlink()
                        symlink.symlink_to(os.path.basename(checkpoint_path))

                    max_last = 10
                    keep_every = self.opts.checkpoint_interval*10
                    rm_epoch = epoch - (self.opts.checkpoint_interval*max_last)
                    mdl = "{}/{}.pt".format(self.opts.dirname, rm_epoch)
                    if os.path.isfile(mdl) and (rm_epoch) % keep_every !=0 and os.path.basename(os.path.realpath(self.opts.dirname + "/best.pt")) != "{}.pt".format(rm_epoch):
                        script_utils.run(["rm", mdl], quit_on_error=False)
                        script_utils.run(["rm", mdl.replace(f"{rm_epoch}.pt", f"trainer_{rm_epoch}.pt")], quit_on_error=False)

                torch.cuda.empty_cache()

        monitor.display_final()
        logging.info("Finished training")
        if self.opts.num_gpus > 1:
            destroy_process_group()

