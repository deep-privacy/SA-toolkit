#!/usr/bin/env python3

description = """
  This script trains asv-sidekit models.
  It takes a config file
"""

import argparse
import concurrent
import configparser
import datetime
import glob
import logging
import os
import sys
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

import satools

logging.basicConfig(level=logging.INFO, format="satools %(levelname)s: %(message)s")
logging.getLogger("geocoder").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

from dataclasses import dataclass
from carbontracker.tracker import CarbonTracker

@dataclass
class Opts:
    # for cmd
    cpu_cmd: str = "./utils/run.pl"
    cuda_cmd: str = "./utils/run.pl"

    n_gpu:int = 1

    model_file: Path = "./local/chain/aaaa.py"
    model_args: str = ""  # allow empty
    exp_dir: Path = "./exp/"
    dirname: str = "model_a"
    init_weight_model: Path = ""
    train_epoch: str = "0"

    checkpoint_interval: int = 1000 # in step
    training_epochs: int = 1000
    segment_size: int = 48000
    overlap: int = -1
    minibatch_size: int = 8
    examples_per_speaker_in_batch: str = "batch_size/16"
    samples_per_speaker_in_epoch: str = "num_spk/5"
    dev_ratio: float = 0.02
    num_worker_dataloader: int = 4
    test_set: Path = "./data/part"
    train_set: Path = "./data/part"
    final_model: str = ""
    compute_test_set_eer: str= "false"
    logging_interval:int = 20
    patience:int = 100
    augmentation: str = "" # allow empty
    optim: str = ""

    @property
    def dir(self):
        return Path(self.exp_dir) / self.dirname

    @property
    def get_model_args(self):
        if self.model_args == "":
            return []
        return satools.utils.fix_json(self.model_args)

    def get_forcmd(self, key):
        if getattr(self, key) == "":
            return []
        return [f"--{key.replace('_', '-')}", str(getattr(self, key)).replace("\n", "")]

    def load_from_config(self, cfg):
        for key, value in cfg.items():
            if hasattr(self, key):
                type_of_value = self.__annotations__[key]
                setattr(self, key, type_of_value(value))
                #  print("SET:", key, type_of_value, value)
        return self


def train():
    parser = argparse.ArgumentParser(description="ASV model training script")
    parser.add_argument("--stage", default=0, type=int)
    parser.add_argument("--config", default="configs/default")
    args = parser.parse_args()

    logging.info("Reading config")
    if not Path(args.config).is_file():
        logging.error(f"Config '{args.config}' not found")
        quit(1)
    cfg_parse = configparser.ConfigParser()
    cfg_parse.read(args.config)
    cfg_parse = satools.script_utils.vartoml(cfg_parse)

    cfg_cmd = Opts().load_from_config(cfg_parse["cmd"])
    cfg_exp = Opts().load_from_config(cfg_parse["exp"])

    stage = args.stage

    os.makedirs(cfg_exp.dir, exist_ok=True)

    if cfg_exp.train_epoch == "last" or cfg_exp.train_epoch == '"last"':
        pattern = cfg_exp.dir / "*.pt"
        cp_list = glob.glob(str(pattern))
        cp_list = list(map(lambda x: x.split("/")[-1].split(".")[0], cp_list))
        cp_list = [filename for filename in cp_list if 'trainer' not in filename]
        if "final" in cp_list:
            cp_list.remove("final")
        if "best" in cp_list:
            cp_list.remove("best")
        if len(cp_list) == 0 or (len(cp_list) == 1 and cp_list[0] == "0") or int(stage) > 6:
            cfg_exp.train_epoch = "0"
        else:
            cp_list = list(map(lambda x: int(x), cp_list))
            cfg_exp.train_epoch = str(sorted(cp_list)[-1])
            logging.info(f"Last training iter found: {cfg_exp.train_epoch}")

    if stage <= 4 and cfg_exp.train_epoch == "0":
        # create sidekit train dataset
        logging.info("Create egs csv")
        satools.script_utils.run([
            cfg_cmd.cpu_cmd,
            cfg_exp.dir / "log" / "create_train_csv.log",
            "local/create_train_csv_from_kaldi.py",
            "--kaldi-data", f"{cfg_exp.train_set}",
            "--out-csv", cfg_exp.dir / "train.csv"
        ])

        os.makedirs(cfg_exp.dir / os.path.basename(cfg_exp.test_set), exist_ok=True)
        satools.script_utils.run([
            "cp", f"{cfg_exp.test_set}/trials", f"{cfg_exp.test_set}/trials.wav.scp", f"{cfg_exp.test_set}/enroll.wav.scp", f"{cfg_exp.test_set}/enroll.utt2spk", cfg_exp.dir / os.path.basename(cfg_exp.test_set),
        ])

    carbonTracker = CarbonTracker(epochs=1, components="gpu", verbose=2)
    carbonTracker.epoch_start()

    if stage <= 5 and cfg_exp.train_epoch == "0":
        logging.info("Initializing model")
        process_out = subprocess.run([
            cfg_cmd.cpu_cmd,
            cfg_exp.dir / "log" / "init.log",
            cfg_exp.model_file,
            *cfg_exp.get_model_args,
            "--mode", "init",
            "--train-set", cfg_exp.dir / "train.csv",
            *cfg_exp.get_forcmd("dir"),
            *cfg_exp.get_forcmd("init_weight_model"),
            cfg_exp.dir / "0.pt",
        ])
        if process_out.returncode != 0:
            quit(process_out.returncode)

    if stage <= 6:

        # The official documentation for run.pl, queue.pl, slurm.pl, and ssh.pl:
        #   "Parallelization in Kaldi": http://kaldi-asr.org/doc/queue.html
        # =========================================================~
        if cfg_cmd.cuda_cmd == "./utils/run.pl" and torch.cuda.device_count() == 0:
            logging.error(f"Training requires a gpus, if you are on a gid you can use queue.pl, slurm.pl or ssh.pl cmd job unified interfaces")
            logging.error(f"Or connect yourself to a node before running this file (run.pl)")
            quit(1)

        logging.info(f"Starting training from iter={cfg_exp.train_epoch}")

        python_cmd = ["python3"]
        if cfg_exp.n_gpu != 1:
            #  TODO add support for other cfg_cmd.cuda_cmd than run.pl (ssh.pl with multi nnodes)
            python_cmd = ["OMP_NUM_THREADS=1", "torchrun", "--standalone", "--nnodes=1", "--nproc_per_node", f"{cfg_exp.n_gpu}"]

        f = f"{cfg_exp.dir}/log/train.log"
        if os.path.exists(f):
            a = open(f, "w");a.seek(0);a.truncate()
        tail = subprocess.Popen(f"tail -F {cfg_exp.dir}/log/train.log", stderr=subprocess.PIPE, shell=True)
        satools.script_utils.run([
            cfg_cmd.cuda_cmd,
            f"{cfg_exp.dir}/log/train.log",
            *python_cmd,
            cfg_exp.model_file,
            *cfg_exp.get_model_args,
            "--mode", "train",
            *cfg_exp.get_forcmd("train_set"),
            "--test-set", cfg_exp.dir / os.path.basename(cfg_exp.test_set),
            *cfg_exp.get_forcmd("dev_ratio"),
            *cfg_exp.get_forcmd("compute_test_set_eer"),
            *cfg_exp.get_forcmd("patience"),
            *cfg_exp.get_forcmd("augmentation"),
            *cfg_exp.get_forcmd("optim"),
            *cfg_exp.get_forcmd("overlap"),
            *cfg_exp.get_forcmd("examples_per_speaker_in_batch"),
            *cfg_exp.get_forcmd("samples_per_speaker_in_epoch"),
            *cfg_exp.get_forcmd("num_worker_dataloader"),
            *cfg_exp.get_forcmd("dir"),
            *cfg_exp.get_forcmd("minibatch_size"),
            *cfg_exp.get_forcmd("logging_interval"),
            *cfg_exp.get_forcmd("segment_size"),
            *cfg_exp.get_forcmd("training_epochs"),
            *cfg_exp.get_forcmd("checkpoint_interval"),
            cfg_exp.dir / f"{cfg_exp.train_epoch}.pt",
        ], on_error=lambda x: tail.kill())
        tail.kill()


    if stage <= 5:
        carbonTracker.epoch_end()
        carbonTracker.stop()

    if stage <= 10:
        logging.info(f"Creating JIT model from '{cfg_exp.final_model}'")
        shutil.copy(cfg_exp.dir / f"{cfg_exp.final_model}", cfg_exp.dir / "final.pt")
        satools.script_utils.run([
            cfg_cmd.cpu_cmd,
            cfg_exp.dir / "log" / "jit.log",
            "SA_JIT_TWEAK=True",
            cfg_exp.model_file,
            *cfg_exp.get_model_args,
            "--mode", "jit_save",
            *cfg_exp.get_forcmd("train_set"),
            *cfg_exp.get_forcmd("dir"),
            "--new-model", cfg_exp.dir / f"final.jit",
            cfg_exp.dir / f"final.pt",
        ]
                                 )


if __name__ == "__main__":
    train()


