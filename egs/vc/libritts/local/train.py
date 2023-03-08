#!/usr/bin/env python3

description = """
  This script trains and hifigans models.
  It takes a config file
"""

import argparse
import concurrent
import configparser
import datetime
import glob
import json
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

    model_file: Path = "./local/chain/tuning/aaaa.py"
    model_args: str = ""  # allow empty
    exp_dir: Path = "./exp/"
    dirname: str = "model_a"
    init_weight_model: Path = ""
    cold_restart: str = "False"
    train_stage: int = 0
    final_model: str = ""

    checkpoint_interval: int = 1000 # in step
    training_epochs: int = 1000
    segment_size: int = 16640
    lr_decay: float = 0.999
    lr: float = 0.0002
    minibatch_size: int = 8
    num_worker: int = 4
    dev_set: Path = "./data/part"
    train_set: Path = "./data/part"
    dev_num_uttr_per_spk: int = 10

    @property
    def dir(self):
        return Path(self.exp_dir) / self.dirname

    @property
    def get_model_args(self):
        if self.model_args == "":
            return []
        return json.loads(self.model_args)

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


def cuda_env(cfg_cmd, cfg_exp, job_id):
    """
    for parallel training
    """
    if "run.pl" not in cfg_cmd.cuda_cmd:
        return []
    cuda_device = job_id - 1
    cuda_device %= cfg_exp.max_concurrent_jobs
    cuda_device %= torch.cuda.device_count()
    return ["env", "CUDA_VISIBLE_DEVICES=" + str(cuda_device)]


def run_job(cmd):
    """
    sub a single run job and let ThreadPoolExecutor monitor its progress
    """
    #  print(cmd, flush=True) # DEBUG
    process_out = subprocess.run(cmd)
    return process_out.returncode


def train():
    parser = argparse.ArgumentParser(description="Acoustic model training script")
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


    if cfg_exp.train_stage == "last" or cfg_exp.train_stage == '"last"':
        pattern = os.path.join(cfg_exp.dir, "g_" + "????????")
        cp_list = glob.glob(pattern)
        if len(cp_list) != 0:
            cfg_exp.train_stage = sorted(cp_list)[-1]
            logging.info(f"Last training iter found: {cfg_exp.train_stage}")
        else:
            cfg_exp.train_stage = ""

    carbonTracker = CarbonTracker(epochs=1, components="gpu", verbose=2)
    carbonTracker.epoch_start()

    #   start the training
    if stage <= 5 and cfg_exp.train_stage == 0:
        logging.info("Initializing model")
        process_out = subprocess.run([
                cfg_cmd.cpu_cmd,
                cfg_exp.dir / "log" / "init.log",
                cfg_exp.model_file,
                *cfg_exp.get_model_args,
                "--mode", "init",
                *cfg_exp.get_forcmd("dir"),
                *cfg_exp.get_forcmd("init_weight_model"),
                cfg_exp.dir / "g_0.pt",
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


        logging.info(f"Starting training from iter={cfg_exp.train_stage}")

        a = open(f"{cfg_exp.dir}/log/train.log", "w");a.seek(0);a.truncate()
        tail = subprocess.Popen(f"tail -F {cfg_exp.dir}/log/train.log", stderr=subprocess.PIPE, shell=True)
        satools.script_utils.run([
                cfg_cmd.cuda_cmd,
                f"{cfg_exp.dir}/log/train.log",
                 cfg_exp.model_file,
                 *cfg_exp.get_model_args,
                "--mode", "train",
                *cfg_exp.get_forcmd("train_set"),
                *cfg_exp.get_forcmd("dev_set"),
                cfg_exp.dir / f"g_{cfg_exp.train_stage}.pt",
            ], shell=True, on_error=lambda x: tail.kill()
        )


    if stage <= 5:
        carbonTracker.epoch_end()
        carbonTracker.stop()

    if stage <= 10:
        logging.info(f"Creating JIT model from '{cfg_exp.final_model}.pt'")
        shutil.copy(cfg_exp.dir / f"{cfg_exp.final_model}.pt", cfg_exp.dir / "final.pt")
        satools.script_utils.run([
                cfg_cmd.cpu_cmd,
                cfg_exp.dir / "log" / "jit.log",
                 cfg_exp.model_file,
                 *cfg_exp.get_model_args,
                "--mode", "jit_save",
                 *cfg_exp.get_forcmd("dir"),
                "--new-model", cfg_exp.dir / f"final.jit",
                cfg_exp.dir / f"final.pt",
            ]
        )


if __name__ == "__main__":
    train()

