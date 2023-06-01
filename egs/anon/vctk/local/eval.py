#!/usr/bin/env python3

description = """
  This script evals anon/clear models/speech.
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
from itertools import chain
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

    exp_dir: Path = "./exp/"
    dirname: str = "eval_a"
    gpu: str = "True"

    asr_num_jobs: int = 3
    asr_test_set: Path = "./test"
    asr_suffix: str = ""
    asr_model: Path = "./exp/asr_something/final.pt"
    #
    asr_graph_dir: Path  = "./graph_tgsmall" # where the HCLG.fst
    asr_lang_lp_tgsmall: Path = "./data/lang_lp_test_tgsmall" # G.fst
    asr_lang_lp_fg_large: Path = "./data/lang_lp_test_fglarge" # G.fst

    asv_test_enrolls_set: Path = "./test"
    asv_test_trials_set: Path = "./test"
    asv_suffix: str = ""
    asv_model: Path = "./exp/asv_something/final.pt"


    @property
    def dir(self):
        return Path(self.exp_dir) / self.dirname

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

def print_asr_metrics(out_dir, no_rescoring=False, rescoring=False):
    if no_rescoring:
        logging.info(f"Printing best WER without rescoring {out_dir}...")
        satools.script_utils.run([ "cat", "{}/wer*".format(out_dir), "|", "utils/best_wer.sh", ">", "{}/best_wer".format(out_dir) ], shell=True)
        logging.info(" " + satools.script_utils.read_single_param_file("{}/best_wer".format(out_dir), typename=str))

    if rescoring:
        logging.info(f"Printing best WER with rescoring {out_dir}_fg...")
        satools.script_utils.run([ "cat", "{}_fg/wer*".format(out_dir), "|", "utils/best_wer.sh", ">", "{}_fg/best_wer".format(out_dir)], shell=True)
        logging.info(" " + satools.script_utils.read_single_param_file( f"{out_dir}_fg/best_wer", typename=str))


def print_asv_metrics(out_dir):
    logging.info(f"Printing ASV metrics...")
    with open(out_dir / "metric.json", 'r') as m:
        metrics = satools.utils.fix_json(m.read())
    eer_std = round((metrics['eer_upper'] - metrics['eer_lower'])/2, 3)
    logging.info(f" %EER: {round(metrics['eer'], 3)} ± {eer_std}, Min Cllr: {round(metrics['min_cllr'], 3)}, linkability: {round(metrics['linkability'], 3)}")
    logging.info(f"Printing ASV AS-NORM metrics...")
    metrics = metrics["asnorm"]
    eer_std = round((metrics['eer_upper'] - metrics['eer_lower'])/2, 3)
    logging.info(f" %EER: {round(metrics['eer'], 3)} ± {eer_std}, Min Cllr: {round(metrics['min_cllr'], 3)}, linkability: {round(metrics['linkability'], 3)}")

def eval():
    parser = argparse.ArgumentParser(description="Eval script")
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


    """
    ASR eval
    """

    for test_set in str(cfg_exp.asr_test_set).split(","):
        test_set = Path(test_set)
        data_name = os.path.basename(test_set)
        decode_suff = "_{}_iter{}".format(os.path.basename(os.path.dirname(cfg_exp.asr_model))+"_"+os.path.basename(cfg_exp.asr_model).replace(".pt", ""), cfg_exp.asr_suffix)
        out_dir = cfg_exp.dir / f"asr_decode_{data_name}{decode_suff}"

        if stage <= 1:
            num_jobs = satools.utils.split_data(test_set, cfg_exp.asr_num_jobs)
            logging.info(f"Decoding '{data_name}' with  {cfg_exp.asr_num_jobs} jobs...")

            gpu_opts = []
            if bool(cfg_exp.gpu):
                gpu_opts = ["--use-gpu", "True", "--gpu-id", "JOB"]

            feats_scp = f"{test_set}/split{num_jobs}/JOB/wav.scp"

            asr_model_state = torch.load(cfg_exp.asr_model)
            config_path = asr_model_state["install_path"] + "/" + asr_model_state["task_path"] + "/" + asr_model_state["base_model_path"]
            logging.info(f"Loading {config_path} with args \"{str(asr_model_state['base_model_args'])}\" and params \"{str(asr_model_state['base_model_params'])}\"" )

            tqdm = subprocess.Popen(f"exec tail -F {cfg_exp.dir}/log/tqdm", stderr=subprocess.PIPE, shell=True)
            satools.script_utils.run([
                    cfg_cmd.cpu_cmd if bool(cfg_exp.gpu) else cfg_cmd.cpu_cmd,
                    f"JOB=1:{num_jobs}",
                    out_dir / "log" / "decode.JOB.log",
                    config_path,
                    *list(chain(*[[f"--{key.replace('_', '-')}", value] for key, value in asr_model_state['base_model_args'].items()])),
                    "--mode", "decode",
                     *cfg_exp.get_forcmd("dir"),
                    *gpu_opts,
                    "--decode-feats", feats_scp,
                    f"{cfg_exp.asr_model}",
                    "|",
                    "shutil/decode/latgen-faster-mapped.sh",
                    cfg_exp.asr_graph_dir / "words.txt",
                    os.path.dirname(cfg_exp.asr_model) + "/0.trans_mdl",
                    f"{cfg_exp.asr_graph_dir}/HCLG.fst",
                    out_dir / "lat.JOB.gz",
            ])
            tqdm.kill()
            print("", file=sys.stderr)
            satools.script_utils.write_single_param_file(num_jobs, out_dir / "num_jobs")


        if stage <= 2:
            logging.info(f"Scoring...")
            satools.script_utils.run(["local/score.sh", "--cmd", cfg_cmd.cpu_cmd, test_set, cfg_exp.asr_graph_dir, out_dir])

            print_asr_metrics(out_dir, no_rescoring=True)

            logging.info(f"Rescore with a N gram LM...")
            satools.script_utils.run([
                    "steps/lmrescore_const_arpa.sh",
                    "--cmd", cfg_cmd.cpu_cmd,
                    cfg_exp.asr_lang_lp_tgsmall,
                    cfg_exp.asr_lang_lp_fg_large,
                    test_set,
                    out_dir,
                    f"{out_dir}_fg",
            ])
            print_asr_metrics(out_dir, rescoring=True)

            logging.info(f"Computing WER details for {out_dir}_fg...")
            satools.script_utils.run([
                    "./shutil/decode/wer_detail.sh",
                    "--cmd", cfg_cmd.cpu_cmd,
                    "--dataDir", test_set,
                    "--decodeDir", f"{out_dir}_fg",
                    "--langDir", cfg_exp.asr_lang_lp_fg_large,
                ], shell=True,
            )

    if stage <= 3:

        asv_model_state = torch.load(cfg_exp.asv_model)
        config_path = asv_model_state["install_path"] + "/" + asv_model_state["task_path"] + "/" + asv_model_state["base_model_path"]
        logging.info(f"Loading {config_path} with args \"{str(asv_model_state['base_model_args'])}\" and params \"{str(asv_model_state['base_model_params'])}\"" )

        gpu_opts = []
        if bool(cfg_exp.gpu):
            gpu_opts = ["--use-gpu", "True"]

        python_cmd = ["python3"]

        decode_suff = "_{}_iter{}".format(os.path.basename(os.path.dirname(cfg_exp.asv_model))+"_"+os.path.basename(cfg_exp.asv_model).replace(".pt", ""), cfg_exp.asv_suffix)
        out_dir = cfg_exp.dir / f"asv_decode_{data_name}{decode_suff}"

        tqdm = subprocess.Popen(f"exec tail -F {cfg_exp.dir}/log/tqdm", stderr=subprocess.PIPE, shell=True)
        satools.script_utils.run([
                cfg_cmd.cpu_cmd if bool(cfg_exp.gpu) else cfg_cmd.cpu_cmd,
                f"{out_dir}/log/asv.log",
                *python_cmd,
                config_path,
                *list(chain(*[[f"--{key.replace('_', '-')}", value] for key, value in asv_model_state['base_model_args'].items()])),
                "--mode", "eval",
                *gpu_opts,
                *cfg_exp.get_forcmd("dir"),
                "--enrolls-wav-scp", f"{cfg_exp.asv_test_enrolls_set}/wav.scp",
                "--trails-wav-scp" , f"{cfg_exp.asv_test_trials_set}/wav.scp",
                "--enroll-utt2spk" , f"{cfg_exp.asv_test_enrolls_set}/utt2spk", # for x-vector mean before scoring
                "--trials" , f"{cfg_exp.asv_test_trials_set}/trials",
                "--decode-output", out_dir,
                f"{cfg_exp.asv_model}",
            ]
        )
        tqdm.kill()
        print("", file=sys.stderr)
        print_asv_metrics(out_dir)

    if stage >= 10:
        for test_set in str(cfg_exp.asr_test_set).split(","):
            test_set = Path(test_set)
            data_name = os.path.basename(test_set)
            decode_suff = "_{}_iter{}".format(os.path.basename(os.path.dirname(cfg_exp.asr_model))+"_"+os.path.basename(cfg_exp.asr_model).replace(".pt", ""), cfg_exp.asr_suffix)
            out_dir = cfg_exp.dir / f"asr_decode_{data_name}{decode_suff}"
            print_asr_metrics(out_dir, no_rescoring=True, rescoring=True)
        decode_suff = "_{}_iter{}".format(os.path.basename(os.path.dirname(cfg_exp.asv_model))+"_"+os.path.basename(cfg_exp.asv_model).replace(".pt", ""), cfg_exp.asv_suffix)
        out_dir = cfg_exp.dir / f"asv_decode_{data_name}{decode_suff}"
        print_asv_metrics(out_dir)





if __name__ == "__main__":
    eval()
