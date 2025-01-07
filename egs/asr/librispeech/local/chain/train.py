#!/usr/bin/env python3

description = """
  This script trains and tests chain models.
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

from dataclasses import dataclass
from carbontracker.tracker import CarbonTracker

logging.basicConfig(level=logging.INFO, format="satools %(levelname)s: %(message)s")
logging.getLogger("geocoder").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Define the maximum number of retries per step on fail
MAX_RETRIES = 10

@dataclass
class Opts:
    # for cmd
    cpu_cmd: str = "./utils/run.pl"
    cuda_cmd: str = "./utils/run.pl"

    # for training exp
    num_archives_multiplier_factor: int = 3 # as in kaldi 3x for frame_subsampling_factor
    checkpoint_interval: int = 100
    diagnostics_interval: int = 10
    dirname: str = "model_a"
    exp_dir: Path = "./exp/chain/"
    egs_dir: Path = "./egs"
    final_combination_n_model: int = 5
    grad_acc_steps: int = 1
    init_weight_model: str = "" # allow empty
    weight_decay_l2_regularize_factor: str = "0.01"
    lr_final: float = 0.001
    lr_initial: float = 0.01
    minibatch_size: int = 4
    model_file: Path = "./local/chain/tuning/aaaa.py"
    model_args: str = ""  # allow empty
    num_epochs: int = 5
    num_jobs_final: int = 6
    num_jobs_initial: int = 1
    # BucketBatch, BucketBatchSuffle, BucketBatchSuffleAllowSomePadding or Random
    sampler: str = "BucketBatch"
    train_set: Path = "./train_clean"
    train_stage: str = "0"
    tree_dir: Path = "./e2e_biphone_tree"
    xent_regularize: float = 0.025 # LF-MMI
    l2_regularize: float = 1e-4 # LF-MMI
     # for node that have less than num_jobs_final, wait for one job to finish before starting a new one
    max_concurrent_jobs: int = 99
    augmentation: str = "" # allow empty

     # for decoding
    gpu: str = "True"
    num_jobs: int = 3
    suffix: str = ""
    decode_iter: str = "final"
    test_set: Path = "./train_clean"
    graph_dir: Path  = "./graph_tgsmall" # where the HCLG.fst
    lang_lp_tgsmall: Path = "./data/lang_lp_test_tgsmall" # G.fst
    lang_lp_fg_large: Path = "./data/lang_lp_test_fglarge" # G.fst

    @property
    def dir(self):
        return Path(self.exp_dir) / self.dirname

    @property
    def data(self):
        return self.train_set

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


def run_job_with_retry(job_args, retries=1, max_retries=MAX_RETRIES):
    """ Run a job and retry if it fails, up to max_retries times """
    result = run_job(job_args)

    if result == 0:
        return 0  # Success
    elif retries < max_retries:
        logging.warning(f"Job {job_args[1]} failed with exit code {result}. Retry {retries + 1}/{max_retries}.")
        return run_job_with_retry(job_args, retries=retries + 1)
    else:
        logging.critical(f"Job {job_args[1]} failed after {max_retries} retries.")
        return result


def submit_diagnostic_jobs(cfg_cmd, cfg_exp, iter_no, args):
    job_pool = []
    with ThreadPoolExecutor(max_workers=max(1, torch.cuda.device_count())) as executor:
        for diagnostic_name in ["train_diagnostic", "valid"]:

            if args.skip_train_diagnostic == "yes" and diagnostic_name == "train_diagnostic":
                continue
            logging.info(f"Submitting diagnostic {diagnostic_name}")
            d = [] if "run.pl" not in cfg_cmd.cuda_cmd else ["env", "CUDA_VISIBLE_DEVICES=" + str([ i for i, value in enumerate(["valid", "train_diagnostic"]) if value == diagnostic_name ][0] if torch.cuda.device_count() > 1 else 0)]
            p = executor.submit(run_job,[
                cfg_cmd.cuda_cmd,
                f"{cfg_exp.dir}/log/compute_prob_{diagnostic_name}.{iter_no}.log",
                *d,
                cfg_exp.model_file,
                *cfg_exp.get_model_args,
                "--mode", "diagnostic",
                *cfg_exp.get_forcmd("dir"),
                *cfg_exp.get_forcmd("data"),
                "--egs", cfg_exp.egs_dir / f"fst_{diagnostic_name}.scp",
                cfg_exp.dir / f"{iter_no}.pt"
            ])
            job_pool.append(p)
            #  print("Pool size:", job_pool, flush=True)
    return job_pool


def train():
    parser = argparse.ArgumentParser(description="Acoustic model training script")
    parser.add_argument("--decode-iter", default="_")
    parser.add_argument("--stage", default=0, type=int)
    parser.add_argument("--config", default="configs/default")
    parser.add_argument("--skip-train-diagnostic", default="no")
    parser.add_argument("--upload", default="no")
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
    cfg_decode = Opts().load_from_config(cfg_parse["test"])

    stage = args.stage

    os.makedirs(cfg_exp.dir, exist_ok=True)

    if cfg_exp.train_stage == "last" or cfg_exp.train_stage == '"last"':
        pattern = cfg_exp.dir / "*.pt"
        cp_list = glob.glob(str(pattern))
        cp_list = list(map(lambda x: x.split("/")[-1].split(".")[0], cp_list))
        cp_list = list(filter(str.isdigit, cp_list))
        if len(cp_list) == 0 or (len(cp_list) == 1 and cp_list[0] == "0") or int(stage) > 6:
            cfg_exp.train_stage = "0"
        else:
            cp_list = list(map(lambda x: int(x), cp_list))
            cfg_exp.train_stage = str(sorted(cp_list)[-1])
            logging.info(f"Last training iter found: {cfg_exp.train_stage}")
    cfg_exp.train_stage = int(cfg_exp.train_stage)

    logging.info(cfg_exp.tree_dir)
    if not os.path.isfile(cfg_exp.dir / "tree"):
        shutil.copy(cfg_exp.tree_dir / "tree", cfg_exp.dir)

    if not os.path.isfile(cfg_exp.dir / "num_pdfs"):
        logging.info(f"Creating num_pdfs file in {cfg_exp.dir}")
        num_pdfs = (
            subprocess.check_output(["tree-info", cfg_exp.tree_dir / "tree"])
            .decode()
            .strip()
            .split("\n")[0]
            .split()[1]
        )
        satools.script_utils.write_single_param_file(num_pdfs, cfg_exp.dir / "num_pdfs")

    if not os.path.isfile(cfg_exp.dir / "0.trans_mdl"):
        satools.script_utils.run([
                "copy-transition-model",
                cfg_exp.tree_dir / "final.mdl",
                cfg_exp.dir / "0.trans_mdl",
        ])
    #   copy from the egs folder
    if not os.path.isfile(cfg_exp.dir / "den.fst"):
        satools.script_utils.copy_folder(cfg_exp.egs_dir / ".." / "den.fst", cfg_exp.dir / "den.fst")
    if not os.path.isfile(cfg_exp.dir / "normalization.fst"):
        satools.script_utils.copy_folder(cfg_exp.egs_dir / ".." / "normalization.fst", cfg_exp.dir / "normalization.fst")

    # we start training with
    num_archives = satools.script_utils.get_egs_info(cfg_exp.egs_dir)
    num_epochs = cfg_exp.num_epochs
    # we don't use num of jobs b/c it is 1 for now
    num_archives_to_process = num_archives * num_epochs * cfg_exp.num_archives_multiplier_factor
    num_iters = (num_archives_to_process * 2) // (
        cfg_exp.num_jobs_initial + cfg_exp.num_jobs_final
    )
    num_iters_last_epoch = (
        num_archives * (num_epochs - 1) * cfg_exp.num_archives_multiplier_factor * 2
    ) // (cfg_exp.num_jobs_initial + cfg_exp.num_jobs_final)

    carbonTracker = CarbonTracker(epochs=1, components="gpu", verbose=1)
    carbonTracker.logger.logger_err.setLevel(logging.ERROR)
    carbonTracker.logger.logger.setLevel(logging.ERROR)
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


        logging.info(f"Iter num_archives_to_process={num_archives_to_process}, num_archives={num_archives}, num_archives_multiplier_factor={cfg_exp.num_archives_multiplier_factor}, num_epochs={num_epochs}")
        logging.info(f"Starting training from iter={cfg_exp.train_stage}")
        logging.info(f"Watch logs with:\n  tail -F {cfg_exp.dir}/log/train.{{0..{num_iters}}}.{{1..{cfg_exp.num_jobs_final}}}.log {cfg_exp.dir}/log/init.log {cfg_exp.dir}/log/compute_prob_valid.{{1..{num_iters}}}.log | ./shutil/grcat ./shutil/conf.log")
        logging.info("Open tensorbord with:\n  tensorboard --logdir_spec $(find $(pwd) -name 'runs' | awk  '{split($NF,a,\"/exp/chain/\"); split(a[2],b,\"/\"); print b[1]\":\"$0}' | tr \"\\n\" \",\" |  sed 's/,$/ /g')")
        logging.info("  configure regex color to: ([0-9]{4}-[0-9]{2}-[0-9]{2} ([0-1]?[0-9]|2[0-3]):[0-5][0-9])")
        num_archives_processed = 0
        for iter_no in range(0, num_iters):
            num_jobs = satools.script_utils.get_current_num_jobs(
                iter_no,
                num_iters,
                cfg_exp.num_jobs_initial,
                1,  # we don't play with num-jobs-step
                cfg_exp.num_jobs_final,
            )
            if iter_no < cfg_exp.train_stage:
                num_archives_processed += num_jobs
                continue
            assert num_jobs > 0
            lr = satools.script_utils.get_learning_rate(
                iter_no,
                num_jobs,
                num_iters,
                num_archives_processed,
                num_archives_to_process,
                cfg_exp.lr_initial,
                cfg_exp.lr_final,
                schedule_type="exponential",
            )
            if (
                iter_no == 1
                or (iter_no % cfg_exp.diagnostics_interval == 0 and iter_no != 0)
                or (iter_no + 1 == num_iters)
            ):
                diagnostic_job_pool = submit_diagnostic_jobs(cfg_cmd, cfg_exp, iter_no, args)
                for p in as_completed(diagnostic_job_pool):
                    if p.result() != 0:
                        quit(p.result())

            logging.info("{} Running iter: {} of {} with {} jobs and lr: {:.6f}".format(
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    iter_no,
                    num_iters,
                    num_jobs,
                    lr,
            ))
            with ThreadPoolExecutor(max_workers=cfg_exp.max_concurrent_jobs) as executor:

                if cfg_exp.weight_decay_l2_regularize_factor == "1.0/num_jobs":
                    cfg_exp.weight_decay_l2_regularize_factor = str(1.0/num_jobs)

                job_pool = []
                job_pool_args = []
                for job_id in range(1, num_jobs + 1):
                    job_args = [
                        cfg_cmd.cuda_cmd,
                        f"JOB={job_id}",
                        f"{cfg_exp.dir}/log/train.{iter_no}.JOB.log",
                        *cuda_env(cfg_cmd, cfg_exp, job_id),
                        cfg_exp.model_file,
                        *cfg_exp.get_model_args,
                        "--mode", "training",
                        *cfg_exp.get_forcmd("dir"),
                        "--lr", str(lr),
                        "--egs", f"{cfg_exp.egs_dir}/fst_train.{num_archives_processed % num_archives + 1}.scp",
                        *cfg_exp.get_forcmd("minibatch_size"),
                        *cfg_exp.get_forcmd("augmentation"),
                        *cfg_exp.get_forcmd("weight_decay_l2_regularize_factor"),
                        *cfg_exp.get_forcmd("l2_regularize"),
                        *cfg_exp.get_forcmd("xent_regularize"),
                        *cfg_exp.get_forcmd("sampler"),
                        *cfg_exp.get_forcmd("grad_acc_steps"),
                        *cfg_exp.get_forcmd("data"),
                        "--num-iter", str(num_iters),
                        "--new-model", cfg_exp.dir / f"{iter_no}.{job_id}.pt",
                        cfg_exp.dir / f"{iter_no}.pt"
                    ]
                    p = executor.submit(run_job, job_args)
                    num_archives_processed += 1
                    job_pool.append(p)
                    job_pool_args.append(job_args)

                     # max_concurrent_jobs wait for all task in the pool to complete
                    if len(job_pool) == cfg_exp.max_concurrent_jobs:
                        for p in as_completed(job_pool):
                            if p.result() != 0:
                                logging.critical(f"Job failed with exit code {p.result()}")
                                quit(p.result())

                for j in job_pool: # to debug most failed jobs
                    if not j.running() and cfg_exp.max_concurrent_jobs == 99:
                        print(job_pool, flush=True)
                        executor._threads.clear()
                        concurrent.futures.thread._threads_queues.clear()
                        logging.error(f"Job {job_pool_args[j]} is not running, exiting. (DEBUG)")
                        quit(j.result())

                # Process completed jobs and handle retries
                for i, p in enumerate(as_completed(job_pool)):
                    if p.result() != 0:
                        logging.warning(f"Re-running job {iter_no}.{i+1}")
                        new_job_pool = [executor.submit(run_job_with_retry, job_pool_args[i])]
                        for np in as_completed(new_job_pool):
                            if np.result() != 0:
                                logging.critical(f"Exiting.")
                                quit(np.result())

            if num_jobs > 1:
                model_list = [
                    str(cfg_exp.dir / f"{iter_no}.{job_id}.pt")
                    for job_id in range(1, num_jobs + 1)
                ]
                process_out = subprocess.run([
                        cfg_cmd.cuda_cmd,
                        f"{cfg_exp.dir}/log/merge.{iter_no+1}.log",
                         cfg_exp.model_file,
                         *cfg_exp.get_model_args,
                         "--mode", "merge",
                         *cfg_exp.get_forcmd("dir"),
                        "--new-model",
                        cfg_exp.dir / f"{iter_no+1}.pt",
                        ",".join(model_list),
                ])
                for mdl in model_list:
                    satools.script_utils.run(["rm", mdl])
            else:
                satools.script_utils.run([
                        "mv",
                        cfg_exp.dir / f"{iter_no}.1.pt",
                        cfg_exp.dir / f"{iter_no+1}.pt",
                ])
            # remove old model
            if iter_no >= 15 and (iter_no - 10) % cfg_exp.checkpoint_interval != 0:
                mdl = cfg_exp.dir / f"{iter_no - 10}.pt"
                if os.path.isfile(mdl):
                    satools.script_utils.run(["rm", mdl])


    if stage <= 7:
        # do final model combination
        model_list = [
            str(cfg_exp.dir / f"{i}.pt")
            for i in range(num_iters, num_iters - cfg_exp.final_combination_n_model, -1)
        ]
        logging.info(f"Final model combination ... N last model: {cfg_exp.final_combination_n_model}")
        satools.script_utils.run([
                cfg_cmd.cuda_cmd,
                cfg_exp.dir / "log/combine.log",
                "env", "CUDA_VISIBLE_DEVICES=0",
                cfg_exp.model_file,
                *cfg_exp.get_model_args,
                "--mode", "final_combination",
                *cfg_exp.get_forcmd("dir"),
                "--new-model", cfg_exp.dir / "final.pt",
                "--egs", cfg_exp.egs_dir / "fst_valid.scp",
                *cfg_exp.get_forcmd("data"),
                ",".join(model_list),
        ])

    if stage <= 5:
        carbonTracker.epoch_end()
        carbonTracker.stop()

    for test_set in str(cfg_decode.test_set).split(","):
        test_set = Path(test_set)
        data_name = os.path.basename(test_set)
        decode_iter = args.decode_iter.replace(".pt", "") if args.decode_iter != "_" else cfg_decode.decode_iter
        decode_suff = "_iter{}{}".format(decode_iter, cfg_decode.suffix)
        out_dir = cfg_exp.dir / f"decode_{data_name}{decode_suff}"

        if stage <= 8:
            num_jobs = satools.utils.split_data(test_set, cfg_decode.num_jobs)
            logging.info(f"Decoding '{data_name}' with  {cfg_decode.num_jobs} jobs...")

            gpu_opts = []
            if bool(cfg_decode.gpu):
                gpu_opts = ["--use-gpu", "True", "--gpu-id", "JOB"]

            feats_scp = f"{test_set}/split{num_jobs}/JOB/wav.scp"

            tqdm = subprocess.Popen(f"exec tail -F {cfg_exp.dir}/log/tqdm", stderr=subprocess.PIPE, shell=True)

            satools.script_utils.run([
                    cfg_cmd.cpu_cmd if bool(cfg_decode.gpu) else cfg_cmd.cpu_cmd,
                    f"JOB=1:{num_jobs}",
                    out_dir / "log" / "decode.JOB.log",
                     cfg_exp.model_file,
                     *cfg_exp.get_model_args,
                    "--mode", "decode",
                     *cfg_exp.get_forcmd("dir"),
                    *gpu_opts,
                    "--decode-feats", feats_scp,
                    cfg_exp.dir / f"{decode_iter}.pt",
                    "|",
                    "shutil/decode/latgen-faster-mapped.sh",
                    cfg_decode.graph_dir / "words.txt",
                    cfg_exp.dir / "0.trans_mdl",
                    f"{cfg_decode.graph_dir}/HCLG.fst",
                    out_dir / "lat.JOB.gz",
            ])
            tqdm.kill()
            print("", file=sys.stderr)
            satools.script_utils.write_single_param_file(num_jobs, out_dir / "num_jobs")


        if stage <= 9:
            logging.info(f"Scoring...")
            if not os.path.isfile(out_dir / "../final.mdl") and os.path.isfile(out_dir / "../0.trans_mdl"):
                satools.script_utils.run([ "ln", "-r", "-s", out_dir / "../0.trans_mdl", out_dir / "../final.mdl" ])
            satools.script_utils.run(["local/score.sh", "--cmd", cfg_cmd.cpu_cmd, test_set, cfg_decode.graph_dir, out_dir])

            logging.info(f"Printing best WER without rescoring {out_dir}...")
            satools.script_utils.run([ "cat", "{}/wer*".format(out_dir), "|", "utils/best_wer.sh", ">", "{}/best_wer".format(out_dir) ], shell=True)
            logging.info(" " + satools.script_utils.read_single_param_file("{}/best_wer".format(out_dir), typename=str))

            # logging.info(f"Rescore with a N gram LM...")
            # satools.script_utils.run([
            #         "steps/lmrescore_const_arpa.sh",
            #         "--cmd", cfg_cmd.cpu_cmd,
            #         cfg_decode.lang_lp_tgsmall,
            #         cfg_decode.lang_lp_fg_large,
            #         test_set,
            #         out_dir,
            #         f"{out_dir}_fg",
            # ])
            # logging.info(f"Printing best WER with rescoring {out_dir}_fg...")
            # satools.script_utils.run([ "cat", "{}_fg/wer*".format(out_dir), "|", "utils/best_wer.sh", ">", "{}_fg/best_wer".format(out_dir)], shell=True)
            # logging.info(" " + satools.script_utils.read_single_param_file( f"{out_dir}_fg/best_wer", typename=str))

            # logging.info(f"Computing WER details for {out_dir}_fg...")
            # satools.script_utils.run([
            #         "./shutil/decode/wer_detail.sh",
            #         "--cmd", cfg_cmd.cpu_cmd,
            #         "--dataDir", test_set,
            #         "--decodeDir", f"{out_dir}_fg",
            #         "--langDir", cfg_decode.lang_lp_fg_large,
            #     ], shell=True,
            # )

    if stage <= 10:
        logging.info(f"Creating JIT model")
        satools.script_utils.run([
                cfg_cmd.cpu_cmd,
                cfg_exp.dir / "log" / "jit.log",
                 cfg_exp.model_file,
                 *cfg_exp.get_model_args,
                "--mode", "jit_save",
                 *cfg_exp.get_forcmd("dir"),
                "--new-model", cfg_exp.dir / f"{decode_iter}.jit",
                cfg_exp.dir / f"{decode_iter}.pt",
            ]
        )
        if args.upload != "no":
            logging.info(f"Upload model to a github release")

            parsed_cfg_file = cfg_exp.dir / f"parsed_cfg.configs.{os.path.basename(args.config).replace('.', '')}"
            if "var" in cfg_parse:
                del cfg_parse["var"]
            satools.script_utils.write_single_param_file(cfg_parse, parsed_cfg_file)

            score = []
            up_as = []
            for test_set in str(cfg_decode.test_set).split(","):
                if (cfg_exp.dir / f"decode_{os.path.basename(test_set)}{decode_suff}_fg/best_wer").exists():
                    score.append(cfg_exp.dir / f"decode_{os.path.basename(test_set)}{decode_suff}_fg/best_wer")
                    up_as.append(f"decode_{os.path.basename(test_set)}{decode_suff}_fg_best_wer.txt")
                if (cfg_exp.dir / f"decode_{os.path.basename(test_set)}{decode_suff}/best_wer").exists():
                    score.append(cfg_exp.dir / f"decode_{os.path.basename(test_set)}{decode_suff}/best_wer")
                    up_as.append(f"decode_{os.path.basename(test_set)}{decode_suff}_best_wer.txt")
            up_as = {k:v for k, v in zip(score, up_as)}
            up_as[args.config] = "default_cfg."+args.config

            satools.script_utils.push_github_model(
                tag_name=args.upload,
                up_assets=[
                    *score,
                    cfg_exp.dir / f"conf.pt",
                    cfg_exp.dir / f"{decode_iter}.pt", cfg_exp.dir / f"{decode_iter}.jit",
                    cfg_decode.lang_lp_tgsmall+"/G.fst", cfg_decode.lang_lp_fg_large+"/G.carpa",
                    cfg_decode.graph_dir / "words.txt",
                    cfg_exp.dir / "0.trans_mdl",
                    f"{cfg_decode.graph_dir}/HCLG.fst",
                    parsed_cfg_file,
                    args.config,
                ], up_as_name=up_as, force=False
            )


if __name__ == "__main__":
    train()
