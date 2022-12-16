#!/usr/bin/env python3

description = """
  This script trains and tests chain models.
  It takes a config file, if none is provided the script will look at configs/default.
"""

import argparse
import concurrent
import configparser
import datetime
import glob
import json
import logging
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import satools
import torch

logging.basicConfig(level=logging.INFO, format="satools %(levelname)s: %(message)s")
logging.getLogger("geocoder").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

from dataclasses import dataclass
from carbontracker.tracker import CarbonTracker


@dataclass
class TrainerOpts:
    num_jobs_initial: int = 1
    num_jobs_final: int = 6
    lr_initial: float = 0.01
    lr_final: float = 0.001
    iter_no: int = 0
    num_epochs: int = 6
    train_stage: str = "0"
    frames_per_iter: int = 120000
    chunk_width: str = "140"
    cmd: str = "queue.pl -l q_gpu -V"
    diagnostics_interval: int = 10
    checkpoint_interval: int = 100
    srand: int = 1

    def load_from_config(self, cfg):
        for key, value in cfg.items():
            if hasattr(self, key):
                type_of_value = type(getattr(self, key))
                setattr(self, key, type_of_value(value))
        return self


@dataclass
class ModelOpts:
    model_file: str = ""
    dirname: str = "./"
    left_context: int = 0
    right_context: int = 0
    egs_dir: str = "./egs"
    den_graph: str = "./den.fst"
    frame_subsampling_factor: int = 3 # same as in data_prep

    def load_from_config(self, cfg):
        for key, value in cfg.items():
            if hasattr(self, key):
                type_of_value = type(getattr(self, key))
                setattr(self, key, type_of_value(value))
        return self


def run_diagnostics(
    dirname, model_file, iter_no, egs_file, train_set, job_cmd, diagnostic_name="valid"
):
    """
    sub a single diagnostic job and let ThreadPoolExecutor monitor its progress
    """
    log_file = "{}/log/compute_prob_{}.{}.log".format(dirname, diagnostic_name, iter_no)
    logging.info(f"Submitting diagnostic {diagnostic_name}")
    process_out = subprocess.run(
        [
            *job_cmd.split(),
            log_file,
            "env",
            "CUDA_VISIBLE_DEVICES="
            + str(
                [
                    i
                    for i, value in enumerate(["valid", "train_diagnositc"])
                    if value == diagnostic_name
                ][0]
            ),
            *model_file,
            "--dir",
            dirname,
            "--mode",
            "diagnostic",
            "--data",
            train_set,
            "--egs",
            egs_file,
            os.path.join(dirname, "{}.pt".format(iter_no)),
        ]
    )

    return process_out.returncode


def submit_diagnostic_jobs(
    dirname, model_file, iter_no, egs_dir, train_set, job_cmd, exp_cfg, args
):
    job_pool = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for diagnostic_name in ["train_diagnositc", "valid"]:
            if (
                args.skip_train_diagnositc == "yes"
                and diagnostic_name == "train_diagnositc"
            ):
                continue
            egs_file = os.path.join(egs_dir, "fst_{}.scp".format(diagnostic_name))
            p = executor.submit(
                run_diagnostics,
                dirname,
                model_file,
                iter_no,
                egs_file,
                train_set,
                job_cmd,
                diagnostic_name,
            )
            job_pool.append(p)
            #  print("Pool size:", job_pool, flush=True)
    return job_pool


def run_job(
    num_jobs,
    job_id,
    dirname,
    iter_no,
    model_file,
    lr,
    egs,
    train_set,
    minibatch_size,
    num_iters,
    job_cmd,
    sampler,
    xent_regularize=0.025,
    grad_acc_steps="1",
    l2_regularize_factor=None,
):
    """
    sub a single job and let ThreadPoolExecutor monitor its progress
    """
    log_file = "{}/log/train.{}.{}.log".format(dirname, iter_no, job_id)
    cuda_device = job_id - 1
    if cuda_device >= torch.cuda.device_count():
        cuda_device = abs(
            torch.cuda.device_count() - 1 - (cuda_device - torch.cuda.device_count())
        )
    cuda_device = str(cuda_device)
    l2_reg = str(1.0 / num_jobs)
    if l2_regularize_factor:
        l2_reg = l2_regularize_factor
    process_out = subprocess.run(
        [
            *job_cmd.split(),
            log_file,
            "env",
            "CUDA_VISIBLE_DEVICES=" + cuda_device,
            *model_file,
            "--dir",
            dirname,
            "--mode",
            "training",
            "--lr",
            str(lr),
            "--egs",
            egs,
            "--data",
            train_set,
            "--num-iter",
            str(num_iters),
            "--l2-regularize-factor",
            l2_reg,
            "--minibatch-size",
            minibatch_size,
            "--grad-acc-steps",
            grad_acc_steps,
            "--new-model",
            os.path.join(dirname, "{}.{}.pt".format(iter_no, job_id)),
            "--xent-regularize",
            str(xent_regularize),
            "--sampler",
            str(sampler),
            os.path.join(dirname, "{}.pt".format(iter_no)),
        ]
    )
    return process_out.returncode


def train():
    parser = argparse.ArgumentParser(description="Acoustic model training script")
    satools.script_utils.add_chain_recipe_opts(parser)
    # the idea behind a test config is that one can run different configurations of test
    parser.add_argument(
        "--test-config", default="test", help="name of the test to be run"
    )
    parser.add_argument("--decode-iter", default="final")
    parser.add_argument("--config", default="configs/default")
    parser.add_argument("--skip-train-diagnositc", default="no")
    args = parser.parse_args()

    logging.info("Reading config")
    cfg_parse = configparser.ConfigParser()
    cfg_parse.read(args.config)
    cmd = cfg_parse["cmd"]
    cpu_cmd = os.getcwd() + "/" + cmd["cpu_cmd"]
    cuda_cmd = os.getcwd() + "/" + cmd["cuda_cmd"]

    exp_cfg = cfg_parse["exp"]
    assert exp_cfg is not None

    stage = args.stage
    model_file = [exp_cfg["model_file"]]
    model_file += json.loads(exp_cfg["model_args"]) if "model_args" in exp_cfg else []
    exp = exp_cfg["exp"] if "exp" in exp_cfg else "exp"
    chain_affix = exp_cfg["chain_affix"] if "chain_affix" in exp_cfg else ""
    chain_dir = os.path.join(exp, f"chain{chain_affix}")
    dirname = os.path.join(chain_dir, exp_cfg["dirname"])
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    egs_dir = os.path.join(dirname, "fst_egs")
    tree_dir = exp_cfg["tree_dir"]
    train_set = exp_cfg["train_set"]
    multi_egs_loading = exp_cfg["multi_egs_loading"] != "False" if "multi_egs_loading" in exp_cfg else bool("True")

    l2_regularize = args.l2_regularize
    xent_regularize = args.xent_regularize
    if "xent_regularize" in exp_cfg:
        xent_regularize = exp_cfg["xent_regularize"]

    model_opts = ModelOpts().load_from_config(exp_cfg)
    frame_subsampling_factor = model_opts.frame_subsampling_factor
    trainer_opts = TrainerOpts().load_from_config(exp_cfg)
    if trainer_opts.train_stage == "last" or trainer_opts.train_stage == '"last"':
        pattern = os.path.join(dirname, "*.pt")
        cp_list = glob.glob(pattern)
        cp_list = list(map(lambda x: x.split("/")[-1].split(".")[0], cp_list))
        if "final" in cp_list:
            cp_list.remove("final")
        if len(cp_list) == 0:
            trainer_opts.train_stage = "0"
        else:
            cp_list = list(map(lambda x: int(x), cp_list))
            trainer_opts.train_stage = str(sorted(cp_list)[-1])
            logging.info(f"Resuming from stage: {trainer_opts.train_stage}")

    trainer_opts.train_stage = int(trainer_opts.train_stage)

    if not os.path.isdir(dirname):
        os.makedirs(dirname)
        logging.info(f"Created {dirname}")
    if not os.path.isfile(os.path.join(dirname, "tree")):
        shutil.copy(os.path.join(tree_dir, "tree"), dirname)

    if not os.path.isfile(os.path.join(dirname, "num_pdfs")):
        logging.info(f"Creating num_pdfs file in {dirname}")
        num_pdfs = (
            subprocess.check_output(["tree-info", os.path.join(tree_dir, "tree")])
            .decode()
            .strip()
            .split("\n")[0]
            .split()[1]
        )
        with open(os.path.join(dirname, "num_pdfs"), "w") as opf:
            opf.write(num_pdfs)
            opf.close()
    if not os.path.isfile(os.path.join(dirname, "0.trans_mdl")):
        satools.script_utils.run(
            [
                "copy-transition-model",
                os.path.join(tree_dir, "final.mdl"),
                os.path.join(dirname, "0.trans_mdl"),
            ]
        )
    #   create or copy the egs folder
    if stage <= 4 and not ("egs_dir" in exp_cfg and exp_cfg["egs_dir"]):
        logging.info("Creating egs")
        satools.script_utils.run(
            ["./local/chain/e2e/get_egs.sh", train_set, tree_dir, egs_dir]
        )
    else:
        egs_dir = exp_cfg["egs_dir"]

    if not os.path.isfile(os.path.join(dirname, "den.fst")):
        satools.script_utils.run(
            [
                "cp",
                os.path.join(egs_dir + "/../", "den.fst"),
                os.path.join(dirname, "den.fst"),
            ]
        )
    if not os.path.isfile(os.path.join(dirname, "normalization.fst")):
        satools.script_utils.run(
            [
                "cp",
                os.path.join(egs_dir + "/../", "normalization.fst"),
                os.path.join(dirname, "normalization.fst"),
            ]
        )

    # we start training with
    num_archives = satools.script_utils.get_egs_info(egs_dir)
    if multi_egs_loading == False:
        num_archives = 1
    num_epochs = trainer_opts.num_epochs
    # we don't use num of jobs b/c it is 1 for now
    num_archives_to_process = num_archives * num_epochs * frame_subsampling_factor
    num_iters = (num_archives_to_process * 2) // (
        trainer_opts.num_jobs_initial + trainer_opts.num_jobs_final
    )
    num_iters_last_epoch = (
        num_archives * (num_epochs - 1) * frame_subsampling_factor * 2
    ) // (trainer_opts.num_jobs_initial + trainer_opts.num_jobs_final)

    carbonTracker = CarbonTracker(epochs=1, components="gpu", verbose=2)
    carbonTracker.epoch_start()

    #   start the training
    if stage <= 5 and trainer_opts.train_stage == 0:
        logging.info(
            f"Iter num_archives_to_process={num_archives_to_process}, num_archives={num_archives}, frame_subsampling_factor={frame_subsampling_factor}, num_epochs={num_epochs}"
        )
        logging.info("Initializing model")
        additional_ops = []
        if "init_weight_model" in exp_cfg and exp_cfg["init_weight_model"] != "":
            additional_ops += ["--init-weight-model", exp_cfg["init_weight_model"]]
        process_out = subprocess.run(
            [
                *cuda_cmd.split(),
                os.path.join(dirname, "log", "init.log"),
                *model_file,
                "--mode",
                "init",
                "--dir",
                dirname,
                *additional_ops,
                os.path.join(dirname, "0.pt"),
            ]
        )
        if process_out.returncode != 0:
            quit(process_out.returncode)

    if stage <= 6:
        train_stage = trainer_opts.train_stage
        logging.info(f"Starting training from stage={train_stage}")
        logging.info(
            f"Watch logs with:\n  tail -F {dirname}/log/train.{{0..{num_iters}}}.{{1..{trainer_opts.num_jobs_final}}}.log {dirname}/log/init.log {dirname}/log/compute_prob_valid.{{1..{num_iters}}}.log | sed '/LOG.*Numerator/d' | ./local/grcat conf.log"
        )
        logging.info(f"  Open tensorbord with 'tensorboard --logdir {dirname}/runs'")
        assert train_stage >= 0
        num_archives_processed = 0
        for iter_no in range(0, num_iters):
            num_jobs = satools.script_utils.get_current_num_jobs(
                iter_no,
                num_iters,
                trainer_opts.num_jobs_initial,
                1,  # we don't play with num-jobs-step
                trainer_opts.num_jobs_final,
            )
            if iter_no < train_stage:
                num_archives_processed += num_jobs
                continue
            assert num_jobs > 0
            lr = satools.script_utils.get_learning_rate(
                iter_no,
                num_jobs,
                num_iters,
                num_archives_processed,
                num_archives_to_process,
                trainer_opts.lr_initial,
                trainer_opts.lr_final,
                schedule_type="exponential",
            )
            diagnostic_job_pool = None
            if (
                iter_no == 1
                or (iter_no % trainer_opts.diagnostics_interval == 0 and iter_no != 0)
                or (iter_no + 1 == num_iters)
            ):
                diagnostic_job_pool = submit_diagnostic_jobs(
                    dirname,
                    model_file,
                    iter_no,
                    egs_dir,
                    train_set,
                    cuda_cmd,
                    exp_cfg,
                    args,
                )
                for p in as_completed(diagnostic_job_pool):
                    if p.result() != 0:
                        quit(p.result())

            logging.info(
                "{} Running iter={} of {} with {} jobs and lr={:.6f}".format(
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    iter_no,
                    num_iters,
                    num_jobs,
                    lr,
                )
            )
            with ThreadPoolExecutor(max_workers=num_jobs) as executor:
                job_pool = []
                add_praram = {}
                if "grad_acc_steps" in exp_cfg:
                    add_praram["grad_acc_steps"] = exp_cfg["grad_acc_steps"]

                if "l2_regularize_factor" in exp_cfg:
                    add_praram["l2_regularize_factor"] = exp_cfg["l2_regularize_factor"]

                train_egs = "{}/fst_train.{}.scp".format(
                    egs_dir, num_archives_processed % num_archives + 1
                )

                if multi_egs_loading == False:
                    train_egs = "{}/fst_train_shuffle.scp".format(egs_dir)

                for job_id in range(1, num_jobs + 1):
                    p = executor.submit(
                        run_job,
                        num_jobs,
                        job_id,
                        dirname,
                        iter_no,
                        model_file,
                        lr,
                        train_egs,
                        train_set,
                        exp_cfg["minibatch_size"],
                        num_iters,
                        cuda_cmd,
                        exp_cfg["sampler"] if "sampler" in exp_cfg else "BucketBatch",
                        **add_praram,
                        xent_regularize=xent_regularize,
                    )
                    num_archives_processed += 1
                    job_pool.append(p)
                for j in job_pool:
                    if j.running() == False:
                        print(job_pool, flush=True)
                        executor._threads.clear()
                        concurrent.futures.thread._threads_queues.clear()
                        quit(j.result())
                for p in as_completed(job_pool):
                    if p.result() != 0:
                        quit(p.result())
            if num_jobs > 1:
                model_list = [
                    os.path.join(dirname, "{}.{}.pt".format(iter_no, job_id))
                    for job_id in range(1, num_jobs + 1)
                ]
                process_out = subprocess.run(
                    [
                        *cuda_cmd.split(),
                        "{}/log/merge.{}.log".format(dirname, iter_no + 1),
                        *model_file,
                        "--dir",
                        dirname,
                        "--mode",
                        "merge",
                        "--new-model",
                        os.path.join(dirname, "{}.pt".format(iter_no + 1)),
                        ",".join(model_list),
                    ]
                )
                for mdl in model_list:
                    satools.script_utils.run(["rm", mdl])
            else:
                satools.script_utils.run(
                    [
                        "mv",
                        os.path.join(dirname, "{}.1.pt".format(iter_no)),
                        os.path.join(dirname, "{}.pt".format(iter_no + 1)),
                    ]
                )
            # remove old model
            if iter_no >= 15 and (iter_no - 10) % trainer_opts.checkpoint_interval != 0:
                mdl = os.path.join(dirname, "{}.pt".format(iter_no - 10))
                if os.path.isfile(mdl):
                    satools.script_utils.run(["rm", mdl])
    if stage <= 7:
        # do final model combination
        n_models = (
            int(exp_cfg["final_combination_n_model"])
            if "final_combination_n_model" in exp_cfg
            else 10
        )
        model_list = [
            os.path.join(dirname, f"{i}.pt")
            for i in range(num_iters, num_iters - n_models, -1)
        ]
        logging.info("Final model combination...")
        diagnostic_name = "valid"
        egs_file = os.path.join(egs_dir, "fst_valid.scp")
        satools.script_utils.run(
            [
                *cuda_cmd.split(),
                "{}/log/combine.log".format(dirname),
                "env",
                "CUDA_VISIBLE_DEVICES=0",
                *model_file,
                "--dir",
                dirname,
                "--mode",
                "final_combination",
                "--new-model",
                os.path.join(dirname, "final.pt"),
                "--egs",
                egs_file,
                "--data",
                train_set,
                ",".join(model_list),
            ]
        )

    carbonTracker.epoch_end()
    carbonTracker.stop()

    graph_dir = ""
    decode_params = cfg_parse[args.test_config]
    if "graph_dir" in exp_cfg:
        graph_dir = exp_cfg["graph_dir"]
    if "graph_dir" in decode_params:
        graph_dir = decode_params["graph_dir"]
    if not graph_dir:
        graph_dir = os.path.join(dirname, "graph")

    final_iter = num_iters - 1
    data_dir = decode_params["test_set"]
    data_name = os.path.basename(data_dir)
    decode_iter = decode_params["iter"] if "iter" in decode_params else args.decode_iter
    decode_gpu = bool(decode_params["gpu"]) if "gpu" in decode_params else False
    decode_affix = decode_params["suffix"] if "suffix" in decode_params else ""
    decode_suff = "_iter{}{}".format(decode_iter, decode_affix)
    out_dir = os.path.join(dirname, f"decode_{data_name}{decode_suff}")
    graph = "{}/HCLG.fst".format(graph_dir)

    if stage <= 8:
        if "num_jobs" in decode_params:
            num_jobs = satools.utils.split_data(
                data_dir,
                int(decode_params["num_jobs"]),
            )
        else:
            num_jobs = satools.utils.split_data(data_dir)
        logging.info(f"Decoding with {num_jobs} jobs...")

        gpu_opts = []
        if decode_gpu:
            gpu_opts = ["--use-gpu", "True", "--gpu-id", "JOB"]

        feats_scp = "{}/split{}/JOB/wav.scp".format(data_dir, num_jobs)

        tqdm = subprocess.Popen(
            f"tail -F {dirname}/log/tqdm 2> /dev/null", shell=True
        )

        satools.script_utils.run(
            [
                *cpu_cmd.split(),
                "JOB=1:{}".format(num_jobs),
                os.path.join(out_dir, "log", "decode.JOB.log"),
                *model_file,
                "--dir",
                dirname,
                "--mode",
                "decode",
                *gpu_opts,
                "--decode-feats",
                feats_scp,
                os.path.join(dirname, "{}.pt".format(decode_iter)),
                "|",
                "shutil/decode/latgen-faster-mapped.sh",
                os.path.join(graph_dir, "words.txt"),
                os.path.join(dirname, "0.trans_mdl"),
                graph,
                os.path.join(out_dir, "lat.JOB.gz"),
            ]
        )
        tqdm.terminate()
        logging.info("")
        opf = open(os.path.join(out_dir, "num_jobs"), "w")
        opf.write("{}".format(num_jobs))
        opf.close()

    if stage <= 9:
        logging.info(f"Scoring...")
        if not os.path.isfile(os.path.join(out_dir, "../final.mdl")) and os.path.isfile(
            os.path.join(out_dir, "../0.trans_mdl")
        ):
            satools.script_utils.run(
                [
                    "ln",
                    "-r",
                    "-s",
                    os.path.join(out_dir, "../0.trans_mdl"),
                    os.path.join(out_dir, "../final.mdl"),
                ]
            )
        satools.script_utils.run(
            ["local/score.sh", "--cmd", cpu_cmd, data_dir, graph_dir, out_dir]
        )
        logging.info(f"Printing best WER without rescoring {out_dir}...")
        satools.script_utils.run(
            " ".join(
                [
                    "cat",
                    "{}/wer*".format(out_dir),
                    "|",
                    "utils/best_wer.sh",
                    ">",
                    "{}/best_wer".format(out_dir),
                ]
            ),
            shell=True,
        )
        satools.script_utils.run(
            " ".join(
                [
                    "cat",
                    "{}/best_wer".format(out_dir),
                ]
            ),
            shell=True,
        )
        logging.info(
            " "
            + satools.script_utils.read_single_param_file(
                "{}/best_wer".format(out_dir), typename=str
            )
        )

        logging.info(f"Rescore with a N gram LM...")
        satools.script_utils.run(
            [
                "steps/lmrescore_const_arpa.sh",
                "--cmd",
                cpu_cmd,
                "data/lang_lp_test_tgsmall",
                "data/lang_lp_test_fglarge",
                data_dir,
                out_dir,
                f"{out_dir}_fg",
            ]
        )
        logging.info(f"Printing best WER with rescoring {out_dir}_fg...")
        satools.script_utils.run(
            " ".join(
                [
                    "cat",
                    "{}_fg/wer*".format(out_dir),
                    "|",
                    "utils/best_wer.sh",
                    ">",
                    "{}_fg/best_wer".format(out_dir),
                ]
            ),
            shell=True,
        )
        logging.info(
            " "
            + satools.script_utils.read_single_param_file(
                "{}_fg/best_wer".format(out_dir), typename=str
            )
        )
        logging.info(f"Computing WER details for {out_dir}_fg...")
        satools.script_utils.run(
            " ".join(
                [
                    "./local/wer_detail.sh",
                    "--dataDir",
                    "./data/{}".format(data_name),
                    "--decodeDir",
                    "{}_fg".format(out_dir),
                    "--langDir",
                    "data/lang_lp_test_fglarge",
                ]
            ),
            shell=True,
        )


if __name__ == "__main__":
    train()
