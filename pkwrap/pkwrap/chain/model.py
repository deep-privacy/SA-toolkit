import os
import sys
import json
from collections import Counter
import logging
import argparse
from dataclasses import dataclass
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from .. import script_utils
from .. import utils
from .. import tensorboard
from .objf import train_lfmmi_one_iter, compute_chain_objf
from .egs_wav2vec2 import (
    Wav2vec2EgsDataset,
    Wav2vec2DecodeDataset,
    Wav2vec2EgsCollectFn,
)

import kaldiio
import matplotlib.pyplot as plt
from tqdm import tqdm


@dataclass
class TrainerOpts:
    mode: str = ""
    dir: str = ""
    lr: float = 0.001
    num_iter: int = -1
    grad_acc_steps: int = 1
    base_model: str = ""
    sampler: str = "BucketBatch"
    init_weight_model: str = ""


@dataclass
class DecodeOpts:
    use_gpu: bool = False
    gpu_id: int = 0
    decode_feats: str = "data/test/feats.scp"
    decode_output: str = "-"


@dataclass
class ChainModelOpts(TrainerOpts, DecodeOpts):
    dataset: str = ""
    egs: str = ""
    new_model: str = ""
    l2_regularize: float = 1e-4
    l2_regularize_factor: float = 1.0
    out_of_range_regularize: float = 0.01
    leaky_hmm_coefficient: float = 0.1
    xent_regularize: float = 0.025
    minibatch_size: int = 16
    output_dim: int = 1

    def load_from_config(self, cfg):
        for key, value in cfg.items():
            if hasattr(self, key):
                type_of_value = type(getattr(self, key))
                setattr(self, key, type_of_value(value))
        return self


class ChainModel(nn.Module):
    def __init__(self, model_cls, cmd_line=False, **kwargs):
        """initialize a ChainModel"""
        super(ChainModel, self).__init__()
        assert model_cls is not None
        if cmd_line:
            args = self.load_cmdline_args()
            self.chain_opts = ChainModelOpts()
            self.chain_opts.load_from_config(vars(args))
        else:
            self.chain_opts = ChainModelOpts()
            self.chain_opts.load_from_config(kwargs)

        self.Net = model_cls
        self.call_by_mode()

    def call_by_mode(self):
        """A that calls appropriate method based on the value of chain_opts.mode

        So far the modes supported are:
            - init
            - merge
            - train (or training)
            - validate (or diagnostic)
            - infer (or decode)
        """
        self.reset_dims()
        if self.chain_opts.mode == "init":
            self.init()
        elif self.chain_opts.mode == "merge":
            self.merge()
        elif self.chain_opts.mode in ["validate", "diagnostic"]:
            self.validate()
        elif self.chain_opts.mode in ["train", "training"]:
            self.train()
        elif self.chain_opts.mode in ["decode", "infer"]:
            self.infer()
        elif self.chain_opts.mode == "final_combination":
            self.combine_final_model()

    def init(self):
        """Initialize the model and save it in chain_opts.base_model"""
        model = self.Net(self.chain_opts.output_dim)
        if self.chain_opts.init_weight_model != "":
            init_weight_provided = torch.load(self.chain_opts.init_weight_model)
            if hasattr(model, "init_custom_load"):
                init_weight_provided = model.init_custom_load(init_weight_provided)

            init_weight_provided_matched, unmatch = utils.torch.match_state_dict(
                model.state_dict(), init_weight_provided
            )
            not_inited = model.load_state_dict(
                init_weight_provided_matched, strict=False
            )
            logging.info(
                "Init from previous model {}, layers not initialized: {}: layers ommited (wrong shape): {}".format(
                    self.chain_opts.init_weight_model,
                    str(not_inited),
                    str(unmatch.keys()),
                )
            )
        torch.save(model.state_dict(), self.chain_opts.base_model)

    def train(self):
        """Run one iteration of LF-MMI training

        This is called by
        >>> self.train()

        It will probably be renamed as self.fit() since this seems to be
        the standard way other libraries call the training function.
        """
        raise NotImplementedError("Only implementing e2e LF-MMI")

    @torch.no_grad()
    def validate(self):
        from _pkwrap import kaldi  # lazy import (kaldi-free decoding)

        kaldi.InstantiateKaldiCuda()
        chain_opts = self.chain_opts
        den_fst_path = os.path.join(chain_opts.dir, "den.fst")

        model = self.Net(self.chain_opts.output_dim)
        model.load_state_dict(torch.load(chain_opts.base_model))
        model.eval()

        training_opts = kaldi.chain.CreateChainTrainingOptions(
            chain_opts.l2_regularize,
            chain_opts.out_of_range_regularize,
            chain_opts.leaky_hmm_coefficient,
            chain_opts.xent_regularize,
        )

        dataset = Wav2vec2EgsDataset(
            "{}/wav.scp".format(chain_opts.dataset),
            chain_opts.egs,
            "{}/utt2len".format(chain_opts.dataset),
            "{}/0.trans_mdl".format(chain_opts.dir),
            "{}/normalization.fst".format(chain_opts.dir),
        )
        compute_chain_objf(
            model,
            dataset,
            den_fst_path,
            training_opts,
            minibatch_size=chain_opts.minibatch_size,
            tensorboard=tensorboard.PkwrapTwensorBoard(self)
            if "valid" in self.chain_opts.egs
            else None,
        )

    @torch.no_grad()
    def merge(self):
        chain_opts = self.chain_opts
        base_models = chain_opts.base_model.split(",")
        assert len(base_models) > 0
        model0 = self.Net(self.chain_opts.output_dim)
        model0.load_state_dict(torch.load(base_models[0]))
        model_acc = dict(model0.named_parameters())
        for mdl_name in base_models[1:]:
            this_mdl = self.Net(self.chain_opts.output_dim)
            this_mdl.load_state_dict(torch.load(mdl_name))
            for name, params in this_mdl.named_parameters():
                # Only average layers that are trained otherwise we ran
                # into chained numerical division imprecision
                if params.requires_grad:
                    model_acc[name].data.add_(params.data)
        weight = 1.0 / len(base_models)
        for name in model_acc:
            if model_acc[name].requires_grad:
                model_acc[name].data.mul_(weight)
        torch.save(model0.state_dict(), chain_opts.new_model)

    @torch.no_grad()
    def get_forward(
        self,
        device=torch.device("cpu"),
        share_memory=False,
        get_model_module=False,
        load_model=True,
    ):
        chain_opts = self.chain_opts

        model = self.Net(chain_opts.output_dim)
        base_model = chain_opts.base_model
        model = model.to(device)
        if load_model:
            try:
                model.load_state_dict(torch.load(base_model))
            except Exception as e:
                logging.warning("Warning cannot load model {}".format(base_model))
                logging.warning("Retrying with strict=False")
                #  logging.warning(e)
                not_inited = model.load_state_dict(torch.load(base_model), strict=False)
                logging.warning("Incompatible layers: {}".format(not_inited))

        model.eval()
        if share_memory:
            model.share_memory()

        def _forward(waveform, spec_augment=lambda x: x):
            with torch.no_grad():
                post, xent_output = model(waveform, spec_augment=spec_augment)
                return post, model

        if get_model_module:
            return _forward, model
        return _forward

    @torch.no_grad()
    def infer(self):
        chain_opts = self.chain_opts

        device = torch.device("cpu")
        if chain_opts.use_gpu:
            run_on_gpu = (list(range(0, torch.cuda.device_count())) * 200)[
                chain_opts.gpu_id
            ]
            device = torch.device("cuda:{}".format(run_on_gpu))
            logging.info("Using GPU: {}".format(run_on_gpu))

        model = self.get_forward(device=device)

        write_with_kaldi = True
        try:
            from _pkwrap import kaldi  # lazy import (kaldi-free decoding)
        except ImportError as error:
            # shutil/decode/latgen-faster-mapped.sh compatible but slower
            logging.critical(" -- Failed to import kaldi for feat writing --")
            logging.exception(error)
            write_with_kaldi = False

        if write_with_kaldi:
            writer_spec = "ark,t:{}".format(chain_opts.decode_output)
            writer = script_utils.feat_writer(writer_spec)
            close = writer.Close
            writer = writer.Write
            tensor_to_writer = lambda x: kaldi.matrix.TensorToKaldiMatrix(x)
        else:
            logging.info(" -- Using Kaldiio for feat writing --")
            writer = kaldiio.WriteHelper("ark,t:{}".format(chain_opts.decode_output))
            close = writer.close
            tensor_to_writer = lambda x: x.numpy()

        dataset = Wav2vec2DecodeDataset.from_wav_scp(chain_opts.decode_feats)
        dataloader = torch.utils.data.DataLoader(
            dataset, collate_fn=Wav2vec2EgsCollectFn, num_workers=8
        )

        if chain_opts.gpu_id == 0 or chain_opts.gpu_id == 1:
            tqdm_file = open(self.chain_opts.dir + "/log/tqdm", "w")
            dataloader = tqdm(dataloader, file=tqdm_file)

        for feats, key in dataloader:
            if chain_opts.use_gpu:
                feats = feats.to(device)
            post, _ = model(feats)
            post = post.squeeze(0).cpu()
            # batch size = 1 !!
            writer(key[0], tensor_to_writer(post))
            logging.info("Wrote {}".format(key[0]))
        close()
        tqdm_file.seek(0)
        tqdm_file.truncate()

    def reset_dims(self):
        # what if the user wants to pass it? Just override this function
        num_pdfs_filename = os.path.join(self.chain_opts.dir, "num_pdfs")
        self.chain_opts.output_dim = script_utils.read_single_param_file(
            num_pdfs_filename
        )

    def load_cmdline_args(self):
        parser = argparse.ArgumentParser(description="")
        parser.add_argument("--mode", default="init")
        parser.add_argument("--dir", default="")
        parser.add_argument("--lr", default=0.001, type=float)
        parser.add_argument("--egs", default="")
        parser.add_argument("--sampler", default="BucketBatch", type=str)
        parser.add_argument("--dataset", default="")
        parser.add_argument("--new-model", default="")
        parser.add_argument("--l2-regularize", default=1e-4, type=float)
        parser.add_argument(
            "--l2-regularize-factor", default=1.0, type=float
        )  # this is the weight_decay in pytorch
        parser.add_argument("--out-of-range-regularize", default=0.01, type=float)
        parser.add_argument("--xent-regularize", default=0.025, type=float)
        parser.add_argument("--leaky-hmm-coefficient", default=0.1, type=float)
        parser.add_argument("--minibatch-size", default=32, type=int)
        parser.add_argument("--num-iter", default=-1, type=int)
        parser.add_argument("--grad-acc-steps", default=1, type=int)
        parser.add_argument("--decode-feats", default="data/test/feats.scp", type=str)
        parser.add_argument("--decode-output", default="-", type=str)
        parser.add_argument("--decode-iter", default="final", type=str)
        parser.add_argument("--use-gpu", default=False, type=bool)
        parser.add_argument("--gpu-id", default=0, type=int)
        parser.add_argument("--init-weight-model", default="", type=str)
        parser.add_argument("base_model")
        args = parser.parse_args()
        return args

    @torch.no_grad()
    def combine_final_model(self):
        """Implements Kaldi-style model ensembling"""
        from _pkwrap import kaldi  # lazy import (kaldi-free decoding)

        kaldi.InstantiateKaldiCuda()
        chain_opts = self.chain_opts
        den_fst_path = os.path.join(chain_opts.dir, "den.fst")
        base_models = chain_opts.base_model.split(",")
        assert len(base_models) > 0
        training_opts = kaldi.chain.CreateChainTrainingOptions(
            chain_opts.l2_regularize,
            chain_opts.out_of_range_regularize,
            chain_opts.leaky_hmm_coefficient,
            chain_opts.xent_regularize,
        )

        moving_average = self.Net(self.chain_opts.output_dim)
        best_mdl = self.Net(self.chain_opts.output_dim)
        moving_average.load_state_dict(torch.load(base_models[0]))
        moving_average.cuda()
        best_mdl = moving_average
        dataset = Wav2vec2EgsDataset(
            "{}/wav.scp".format(chain_opts.dataset),
            chain_opts.egs,
            "{}/utt2len".format(chain_opts.dataset),
            "{}/0.trans_mdl".format(chain_opts.dir),
            "{}/normalization.fst".format(chain_opts.dir),
        )
        compute_objf = lambda mdl: compute_chain_objf(
            mdl,
            dataset,
            den_fst_path,
            training_opts,
            minibatch_size=chain_opts.minibatch_size,
        )

        _, init_objf = compute_objf(moving_average)
        best_objf = init_objf

        model_acc = dict(moving_average.named_parameters())
        num_accumulated = torch.Tensor([1.0]).reshape(1).cuda()
        best_num_to_combine = 1
        if len(base_models) == 1:
            best_mdl = moving_average
            logging.info(
                "Using last iter model (no combining) {}".format(base_models[0])
            )
        else:
            for mdl_name in base_models[1:]:
                this_mdl = self.Net(self.chain_opts.output_dim)
                logging.info("Combining model {}".format(mdl_name))
                this_mdl.load_state_dict(torch.load(mdl_name))
                this_mdl = this_mdl.cuda()
                # TODO(srikanth): check why is this even necessary
                moving_average.cuda()
                num_accumulated += 1.0
                for name, params in this_mdl.named_parameters():
                    model_acc[name].data.mul_(
                        (num_accumulated - 1.0) / (num_accumulated)
                    )
                    model_acc[name].data.add_(params.data.mul_(1.0 / num_accumulated))
                torch.cuda.empty_cache()
                # with try/catch it works most of the time, otherwise I sometime get some kaldi error
                try:
                    _, this_objf = compute_objf(moving_average)
                except Exception as e:
                    logging.warning("Error: ".format(str(e)))
                    try:
                        _, this_objf = compute_objf(moving_average)
                    except Exception as e:
                        logging.warning(
                            "2nd error ommiting this accumulation: ".format(str(e))
                        )
                        logging.info("Won't update best model")
                        continue
                if this_objf > best_objf:
                    best_objf = this_objf
                    best_mdl = moving_average
                    best_num_to_combine = int(num_accumulated.clone().detach())
                    logging.info("Found best model")
                else:
                    logging.info("Won't update best model")
        logging.info("Combined {} models".format(best_num_to_combine))
        logging.info("Initial objf = {}, Final objf = {}".format(init_objf, best_objf))
        best_mdl.cpu()
        torch.save(best_mdl.state_dict(), chain_opts.new_model)
        return self


class ChainE2EModel(ChainModel):
    """Extension of ChainModel to handle Chain E2E training"""

    def get_optimizer(self, model, lr=0.01, weight_decay=0.001, **kwargs):
        optimizer = optim.Adam(model, lr=lr, weight_decay=weight_decay)
        return optimizer

    def train(self):
        """Run one iteration of LF-MMI training

        This is called by
        >>> self.train()

        It will probably be renamed as self.fit() since this seems to be
        the standard way other libraries call the training function.
        """
        from _pkwrap import kaldi  # lazy import (kaldi-free decoding)

        kaldi.InstantiateKaldiCuda()
        chain_opts = self.chain_opts
        lr = chain_opts.lr
        den_fst_path = os.path.join(chain_opts.dir, "den.fst")

        #           load model
        model = self.Net(self.chain_opts.output_dim)

        training_opts = kaldi.chain.CreateChainTrainingOptions(
            chain_opts.l2_regularize,
            chain_opts.out_of_range_regularize,
            chain_opts.leaky_hmm_coefficient,
            chain_opts.xent_regularize,
        )
        logging.info("xent passed as {}".format(chain_opts.xent_regularize))
        model = model.cuda()
        id_iter = int(chain_opts.base_model.split("/")[-1].split(".")[0])
        if hasattr(model, "set_lr_layers_for_optim"):
            optimizer = model.set_lr_layers_for_optim(
                self.get_optimizer,
                lr=chain_opts.lr,
                weight_decay=chain_opts.l2_regularize_factor,
                iter=id_iter,
                total_iter=chain_opts.num_iter,
            )
        else:
            optimizer = self.get_optimizer(
                model.parameters(),
                lr=chain_opts.lr,
                weight_decay=chain_opts.l2_regularize_factor,
            )
        model.load_state_dict(torch.load(chain_opts.base_model))
        dataset = Wav2vec2EgsDataset(
            "{}/wav.scp".format(chain_opts.dataset),
            chain_opts.egs,
            "{}/utt2len".format(chain_opts.dataset),
            "{}/0.trans_mdl".format(chain_opts.dir),
            "{}/normalization.fst".format(chain_opts.dir),
        )
        new_model = train_lfmmi_one_iter(
            model,
            dataset,
            den_fst_path,
            training_opts,
            minibatch_size=chain_opts.minibatch_size,
            grad_acc_steps=chain_opts.grad_acc_steps,
            lr=chain_opts.lr,
            weight_decay=chain_opts.l2_regularize_factor,
            tensorboard=tensorboard.PkwrapTwensorBoard(self),
            optimizer=optimizer,
            sampler=chain_opts.sampler,
        )
        torch.save(new_model.state_dict(), chain_opts.new_model)

        if hasattr(model, "after_one_iter_hook"):
            model.after_one_iter_hook()
