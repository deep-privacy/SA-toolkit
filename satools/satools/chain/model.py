import argparse
import logging
import os
import sys
import io
import json
from dataclasses import dataclass, fields
import pickle

import kaldiio
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .egs import (
    Wav2vec2EgsDataset,
    Wav2vec2DecodeDataset,
    Wav2vec2EgsCollectFn,
)
from . import objf
from .. import script_utils
import satools


@dataclass
class TrainerOpts:
    dir: str = ""
    egs: str = ""
    grad_acc_steps: int = 1
    init_weight_model: str = ""
    lr: float = 0.001
    mode: str = "init"
    new_model: str = ""
    num_iter: int = -1
    sampler: str = "BucketBatch"
    weight_decay_l2_regularize_factor: float = 1e-5
    xent_regularize: float = 0.01 # LF-MMI
    l2_regularize: float = 1e-4 # LF-MMI
    leaky_hmm_coefficient: float = 0.1 # LF-MMI CONST
    out_of_range_regularize: float = 0.01  # LF-MMI CONST
    minibatch_size: int = 16


@dataclass
class DecodeOpts:
    use_gpu: bool = True
    gpu_id: int = 0
    decode_feats: str = "data/test/feats.scp"
    decode_output: str = "-"


@dataclass
class ChainModelOpts(TrainerOpts, DecodeOpts):
    base_model: str = ""
    base_model_args: str = "{}"
    dataset: str = ""
    output_dim: int = 1

    def load_from_args(self):
        parser = argparse.ArgumentParser(description="")
        for field in fields(self):
            if field.name == "base_model":
                parser.add_argument("base_model") # $1
                continue
            parser.add_argument(f"--{field.name.replace('_', '-')}", default=field.default, type=field.type)
        args = parser.parse_args()
        return self.load_from_config(vars(args))

    def load_from_config(self, cfg):
        for key, value in cfg.items():
            if hasattr(self, key):
                #  type_of_value = self.__annotations__[key]
                type_of_value = type(getattr(self, key))
                setattr(self, key, type_of_value(value))
        logging.info(str(self))
        return self

class ChainModel(nn.Module):
    def __init__(self, model_cls, cmd_line=False, **kwargs):
        """initialize a ChainModel"""
        super(ChainModel, self).__init__()
        assert model_cls is not None
        if cmd_line:
            self.chain_opts = ChainModelOpts()
            self.chain_opts.load_from_args()
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
        model = torch.jit.script(self.Net(self.chain_opts.output_dim))
        model = model.cuda()
        model(torch.rand(1, 16000).cuda())
        if self.chain_opts.init_weight_model != "":
            init_weight_provided = self.load_state_model(self.chain_opts.init_weight_model)

            init_weight_provided_matched, unmatch = satools.utils.torch.match_state_dict(
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

        if hasattr(model, "init"):
            model.init()

        self.save_model(model, self.chain_opts.base_model)

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
        from _satools import kaldi  # lazy import (kaldi-free decoding)

        kaldi.InstantiateKaldiCuda()
        chain_opts = self.chain_opts
        den_fst_path = os.path.join(chain_opts.dir, "den.fst")

        model = self.Net(self.chain_opts.output_dim)
        model.load_state_dict(self.load_state_model(chain_opts.base_model))
        #  if torch.__version__.startswith("2."):
            #  model = torch.compile(model, dynamic=True)
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
        objf.compute_chain_objf(
            model,
            dataset,
            den_fst_path,
            training_opts,
            minibatch_size=chain_opts.minibatch_size,
            tensorboard=satools.tensorboard.SATwensorBoard(self)
            if "valid" in self.chain_opts.egs
            else None,
        )

    @torch.no_grad()
    def merge(self):
        chain_opts = self.chain_opts
        base_models = chain_opts.base_model.split(",")
        assert len(base_models) > 0
        model0 = self.Net(self.chain_opts.output_dim)
        model0.load_state_dict(self.load_state_model(base_models[0]))
        model_acc = dict(model0.named_parameters())
        for mdl_name in base_models[1:]:
            this_mdl = self.Net(self.chain_opts.output_dim)
            this_mdl.load_state_dict(self.load_state_model(mdl_name))
            for name, params in this_mdl.named_parameters():
                # Only average layers that are trained otherwise we ran
                # into chained numerical division imprecision
                if params.requires_grad:
                    model_acc[name].data.add_(params.data)
        weight = 1.0 / len(base_models)
        for name in model_acc:
            if model_acc[name].requires_grad:
                model_acc[name].data.mul_(weight)
        self.save_model(model0)

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
                model.load_state_dict(self.load_state_model(base_model))
            except Exception as e:
                logging.warning("Warning cannot load model {}".format(base_model))
                logging.warning("Retrying with strict=False")
                #  logging.warning(e)
                not_inited = model.load_state_dict(self.load_state_model(base_model), strict=False)
                logging.warning("Incompatible layers: {}".format(not_inited))

        model.eval()
        if share_memory:
            model.share_memory()

        def _forward(waveform):
            with torch.no_grad():
                post, xent_output = model(waveform)
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
            from _satools import kaldi  # lazy import (kaldi-free decoding)
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

    @torch.no_grad()
    def combine_final_model(self):
        """Implements Kaldi-style model ensembling"""
        from _satools import kaldi  # lazy import (kaldi-free decoding)

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
        moving_average.load_state_dict(self.load_state_model(base_models[0]))
        moving_average.cuda()
        best_mdl = moving_average
        dataset = Wav2vec2EgsDataset(
            "{}/wav.scp".format(chain_opts.dataset),
            chain_opts.egs,
            "{}/utt2len".format(chain_opts.dataset),
            "{}/0.trans_mdl".format(chain_opts.dir),
            "{}/normalization.fst".format(chain_opts.dir),
        )
        compute_objf = lambda mdl: objf.compute_chain_objf(
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
                this_mdl.load_state_dict(self.load_state_model(mdl_name))
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
        self.save_model(best_mdl)
        return self

    def load_state_model(self, file):
        m = torch.load(file)
        if "base_model_args_state_dict" in m:
            return m["base_model_args_state_dict"]
        return m

    def save_model(self, model, file=None):
        file = self.chain_opts.new_model if file==None else file
        install_path = os.path.dirname(os.path.dirname(satools.__path__[0])) # dir to git clone

        source_state_dict = model.state_dict()
        model = self.Net(output_dim=self.chain_opts.output_dim)
        model = torch.jit.script(model)
        model.load_state_dict(source_state_dict)
        buffer = io.BytesIO()
        torch.jit.save(model, buffer)

        #  import torch, io; m = torch.jit.load(io.BytesIO(torch.load("exp/chain/e2e_tdnnf_t100_kaldifeat_b/0.pt")['base_model_jit'])); m

        torch.save({"base_model_args_state_dict": model.state_dict(),
                    "base_model_jit": buffer.getvalue(),
                    "task_path": os.getcwd().replace(install_path, ""),
                    "install_path": install_path,
                    "base_model_path": sys.argv[0],
                    "base_model_output_dim": self.chain_opts.output_dim,
                    "base_model_args": json.loads(self.chain_opts.base_model_args),
                    }, file)


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
        chain_opts = self.chain_opts
        lr = chain_opts.lr
        den_fst_path = os.path.join(chain_opts.dir, "den.fst")

        # load model
        model = torch.jit.script(self.Net(self.chain_opts.output_dim))

        from _satools import kaldi  # lazy import (kaldi-free decoding)

        kaldi.InstantiateKaldiCuda()

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
                weight_decay=chain_opts.weight_decay_l2_regularize_factor,
                iter=id_iter,
                total_iter=chain_opts.num_iter,
            )
        else:
            optimizer = self.get_optimizer(
                model.parameters(),
                lr=chain_opts.lr,
                weight_decay=chain_opts.weight_decay_l2_regularize_factor,
            )
        model.load_state_dict(self.load_state_model(chain_opts.base_model))
        #  if torch.__version__.startswith("2."):
            #  model = torch.compile(model, dynamic=True)
        dataset = Wav2vec2EgsDataset(
            "{}/wav.scp".format(chain_opts.dataset),
            chain_opts.egs,
            "{}/utt2len".format(chain_opts.dataset),
            "{}/0.trans_mdl".format(chain_opts.dir),
            "{}/normalization.fst".format(chain_opts.dir),
        )
        new_model = objf.train_lfmmi_one_iter(
            model,
            dataset,
            den_fst_path,
            training_opts,
            minibatch_size=chain_opts.minibatch_size,
            grad_acc_steps=chain_opts.grad_acc_steps,
            lr=chain_opts.lr,
            weight_decay=chain_opts.weight_decay_l2_regularize_factor,
            tensorboard=satools.tensorboard.SATwensorBoard(self),
            optimizer=optimizer,
            sampler=chain_opts.sampler,
        )
        self.save_model(new_model)

        if hasattr(model, "after_one_iter_hook"):
            model.after_one_iter_hook()
