# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

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
from .egs_wav2vec2 import Wav2vec2EgsDataset, Wav2vec2DecodeDataset, Wav2vec2EgsCollectFn

import kaldiio
import matplotlib.pyplot as plt
from damped import disturb


@dataclass
class TrainerOpts:
    mode: str = ""
    dir: str = ""
    lr: float = 0.001
    minibatch_size: int = 16
    grad_acc_steps: int = 1
    base_model: str = ''
    init_weight_model: str = ''

@dataclass
class DecodeOpts:
    use_gpu: bool = False
    gpu_id: int = 0
    decode_feats: str = 'data/test/feats.scp'
    decode_output: str = '-'

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
    grad_acc_steps: int = 1
    frame_shift: int = 0
    output_dim: int = 1
    frame_subsampling_factor: int = 3

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
        if self.chain_opts.mode == 'init':
            self.init()
        elif self.chain_opts.mode == 'merge':
            self.merge()
        elif self.chain_opts.mode in ['validate', 'diagnostic'] :
            if "valid" in self.chain_opts.egs:
                disturb.init(all_to_one=True, rank=1)
                disturb.eval(all_to_one=True)
            self.validate()
            if "valid" in self.chain_opts.egs:
                disturb.train(all_to_one=True)
                disturb.stop(all_to_one=True)
        elif self.chain_opts.mode in ['train', 'training']:
            disturb.init(all_to_one=True)
            disturb.train(all_to_one=True)
            self.train()
            disturb.stop(all_to_one=True)
        elif self.chain_opts.mode in ['decode', 'infer']:
            #  disturb.init(all_to_one=True)
            #  disturb.eval(all_to_one=True)
            self.infer()
            #  disturb.stop(all_to_one=True)
        elif self.chain_opts.mode == 'final_combination':
            self.combine_final_model()
        elif self.chain_opts.mode == 'codebook_analysis':
            self.codebook_analysis()

    def init(self):
        """Initialize the model and save it in chain_opts.base_model"""
        model = self.Net(self.chain_opts.output_dim)
        if self.chain_opts.init_weight_model != "":
            init_weight_provided = torch.load(self.chain_opts.init_weight_model)
            init_weight_provided_matched, unmatch = utils.torch.match_state_dict(model.state_dict(), init_weight_provided)
            not_inited = model.load_state_dict(init_weight_provided_matched, strict=False)
            logging.info("Init from previous model {}, layers not initialized: {}: layers ommited (wrong shape): {}".format(self.chain_opts.init_weight_model, str(not_inited), str(unmatch.keys())))
        torch.save(model.state_dict(), self.chain_opts.base_model)

    def train(self):
        """Run one iteration of LF-MMI training

        This is called by
        >>> self.train()

        It will probably be renamed as self.fit() since this seems to be
        the standard way other libraries call the training function.
        """
        raise "Only implementing e2e LF-MMI"

    @torch.no_grad()
    def validate(self):
        from _pkwrap import kaldi # lazy import (kaldi-free decoding)
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
            tensorboard=tensorboard.PkwrapTwensorBoard(self) if "valid" in self.chain_opts.egs else None,
        )

    @torch.no_grad()
    def merge(self):
        chain_opts = self.chain_opts
        base_models = chain_opts.base_model.split(',')
        assert len(base_models)>0
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
        weight = 1.0/len(base_models)
        for name in model_acc:
            if model_acc[name].requires_grad:
                model_acc[name].data.mul_(weight)
        torch.save(model0.state_dict(), chain_opts.new_model)

    @torch.no_grad()
    def codebook_analysis(self):
        chain_opts = self.chain_opts
        model = self.Net(chain_opts.output_dim)
        base_model = chain_opts.base_model
        model = model.to(device)
        try:
            model.load_state_dict(torch.load(base_model))
        except Exception as e:
            logging.error(e)
            logging.error("Cannot load model {}".format(base_model))
            sys.exit(1)

        if not hasattr(model, 'vq') or not model.vq():
            logging.error("Cannot analyise non VQ model: {}".format(base_model))
            sys.exit(1)
        if not hasattr(model, 'codebook_analysis'):
            logging.error("Cannot analyise VQ model no 'codebook_analysis' attribute found in the model definition: {}".format(base_model))
            sys.exit(1)
        codebook = model.codebook_analysis().embedding.weight.data.cpu()

        mds = MDS(n_components=2, random_state=0, metric='cosine')
        proj = mds.fit_transform(codebook)
        plt.scatter(proj[:,0], proj[:,1], alpha=0.3)
        savepath = os.path.join(os.path.dirname(base_model), "codebook_analysis_mds.png")
        plt.savefig(savepath)
        plt.clf()

        tsne = TSNE(n_components=2, random_state=0, metric='cosine', square_distances=True)
        proj = tsne.fit_transform(codebook)
        plt.scatter(proj[:,0], proj[:,1], alpha=0.3)
        savepath = os.path.join(os.path.dirname(base_model), "codebook_analysis_tsne.png")
        plt.savefig(savepath)

        logging.info("saved scatters to {}".format(os.path.dirname(savepath)))

    @torch.no_grad()
    def get_forward(self, device=torch.device("cpu"), share_memory=False, get_model_module=False):
        chain_opts = self.chain_opts

        model = self.Net(chain_opts.output_dim)
        base_model = chain_opts.base_model
        model = model.to(device)
        try:
            model.load_state_dict(torch.load(base_model))
        except Exception as e:
            logging.error(e)
            logging.error("Cannot load model {}".format(base_model))
            sys.exit(1)

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
            run_on_gpu = (list(range(0, torch.cuda.device_count())) * 200)[chain_opts.gpu_id]
            device = torch.device("cuda:{}".format(run_on_gpu))
            logging.info("Using GPU: {}".format(run_on_gpu))

        model = self.get_forward(device=device)

        write_with_kaldi=True
        try:
            from _pkwrap import kaldi # lazy import (kaldi-free decoding)
        except ImportError as error:
             # shutil/decode/latgen-faster-mapped.sh compatible but slower
            logging.critical(" -- Failed to import kaldi for feat writing --")
            logging.exception(error)
            write_with_kaldi=False

        if write_with_kaldi:
            writer_spec = "ark,t:{}".format(chain_opts.decode_output)
            writer = script_utils.feat_writer(writer_spec)
            close=writer.Close
            writer=writer.Write
            tensor_to_writer= lambda x: kaldi.matrix.TensorToKaldiMatrix(x)
        else:
            logging.info(" -- Using Kaldiio for feat writing --")
            writer = kaldiio.WriteHelper('ark,t:{}'.format(chain_opts.decode_output))
            close=writer.close
            tensor_to_writer= lambda x: x.numpy()

        dataset = Wav2vec2DecodeDataset.from_wav_scp(chain_opts.decode_feats)
        dataloader = torch.utils.data.DataLoader(dataset, collate_fn=Wav2vec2EgsCollectFn, num_workers=8)

        for feats, key in dataloader:
            if chain_opts.use_gpu:
                feats = feats.to(device)
            post, _ = model(feats)
            post = post.squeeze(0).cpu()
             # batch size = 1 !!
            writer(key[0], tensor_to_writer(post))
            logging.info("Wrote {}".format(key[0]))
        close()

    def reset_dims(self):
        # what if the user wants to pass it? Just override this function
        num_pdfs_filename = os.path.join(
            self.chain_opts.dir,
            "num_pdfs"
        )
        self.chain_opts.output_dim = script_utils.read_single_param_file(num_pdfs_filename)


    def load_cmdline_args(self):
        parser = argparse.ArgumentParser(description="")
        parser.add_argument("--mode", default="init")
        parser.add_argument("--dir", default="")
        parser.add_argument("--lr", default=0.001, type=float)
        parser.add_argument("--egs", default="")
        parser.add_argument("--dataset", default="")
        parser.add_argument("--new-model", default="")
        parser.add_argument("--l2-regularize", default=1e-4, type=float)
        parser.add_argument("--l2-regularize-factor", default=1.0, type=float) # this is the weight_decay in pytorch
        parser.add_argument("--out-of-range-regularize", default=0.01, type=float)
        parser.add_argument("--xent-regularize", default=0.025, type=float)
        parser.add_argument("--leaky-hmm-coefficient", default=0.1, type=float)
        parser.add_argument("--minibatch-size", default=32, type=int)
        parser.add_argument("--grad-acc-steps", default=1, type=int)
        parser.add_argument("--decode-feats", default="data/test/feats.scp", type=str)
        parser.add_argument("--decode-output", default="-", type=str)
        parser.add_argument("--decode-iter", default="final", type=str)
        parser.add_argument("--frame-shift", default=0, type=int)
        parser.add_argument("--use-gpu", default=False, type=bool)
        parser.add_argument("--gpu-id", default=0, type=int)
        parser.add_argument("--init-weight-model", default='', type=str)
        parser.add_argument("base_model")
        args = parser.parse_args()
        return args

    @torch.no_grad()
    def combine_final_model(self):
        """Implements Kaldi-style model ensembling"""
        from _pkwrap import kaldi # lazy import (kaldi-free decoding)
        kaldi.InstantiateKaldiCuda()
        chain_opts = self.chain_opts
        den_fst_path = os.path.join(chain_opts.dir, "den.fst")
        base_models = chain_opts.base_model.split(',')
        assert len(base_models)>0
        training_opts = kaldi.chain.CreateChainTrainingOptions(
                chain_opts.l2_regularize,
                chain_opts.out_of_range_regularize,
                chain_opts.leaky_hmm_coefficient,
                chain_opts.xent_regularize,
        )

        moving_average = self.Net(self.chain_opts.output_dim)
        best_mdl =  self.Net(self.chain_opts.output_dim)
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
            frame_shift=chain_opts.frame_shift,
        )

        _, init_objf = compute_objf(moving_average)
        best_objf = init_objf

        model_acc = dict(moving_average.named_parameters())
        num_accumulated = torch.Tensor([1.0]).reshape(1).cuda()
        best_num_to_combine = 1
        if len(base_models) == 1:
            best_mdl = moving_average
            logging.info("Using last iter model (no combining) {}".format(base_models[0]))
        else:
            for mdl_name in base_models[1:]:
                this_mdl = self.Net(self.chain_opts.output_dim)
                logging.info("Combining model {}".format(mdl_name))
                this_mdl.load_state_dict(torch.load(mdl_name))
                this_mdl = this_mdl.cuda()
                # TODO(srikanth): check why is this even necessary
                moving_average.cuda()
                num_accumulated += 1.
                for name, params in this_mdl.named_parameters():
                    model_acc[name].data.mul_((num_accumulated-1.)/(num_accumulated))
                    model_acc[name].data.add_(params.data.mul_(1./num_accumulated))
                # with try/catch it works all the time, otherwise I might get some kaldi error
                try:
                    _, this_objf = compute_objf(moving_average)
                except Exception as e:
                    logging.warining("Error: ".format(str(e)))
                    _, this_objf = compute_objf(moving_average)
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
        optimizer = optim.Adam(
            model,
            lr=lr,
            weight_decay=weight_decay
        )
        return optimizer

    def train(self):
        """Run one iteration of LF-MMI training

        This is called by
        >>> self.train()

        It will probably be renamed as self.fit() since this seems to be
        the standard way other libraries call the training function.
        """
        from _pkwrap import kaldi # lazy import (kaldi-free decoding)
        kaldi.InstantiateKaldiCuda()
        chain_opts = self.chain_opts
        lr = chain_opts.lr
        den_fst_path = os.path.join(chain_opts.dir, "den.fst")

#           load model
        model = self.Net(self.chain_opts.output_dim)
        model.load_state_dict(torch.load(chain_opts.base_model))

        training_opts = kaldi.chain.CreateChainTrainingOptions(
                chain_opts.l2_regularize,
                chain_opts.out_of_range_regularize,
                chain_opts.leaky_hmm_coefficient,
                chain_opts.xent_regularize,
        )
        logging.info("xent passed as {}".format(chain_opts.xent_regularize))
        model = model.cuda()
        id_iter = int(chain_opts.base_model.split("/")[-1].split(".")[0])
        if hasattr(model, 'set_lr_layers_for_optim'):
            optimizer = model.set_lr_layers_for_optim(self.get_optimizer, lr=chain_opts.lr, weight_decay=chain_opts.l2_regularize_factor, iter=id_iter)
        else:
            optimizer = self.get_optimizer(model.parameters(), lr=chain_opts.lr, weight_decay=chain_opts.l2_regularize_factor)
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
            frame_shift=chain_opts.frame_shift,
            tensorboard = tensorboard.PkwrapTwensorBoard(self),
            optimizer=optimizer,
            e2e = True,
        )
        torch.save(new_model.state_dict(), chain_opts.new_model)

