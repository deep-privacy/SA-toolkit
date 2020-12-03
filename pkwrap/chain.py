# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

import sys
import os
import random
from collections import OrderedDict, Counter
import logging
import argparse
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_
import torch.optim as optim
from _pkwrap import kaldi
from . import chain
from . import matrix
from . import script_utils

class KaldiChainObjfFunction(torch.autograd.Function):
    """LF-MMI objective function for pytorch

    This Function wraps MMI loss function implemented in Kaldi.
    See Pytorch documentation of how to extend Function to check
    the attributes present. I expect that this class will be used
    as follows

    ```
        lfmmi_loss = KaldiChainObjfFunction.apply
        ...
        lfmmi_loss(chain_opts, den_graph, egs, nnet_output, xent_output)
    ```
    """
    @staticmethod
    def forward(ctx, opts, den_graph, supervision, nnet_output_tensor,
                xent_out_tensor):
        """This function computes the loss for a single minibatch. 

        This function calls Kaldi's ComputeChainObjfAndDeriv through our
        pybind11 wrapper. It takes the network outputs, rearranges them
        in the way Kaldi expects, gets back the derivates of the outputs.
        We pre-allocate the space for derivatives before passing to Kaldi.
        No extra space is used by Kaldi as we pass only the poitners.

        Args:
            opts: training options for the loss function
            den_graph: Denominator graph
            supervision: merged egs for the current minibatch
            nnet_output_tensor: output generated by the network
            xent_out_tensor: the corresponding cross-entropy output

        Returns:
            We normally don't use the output returned by the function.
            The derivatives are stored in the context and used by the backward()
            function.
        """
        objf = torch.zeros(1, requires_grad=False)
        l2_term = torch.zeros(1, requires_grad=False)
        weight = torch.zeros(1, requires_grad=False)
        mb, T, D = nnet_output_tensor.shape
        # Kaldi expects the outputs to be groups by time frames. So
        # we need to permut the output
        nnet_output_copy = nnet_output_tensor.permute(1, 0, 2).reshape(-1, D).contiguous()
        nnet_deriv = torch.zeros_like(nnet_output_copy)
        xent_deriv = torch.zeros_like(nnet_output_copy)
        kaldi.chain.ComputeChainObjfAndDeriv(
            opts,
            den_graph,
            supervision,
            nnet_output_copy,
            objf,
            l2_term,
            weight,
            nnet_deriv,
            xent_deriv,
        )
        # return the derivates in the original order
        nnet_deriv = nnet_deriv.reshape(T, mb, D).permute(1, 0, 2)
        xent_deriv = xent_deriv.reshape(T, mb, D).permute(1, 0, 2)

        ctx.save_for_backward(nnet_deriv, xent_deriv)
        with torch.no_grad():
            xent_objf = (xent_out_tensor*xent_deriv).sum()/(mb*T)
            objf[0] = objf[0]/weight[0]
            sys.stderr.write(
                "objf={}, l2={}, xent_objf={}\n".format(
                    objf[0],
                    l2_term[0]/weight[0],
                    xent_objf,
                )
            )
        return objf

    @staticmethod
    def backward(ctx, dummy):
        """returns the derivatives"""
        nnet_deriv, xent_deriv = ctx.saved_tensors
        return None, None, None, -nnet_deriv, -0.1*xent_deriv


class OnlineNaturalGradient(torch.autograd.Function):
    """A wrapper to NG-SGD class in Kaldi

    This class wraps Natural Gradient implemented in Kaldi by calling
    nnet3's precondition_directions (wrapped through pybind11)
    When implemented as an autograd Function we can easily wrap
    it in a Linear layer. See pkwrap.nn.NaturalAffineTransform.
    """
    @staticmethod
    def forward(ctx, input, weight, bias, in_state, out_state):
        """Forward pass for NG-SGD layer
        
        Args:
            input: the input to the layer (a Tensor)
            weight: weight matrix of the layer (a Tensor)
            bias: the bias parameters of the layer (a Tensor)
            in_state: state of the input (a kaldi.nnet3.OnlineNaturalGradient object)
            out_state: state of the output (a kaldi.nnet3.OnlineNaturalGradient object)
        
        Returns:
            Linear transformation of the input with weight and bias.
            The other inputs are saved in the context to be used during the call
            to backward.
        """
        ctx.save_for_backward(input, weight, bias)
        ctx.states = [in_state, out_state]
        # the code below is based on pytorch's F.linear
        if input.dim() == 2 and bias is not None:
            output = torch.addmm(bias, input, weight.t())
        else:
            output = input.matmul(weight.t())
            if bias is not None:
                output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for NG-SGD layer

        We pass the gradients computed by Pytorch to Kaldi's precondition_directions
        given the states of the layer.
        """
        input, weight, _ = ctx.saved_tensors
        in_state, out_state = ctx.states
        if input.dim() == 3:
            mb, T, D = input.shape
            mb_T = mb*T
        else:
            mb_T, D = input.shape
        input_temp = torch.zeros(mb_T, D+1, device=input.device, requires_grad=False).contiguous()
        input_temp[:,-1] = 1.0
        input_temp[:,:-1].copy_(input.reshape(mb_T, D))
        grad_weight = grad_bias = None
        if grad_output.dim() == 3:
            grad_input = grad_output.matmul(weight)
            grad_input = grad_input.reshape(mb, T, D)
        else:
            grad_input = grad_output.mm(weight)
        in_scale = kaldi.nnet3.precondition_directions(in_state, input_temp)
        out_dim = grad_output.shape[-1]
        grad_output_temp = grad_output.view(-1, out_dim)
        out_scale = kaldi.nnet3.precondition_directions(out_state, grad_output_temp) # hope grad_output is continguous!
        scale = in_scale*out_scale
        grad_output.data.mul_(scale)
        # TODO: check if we should use data member instead?
        grad_weight = grad_output_temp.t().mm(input_temp[:,:-1])
        grad_bias = grad_output_temp.t().mm(input_temp[:,-1].reshape(-1,1))
        grad_weight.data.mul_(scale)
        grad_bias.data.mul_(scale)
        return grad_input, grad_weight, grad_bias.t(), None, None


# take a scp file
class ChainExample(torch.utils.data.Dataset):
    """A Dataset wrapper to egs objects in Kaldi

    This is a generic wrapper to handling egs files generated by Kaldi.
    With this class we can iterate over the examples easily.
    """
    def __init__(self, egs_file, output_file=None):
        """Initialize a ChainExample object

        Given a egs_file, currently we support only scp files, we load
        the keys and pointers in the ark file into a dictionary, so that
        we don't have to load the entire egs file in memory

        Args:
            egs_file: scp file containing entries of egs
            output_file: this argument is used when each egs may belong to different languages
        """
        if output_file and egs_file.startswith('scp:'):
            raise ValueError("need egs_file to start to be of type scp when using output_file")
        self.egs_file = egs_file
#       TODO: error handling
        egs_list = [ln.strip().split() for ln in open(egs_file)]
        self.egs_dict = OrderedDict(egs_list)
        self.egs_keys = [x for x in self.egs_dict]
        if output_file:
            self.output_file = output_file
            self.lang_ids = dict([ln.strip().split() for ln in open(output_file)])
        else:
            self.output_file = None

    def __len__(self):
        return len(self.egs_dict)

    def __getitem__(self, idx):
        # key is the utterance id. it is likely to be a sub-utterance id
        key = self.egs_keys[idx]
        value = self.egs_dict[key]
        # if the output_file was passed, then language ids should exist
        if self.output_file:
            if key in self.lang_ids:
                lang_id = self.lang_ids[key]
            else:
                lang_id = -1
            return (key, value, lang_id)
        else:
            return (key, value, lang_id)
        
def load_egs(egs_file):
    """Loads the contents of the egs file.

    Given an egs file created for chain model training, load the
    contents of the and return as an array of NnetChainExample

    Args:
        egs_file: scp or ark file, should be prefix accordingly just like Kaldi
    
    Returns:
        A list of NnetChainExample
    """
    return kaldi.chain.ReadChainEgsFile(egs_file, 0)

def prepare_minibatch(egs_file, minibatch_size):
    """Prepare an array of minibatches from an egs file

    It loads the contents of the egs_file in memory, shuffles them
    and returns an array of minibatches.

    Args:
        egs_file: scp or ark file (a string), should be prefix accordingly just like Kaldi
        minibatch_size: a string of minibatch sizes separated by commas. E.g "64" or "128,64"
    
    Returns:
        A list of NnetChainExample. Each item contains merged examples with number of 
        sequences as given in the minibatch_size
    """
    egs = load_egs(egs_file)
    random.shuffle(egs)
    merged_egs = kaldi.chain.MergeChainEgs(egs, str(minibatch_size))
    return merged_egs

def train_lfmmi_one_iter(model, egs_file, den_fst_path, training_opts, feat_dim, 
                         minibatch_size="64", use_gpu=True, lr=0.0001, 
                         weight_decay=0.25, frame_shift=0, 
                         left_context=0,
                         right_context=0,
                         print_interval=10,
                         frame_subsampling_factor=3,
    ):
    """Run one iteration of LF-MMI training

    The function loads the latest model, takes a list of egs, path to denominator
    fst and runs through the merged egs for one iteration of training. This is 
    similar to how one iteration of training is completed in Kaldi.

    Args:
        model: Path to pytorch model (.pt file)
        egs_file: scp or ark file (a string), should be prefix accordingly just like Kaldi
        den_fst_path: path to den.fst file
        training_opts: options of type ChainTrainingOpts
        feat_dim: dimension of features (e.g. 40 for MFCC hires features)
        minibatch_size: a string of minibatch sizes separated by commas. E.g "64" or "128,64"
        use_gpu: a boolean to set or unset the use of GPUs while training
        lr: learning rate
        frame_shift: an integer (usually 0, 1, or 2) used to shift the training features
        print_interval: the interval (a positive integer) to print the loss value

    Returns:
        updated model in CPU
    """
    # this is required to make sure Kaldi uses GPU
    kaldi.InstantiateKaldiCuda()
    if training_opts is None:
        training_opts = kaldi.chain.CreateChainTrainingOptionsDefault()
    den_graph = kaldi.chain.LoadDenominatorGraph(den_fst_path, model.output_dim)
    criterion = KaldiChainObjfFunction.apply
    if use_gpu:
        model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    acc_sum = torch.tensor(0., requires_grad=False)
    for mb_id, merged_egs in enumerate(prepare_minibatch(egs_file, minibatch_size)):
        chunk_size = kaldi.chain.GetFramesPerSequence(merged_egs)*frame_subsampling_factor
        features = kaldi.chain.GetFeaturesFromEgs(merged_egs)
        features = features[:,frame_shift:frame_shift+chunk_size+left_context+right_context,:]
        features = features.cuda()
        output, xent_output = model(features)
        sup = kaldi.chain.GetSupervisionFromEgs(merged_egs)
        deriv = criterion(training_opts, den_graph, sup, output, xent_output)
        acc_sum.add_(deriv[0])
        if mb_id>0 and mb_id%print_interval==0:
            logging.info("Overall objf={}\n".format(acc_sum/print_interval))
            acc_sum.zero_()
        optimizer.zero_grad()
        deriv.backward()
        clip_grad_value_(model.parameters(), 5.0)
        optimizer.step()
    model = model.cpu()
    return model

def compute_chain_objf(model, egs_file, den_fst_path, training_opts, feat_dim, 
    minibatch_size="64", use_gpu=True, lr=0.0001, weight_decay=0.25, frame_shift=0, 
    left_context=0,
    right_context=0,
    frame_subsampling_factor=3):
    """Function to compute objective value from a minibatch, useful for diagnositcs"""
    kaldi.InstantiateKaldiCuda()
    if training_opts is None:
        training_opts = kaldi.chain.CreateChainTrainingOptionsDefault()
    den_graph = kaldi.chain.LoadDenominatorGraph(den_fst_path, model.output_dim)
    criterion = chain.KaldiChainObjfFunction.apply
    if use_gpu:
        model = model.cuda()
    acc_sum = torch.tensor(0., requires_grad=False)
    tot_weight = 0.
    for mb_id, merged_egs in enumerate(prepare_minibatch(egs_file, minibatch_size)):
        chunk_size = kaldi.chain.GetFramesPerSequence(merged_egs)*frame_subsampling_factor
        features = kaldi.chain.GetFeaturesFromEgs(merged_egs)
        features = features[:,frame_shift:frame_shift+chunk_size+left_context+right_context,:]
        features = features.cuda()
        output, xent_output = model(features)
        sup = kaldi.chain.GetSupervisionFromEgs(merged_egs)
        deriv = criterion(training_opts, den_graph, sup, output, xent_output)
        mb, num_seq, _ = features.shape
        tot_weight += mb*num_seq
        acc_sum.add_(deriv[0]*mb*num_seq)
    logging.info("Objective = {}".format(acc_sum/tot_weight))
    model = model.cpu()
    return model


@dataclass
class TrainerOpts:
    mode: str = "init"
    dir: str = ""
    lr: float = 0.001
    minibatch_size: int = 32
    base_model: str = ''

@dataclass
class DecodeOpts:
    decode_feats: str = 'data/test/feats.scp'
    decode_output: str = '-'

@dataclass
class ChainModelOpts(TrainerOpts, DecodeOpts):
    egs: str = ""
    new_model: str = ""
    l2_regularize: float = 1e-4
    l2_regularize_factor: float = 1.0
    out_of_range_regularize: float = 0.01
    leaky_hmm_coefficient: float = 0.1
    xent_regularize: float = 0.025
    minibatch_size: str = "32"
    frame_shift: int = 0
    output_dim: int = 1
    feat_dim: int = 1
    context: int = 0
    frame_subsampling_factor: int = 3
    
    def load_from_config(self, cfg):
        for key, value in cfg.items():
            if hasattr(self, key):
                type_of_value = type(getattr(self, key))
                setattr(self, key, type_of_value(value))
        return self


class ChainModel(nn.Module):
    def __init__(self, model_cls, cmd_line=False, **kwargs):
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
        self.reset_dims()
        if self.chain_opts.mode != 'context':
            self.load_context()
        if self.chain_opts.mode == 'init':
            self.init()
        elif self.chain_opts.mode == 'context':
            self.context()
        elif self.chain_opts.mode == 'merge':
            self.merge()
        elif self.chain_opts.mode in ['validate', 'diagnostic'] :
            self.validate()
        elif self.chain_opts.mode in ['train', 'training']:
            self.train()
        elif self.chain_opts.mode in ['decode', 'infer']:
            self.infer()

    def init(self):
        model = self.Net(self.chain_opts.feat_dim, self.chain_opts.output_dim)
        torch.save(model.state_dict(), self.chain_opts.base_model)

    def train(self):
        chain_opts = self.chain_opts
        lr = chain_opts.lr
        den_fst_path = os.path.join(chain_opts.dir, "den.fst")

#           load model
        model = self.Net(self.chain_opts.feat_dim, self.chain_opts.output_dim)
        model.load_state_dict(torch.load(chain_opts.base_model))

        training_opts = kaldi.chain.CreateChainTrainingOptions(
                chain_opts.l2_regularize, 
                chain_opts.out_of_range_regularize, 
                chain_opts.leaky_hmm_coefficient, 
                chain_opts.xent_regularize,
        )
        context = chain_opts.context 
        new_model = train_lfmmi_one_iter(
            model,
            chain_opts.egs, 
            den_fst_path, 
            training_opts, 
            chain_opts.feat_dim, 
            minibatch_size=chain_opts.minibatch_size, 
            left_context=context,
            right_context=context,
            lr=chain_opts.lr,
            weight_decay=chain_opts.l2_regularize_factor,
            frame_shift=chain_opts.frame_shift
        )
        torch.save(new_model.state_dict(), chain_opts.new_model)

    def validate(self):
        chain_opts = self.chain_opts
        den_fst_path = os.path.join(chain_opts.dir, "den.fst")

#           load model
        model = self.Net(self.chain_opts.feat_dim, self.chain_opts.output_dim)
        model.load_state_dict(torch.load(chain_opts.base_model))

        training_opts = kaldi.chain.CreateChainTrainingOptions(
                chain_opts.l2_regularize, 
                chain_opts.out_of_range_regularize, 
                chain_opts.leaky_hmm_coefficient, 
                chain_opts.xent_regularize,
        ) 
        with torch.no_grad():
            compute_chain_objf(
                model,
                chain_opts.egs, 
                den_fst_path, 
                training_opts, 
                chain_opts.feat_dim, 
                minibatch_size="1:64",
                left_context=chain_opts.context,
                right_context=chain_opts.context,
                lr=chain_opts.lr,
                weight_decay=chain_opts.l2_regularize_factor,
                frame_shift=chain_opts.frame_shift,
            )

    def merge(self):
        with torch.no_grad():
            chain_opts = self.chain_opts
            base_models = chain_opts.base_model.split(',')
            assert len(base_models)>0
            model0 = self.Net(self.chain_opts.feat_dim, self.chain_opts.output_dim)
            model0.load_state_dict(torch.load(base_models[0]))
            model_acc = dict(model0.named_parameters())
            for mdl_name in base_models[1:]:
                this_mdl = self.Net(self.chain_opts.feat_dim, self.chain_opts.output_dim)
                this_mdl.load_state_dict(torch.load(mdl_name))
                for name, params in this_mdl.named_parameters():
                    model_acc[name].data.add_(params.data)
            weight = 1.0/len(base_models)
            for name in model_acc:
                model_acc[name].data.mul_(weight)
            torch.save(model0.state_dict(), chain_opts.new_model)

    def infer(self):
        with torch.no_grad():
            chain_opts = self.chain_opts
            model = self.Net(chain_opts.feat_dim, chain_opts.output_dim)
            base_model = chain_opts.base_model
            try:
                model.load_state_dict(torch.load(base_model))
            except Exception as e:
                logging.error(e)
                logging.error("Cannot load model {}".format(base_model))
                quit(1)
            # TODO(srikanth): make sure context is a member of chain_opts
            context = chain_opts.context
            model.eval()
            writer_spec = "ark,t:{}".format(chain_opts.decode_output)
            writer = script_utils.feat_writer(writer_spec)
            for key, feats in script_utils.feat_reader_gen(chain_opts.decode_feats):
                feats_with_context = matrix.add_context(feats, context, context).unsqueeze(0)
                post, _ = model(feats_with_context)
                post = post.squeeze(0)
                writer.Write(key, kaldi.matrix.TensorToKaldiMatrix(post))
                logging.info("Wrote {}\n ".format(key))
            writer.Close()

    def context(self):
        """Find context by brute force

        WARNING: it only works for frame_subsampling_factor=3
        """

        logging.warning("context function called. it only works for frame_subsampling_factor=3")
        visited = Counter()
        with torch.no_grad():
          feat_dim = 40
          num_pdfs = 300 
          model = self.Net(40, 300)
          # output len = ceil(chunk size / frame_subsampling_factor)
          chunk_sizes = [(150,50), (50, 17), (100, 34), (10, 4), (20, 7)]
          frame_shift = 0
          left_context = 0
          logging.info("Searching for context...")
          while True:
              right_context = left_context
              found = []
              for chunk_len, output_len in chunk_sizes:
                  feat_len = chunk_len+left_context+right_context
                  assert feat_len > 0
                  try:
                      test_feats = torch.zeros(32, feat_len, feat_dim)
                      y = model(test_feats)
                  except Exception as e:
                      visited[left_context] += 1
                      if visited[left_context] > 10:
                          break
                  if y[0].shape[1] == output_len:
                      found.append(True)
                  else:
                      found.append(False)
                      break
              if all(found):
                      self.save_context(left_context)
                      return
              left_context += 1
              if left_context >= 100:
                  raise NotImplementedError("more than context of 100")
          raise Exception("No context found")

    def save_context(self, value):
          logging.info(f"Left_context = {value}")
          with open(os.path.join(self.chain_opts.dir, 'context'), 'w') as opf:
              opf.write(f'{value}')
              opf.close()

    def load_context(self):
        self.chain_opts.context = script_utils.read_single_param_file(
                os.path.join(self.chain_opts.dir, 'context'),
        )

    def reset_dims(self):
        # what if the user wants to pass it? Just override this function
        num_pdfs_filename = os.path.join(
            self.chain_opts.dir, 
            "num_pdfs"
        )
        self.chain_opts.output_dim = script_utils.read_single_param_file(num_pdfs_filename)

        feat_dim_filename = os.path.join(
            self.chain_opts.dir, 
            "feat_dim"
        )
        # checking this because we don't always need feat_dim (e.g. when
        # generating context)
        if os.path.isfile(feat_dim_filename):
            self.chain_opts.feat_dim = script_utils.read_single_param_file(feat_dim_filename)


    def load_model_context(self):
        context_file_name = os.path.join(self.chain_opts.dir, 'context')
        context = script_utils.read_single_param_file(context_file_name)

    def load_cmdline_args(self):
        parser = argparse.ArgumentParser(description="")
        parser.add_argument("--mode", default="init")
        parser.add_argument("--dir", default="")
        parser.add_argument("--lr", default=0.001, type=float)
        parser.add_argument("--egs", default="")
        parser.add_argument("--new-model", default="")
        parser.add_argument("--l2-regularize", default=1e-4, type=float)
        parser.add_argument("--l2-regularize-factor", default=1.0, type=float) # this is the weight_decay in pytorch
        parser.add_argument("--out-of-range-regularize", default=0.01, type=float)
        parser.add_argument("--xent-regularize", default=0.025, type=float)
        parser.add_argument("--leaky-hmm-coefficient", default=0.1, type=float)
        parser.add_argument("--minibatch-size", default="32", type=str)
        parser.add_argument("--decode-feats", default="data/test/feats.scp", type=str)
        parser.add_argument("--decode-output", default="-", type=str)
        parser.add_argument("--decode-iter", default="final", type=str)
        parser.add_argument("--frame-shift", default=0, type=int)
        parser.add_argument("base_model")
        args = parser.parse_args()
        return args

