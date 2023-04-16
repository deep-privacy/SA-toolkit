import logging

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_value_

from .dataset import (
    BatchSampler,
    EgsCollectFn,
    GetSupervisionFromEgs,
)

try:
    from _satools import kaldi  # lazy import (kaldi-free decoding)
except ImportError as error:
    pass


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
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, opts, den_graph, supervision, nnet_output_tensor, xent_out_tensor):
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
        nnet_output_copy = (
            nnet_output_tensor.permute(1, 0, 2).reshape(-1, D).contiguous()
        )
        nnet_deriv = torch.zeros_like(nnet_output_copy).contiguous()
        if xent_out_tensor is not None:
            xent_deriv = torch.zeros_like(nnet_output_copy).contiguous()
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
            nnet_deriv = nnet_deriv.reshape(T, mb, D).permute(1, 0, 2).contiguous()
            xent_deriv = xent_deriv.reshape(T, mb, D).permute(1, 0, 2).contiguous()
            xent_objf = (xent_out_tensor * xent_deriv).sum() / (mb * T)
            objf[0] = objf[0] / weight[0]
            # Don't penalize the network like kaldi does (-10.0 loss value)
            # when there is a loss calcluation error.
            if torch.isnan(xent_objf).any() or objf[0] == -10.0:
                ctx.save_for_backward(torch.zeros_like(nnet_deriv).contiguous())
                return objf
            logging.debug(
                "objf={:.4g}, l2={:.4g}, xent_objf={:.4g}".format(
                    objf[0],
                    l2_term[0] / weight[0],
                    xent_objf,
                )
            )
            ctx.save_for_backward(
                nnet_deriv,
                xent_deriv,
                torch.tensor(opts.xent_regularize, requires_grad=False),
            )
        else:
            kaldi.chain.ComputeChainObjfAndDerivNoXent(
                opts,
                den_graph,
                supervision,
                nnet_output_copy,
                objf,
                l2_term,
                weight,
                nnet_deriv,
            )
            nnet_deriv = nnet_deriv.reshape(T, mb, D).permute(1, 0, 2).contiguous()
            xent_deriv = None
            objf[0] = objf[0] / weight[0]
            logging.debug(
                "objf={:.4g}, l2={:.4g}".format(
                    objf[0],
                    l2_term[0] / weight[0],
                )
            )
            ctx.save_for_backward(nnet_deriv)
        # return the derivates in the original order
        return objf

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dummy):
        """returns the derivatives"""
        if len(ctx.saved_tensors) == 3:
            nnet_deriv, xent_deriv, xent_regularize = ctx.saved_tensors
            return None, None, None, -nnet_deriv, -xent_regularize * xent_deriv
        else:
            nnet_deriv = ctx.saved_tensors[0]
            return None, None, None, -nnet_deriv, None


def OnlineNaturalGradient_apply(x, weight, bias):
        if x.dim() == 2 and bias is not None:
            return torch.addmm(bias, x, weight.t())
        else:
            output = x.matmul(weight.t())
            if bias is not None:
                output += bias
            return output


class OnlineNaturalGradient(torch.autograd.Function):
    """A wrapper to NG-SGD class in Kaldi

    This class wraps Natural Gradient implemented in Kaldi by calling
    nnet3's precondition_directions (wrapped through pybind11)
    When implemented as an autograd Function we can easily wrap
    it in a Linear layer. See satools.nn.NaturalAffineTransform.
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x, weight, bias, in_state, out_state):
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
        ctx.save_for_backward(x, weight, bias)
        ctx.states = [in_state, out_state]
        # the code below is based on pytorch's F.linear
        return OnlineNaturalGradient_apply(x, weight, bias)

    @staticmethod
    @torch.no_grad()
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        """Backward pass for NG-SGD layer

        We pass the gradients computed by Pytorch to Kaldi's precondition_directions
        given the states of the layer.
        """
        input, weight, _ = ctx.saved_tensors
        in_state, out_state = ctx.states
        assert (
            in_state != None
        ), "in_state == None - libkaldi-base.so should be in LD_PATH (source ./path.sh)"
        assert (
            out_state != None
        ), "out_state == None - libkaldi-base.so should be in LD_PATH (source ./path.sh)"
        if input.dim() == 3:
            mb, T, D = input.shape
            mb_T = mb * T
        else:
            mb_T, D = input.shape
        input_temp = torch.zeros(
            mb_T, D + 1, device=grad_output.device, requires_grad=False
        ).contiguous()
        input_temp[:, -1] = 1.0
        input_temp[:, :-1].copy_(input.reshape(mb_T, D))
        grad_weight = grad_bias = None
        if grad_output.dim() == 3:
            grad_input = grad_output.matmul(weight)
            grad_input = grad_input.reshape(mb, T, D)
        else:
            grad_input = grad_output.mm(weight)

        in_scale = kaldi.nnet3.precondition_directions(
            in_state, input_temp.clone().detach())

        out_dim = grad_output.shape[-1]
        grad_output_temp = grad_output.view(-1, out_dim)

        out_scale = kaldi.nnet3.precondition_directions(
            out_state, grad_output_temp)

        scale = torch.tensor(in_scale * out_scale, device=grad_output.device)
        grad_output.data.mul_(scale)
        # TODO: check if we should use data member instead?
        grad_weight = grad_output_temp.t().mm(input_temp[:, :-1])
        grad_bias = grad_output_temp.t().mm(input_temp[:, -1].reshape(-1, 1))
        grad_weight.data.mul_(scale)
        grad_bias.data.mul_(scale)
        return grad_input, grad_weight, grad_bias.t(), None, None


def train_lfmmi_one_iter(
    model,
    dataset,
    den_fst_path,
    training_opts,
    minibatch_size=16,
    lr=0.0001,
    weight_decay=0.25,
    print_interval=30,
    grad_acc_steps=1,
    tensorboard=None,
    optimizer=None,
    sampler=" BucketBatch",
):
    """Run one iteration of LF-MMI training

    The function loads the latest model, takes a list of egs, path to denominator
    fst and runs through the merged egs for one iteration of training. This is
    similar to how one iteration of training is completed in Kaldi.

    Args:
        model: Path to pytorch model (.pt file)
        dataset: a EgsDataset dataset
        den_fst_path: path to den.fst file
        training_opts: options of type ChainTrainingOpts
        grad_acc_steps: Number of training steps for which the gradients should be accumulated.
                        Useful to achieve larger effective batch sizes that would not fit in GPU memory.
        minibatch_size:
        lr: learning rate
        print_interval: the interval (a positive integer) to print the loss value
        sampler: BucketBatch, BucketBatchSuffle, BucketBatchSuffleAllowSomePadding or Random

    Returns:
        updated model in CPU
    """

    # this is required to make sure Kaldi uses GPU
    kaldi.InstantiateKaldiCuda()
    if training_opts is None:
        training_opts = kaldi.chain.CreateChainTrainingOptionsDefault()
    den_graph = kaldi.chain.LoadDenominatorGraph(den_fst_path, model.output_dim)
    criterion = KaldiChainObjfFunction.apply
    model = model.cuda()
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    acc_sum = torch.tensor(0.0, requires_grad=False)
    #  scaler = torch.cuda.amp.GradScaler()

    _model = model

    #  if torch.cuda.device_count() > 1:
    #  logging.info("Let's use " + str(torch.cuda.device_count()) + " GPUs!")
    #  model = nn.DataParallel(model.to("cuda:0"))
    #  minibatch_size *= torch.cuda.device_count()
    #  _model = model.module

    add_param = {}
    if sampler == "BucketBatch":
        logging.info("using sequence-length buckets sampler")
        batch_sampler = BatchSampler(
            dataset.egs_holder,
            batch_size=minibatch_size,
            drop_last=False,
        )
        add_param["batch_sampler"]=batch_sampler

    if sampler == "BucketBatchSuffle":
        logging.info("using sequence-length buckets sampler with dataset shuffle")
        batch_sampler = BatchSampler(
            dataset.egs_holder,
            batch_size=minibatch_size,
            drop_last=False,
            ran_iter=True
        )
        add_param["batch_sampler"]=batch_sampler
    if sampler == "BucketBatchSuffleAllowSomePadding":
        logging.info("using sequence-length buckets sampler with dataset shuffle and allow some padding")
        batch_sampler = BatchSampler(
            dataset.egs_holder,
            batch_size=minibatch_size,
            drop_last=False,
            ran_iter=True,
            allow_some_padding=True
        )
        add_param["batch_sampler"]=batch_sampler

    # TODO: make the num_workers configurable (this can significantly speedup the training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        **add_param,
        collate_fn=EgsCollectFn,
        num_workers=5,
    )

    #  for mb_id, data in enumerate(dataloader):
    #  print(mb_id, data[0].shape, GetSupervisionFromEgs(dataset.transition_model, dataset.normalization_fst, data[1], 500), flush=True)

    optimizer.zero_grad()
    for mb_id, data in enumerate(dataloader):
        features = data[0].cuda()

        output, xent_output = model(features)
        #  print("OUT:", output.shape, flush=True)

        num_output_frames = output.shape[1]
        sup = GetSupervisionFromEgs(
            dataset.transition_model,
            dataset.normalization_fst,
            data[1],
            num_output_frames,
        )
        deriv = criterion(training_opts, den_graph, sup, output, xent_output)

        if grad_acc_steps > 1:
            deriv = deriv / grad_acc_steps

        if not hasattr(_model, "trainer_do_backward") or _model.trainer_do_backward:
            deriv.backward()

        acc_sum.add_(deriv[0] * grad_acc_steps)
        if mb_id > 0 and mb_id % print_interval == 0:
            logging.info("Overall objf={}".format(acc_sum / print_interval))
            if tensorboard:
                tensorboard.add_scalar(
                    "ASR_objf/train", acc_sum / print_interval, mb_id
                )
            acc_sum.zero_()

        if hasattr(_model, "additional_obj"):
            _model.additional_obj(
                deriv, data,
                should_log=mb_id > 0 and mb_id % print_interval == 0,
                print_interval=print_interval,
                tensorboard=tensorboard,
                mb_id=mb_id,
            )

        if mb_id == 0 or (mb_id + 1) % (len(dataloader) / 10) == 0:
            SR = 16000
            logging.info(
                "Training with batch_size of:"
                + str((data[0][0].shape[0] * data[0].shape[0] * grad_acc_steps) / SR)
                + " seconds (imprecise estimation)"
            )
        if (mb_id + 1) % grad_acc_steps == 0 or (mb_id + 1 == len(dataloader)):
            clip_grad_value_(_model.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad()

        #  optimizer.zero_grad()
        #  #  scaler.scale(deriv.cuda()).backward()
        #  deriv.backward()
        #  #  scaler.unscale_(optimizer)
        #  clip_grad_value_(_model.parameters(), 5.0)
        #  #  scaler.step(optimizer)
        #  #  scaler.update()
        #  optimizer.step()

        #  return model # fast_test
    model = _model.cpu()
    if tensorboard:
        tensorboard.close()
    return model


@torch.no_grad()
def compute_chain_objf(
    model,
    dataset,
    den_fst_path,
    training_opts,
    minibatch_size=16,
    tensorboard=None,
):
    """Function to compute objective value from a minibatch, useful for diagnositcs.

    Args:
        model: the model to run validation on
        dataset: a EgsDataset dataset
        den_fst_path: path to den.fst
        training_opts: ChainTrainingOpts object
    """
    if training_opts is None:
        training_opts = kaldi.chain.CreateChainTrainingOptionsDefault()
    den_graph = kaldi.chain.LoadDenominatorGraph(den_fst_path, model.output_dim)
    criterion = KaldiChainObjfFunction.apply
    model = model.cuda()
    acc_sum = torch.tensor(0.0, requires_grad=False)
    tot_weight = 0.0

    batch_sampler = BatchSampler(
        dataset.egs_holder,
        batch_size=minibatch_size,
        drop_last=False,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=EgsCollectFn,
        num_workers=5,
    )

    for mb_id, data in enumerate(dataloader):
        features = data[0].cuda()

        output, xent_output = model(features)

        num_output_frames = output.shape[1]
        sup = GetSupervisionFromEgs(
            dataset.transition_model,
            dataset.normalization_fst,
            data[1],
            num_output_frames,
        )
        deriv = criterion(training_opts, den_graph, sup, output, xent_output)

        mb, num_seq = features.shape
        tot_weight += mb * num_seq
        acc_sum.add_(deriv[0] * mb * num_seq)

        if hasattr(model, "additional_obj"):
            model.additional_obj(mb * num_seq, data, for_valid=True)

    objf = acc_sum / tot_weight
    logging.info("Objective = {}".format(objf))
    if tensorboard:
        tensorboard.add_scalar("ASR_objf/valid", objf, 1)

    if hasattr(model, "additional_obj"):
        model.additional_obj(
            0, data, for_valid=True, should_log=True, print_interval=tot_weight, tensorboard=tensorboard
        )

    if tensorboard:
        tensorboard.close()

    model = model.cpu()
    return model, objf
