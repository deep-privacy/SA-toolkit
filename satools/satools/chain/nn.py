""" Neural network architectures and relevant utility functions"""

import logging
from dataclasses import dataclass
from itertools import combinations, product

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import List, Union, Dict, Optional, Callable, TypeVar, Any

from .objf import OnlineNaturalGradient, OnlineNaturalGradient_apply


log_kaldi_warning = False
try:
    from _satools import kaldi  # lazy import (kaldi-free decoding)
except ImportError as error:
    log_kaldi_warning = True


@dataclass
class NGState:
    """NGState value container"""

    alpha: float = 4.0
    num_samples_history: float = 2000.0
    update_period: float = 4.0

    def asdict(self) -> Dict[str, float]:
        return {"alpha":self.alpha,
                "num_samples_history":self.num_samples_history,
                "update_period":self.update_period}


def get_preconditioner_from_ngstate(ngstate: Dict[str, float]):
    # KALDI preconditioner
    global log_kaldi_warning
    if log_kaldi_warning:
        logging.critical(
            "satools: -- Failed to import kaldi you better not be in training mode (no backward possible) --"
        )
        log_kaldi_warning = False
        return None
    preconditioner = kaldi.nnet3.OnlineNaturalGradient()
    preconditioner.SetAlpha(ngstate['alpha'])
    preconditioner.SetNumSamplesHistory(ngstate['num_samples_history'])
    preconditioner.SetUpdatePeriod(int(ngstate['update_period']))
    return preconditioner


class NaturalAffineTransform(nn.Module):
    """Linear layer wrapped in NG-SGD

    This is an implementation of NaturalGradientAffineTransform in Kaldi.
    It wraps the linear transformation with chain.OnlineNaturalGradient to
    achieve this.
    """

    def __init__(
        self,
        feat_dim,
        out_dim,
        ngstate=None,
        bias=True,
    ):
        """Initialize NaturalGradientAffineTransform layer

        The function initializes NG-SGD states and parameters of the layer

        Args:
            feat_dim: (int, required) input dimension of the transformation
            out_dim: (int, required) output dimension of the transformation
            bias: (bool, optional) set False to not use bias. True by default.
            ngstate: a dictionary containing the following keys
                alpha: a floating point value (default is 4.0)
                num_samples_history: a floating point value (default is 2000.)
                update_period: an integer (default is 4)

        Returns:
            NaturalAffineTransform object
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.out_dim = out_dim
        if ngstate is None:
            ngstate = NGState()
        self.ngstate = ngstate.asdict()
        # lazyinit (not required for decoding enables kaldi free execution)
        self.preconditioner_in = None
        self.preconditioner_out = None
        self.preconditioner_init = False

        self.weight = nn.Parameter(torch.Tensor(out_dim, feat_dim))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_dim))
        else:
            self.register_parameter("bias", None)
        self.init_parameters()

    def __repr__(self):
        return ("{}(feat_dim={}, out_dim={})").format(
            self.__class__.__name__,
            self.feat_dim,
            self.out_dim,
        )

    def init_parameters(self):
        """Initialize the parameters (weight and bias) of the layer"""

        self.weight.data.normal_()
        self.weight.data.mul_(1.0 / pow(self.feat_dim * self.out_dim, 0.5))
        self.bias.data.normal_()

    @torch.jit.unused
    def _train_forward(self, input):
        if not self.preconditioner_init:
            self.preconditioner_init = True
            self.preconditioner_in = get_preconditioner_from_ngstate(self.ngstate)
            self.preconditioner_out = get_preconditioner_from_ngstate(self.ngstate)
        return OnlineNaturalGradient.apply(
            input,
            self.weight,
            self.bias,
            self.preconditioner_in,
            self.preconditioner_out,
        )

    def forward(self, x):
        if self.training and self.weight.requires_grad:
            # DOES NOT WORK WITH JIT MODEL, ONLY FOR EVAL MODE
            return self._train_forward(x)
        else:
            return OnlineNaturalGradient_apply(x, self.weight, self.bias)


@torch.no_grad()
def constrain_orthonormal(M: Tensor, scale:float, update_speed:float=0.125):
    rows, cols = M.shape
    d = rows
    if rows < cols:
        M = M.T
        d = cols
    # we don't update it. we just compute the gradient
    P = M.mm(M.T)

    if scale < 0.0:
        trace_P_Pt = P.pow(2.0).sum()
        trace_P = P.trace()
        ratio = trace_P_Pt / trace_P
        scale = ratio.sqrt()
        ratio = ratio * d / trace_P
        if ratio > 1.1:
            update_speed *= 0.25
        elif ratio > 1.02:
            update_speed *= 0.5
    scale2 = scale ** 2
    P[list(range(d)), list(range(d))] -= scale2
    M.data.add_(P.mm(M), alpha=-4 * update_speed / scale2)


class OrthonormalLinear(nn.Module):
    def __init__(
        self,
        feat_dim,
        out_dim,
        bias=True,
        scale=0.0,
        ngstate=NGState(),
    ):
        super().__init__()
        self.inner_nat = NaturalAffineTransform(feat_dim, out_dim, bias=bias, ngstate=ngstate)
        self.scale = torch.tensor(scale, requires_grad=False)

    def forward(self, input):
        """Forward pass"""
        # do it before forward pass
        if self.training and self.inner_nat.weight.requires_grad:
            with torch.no_grad():
                constrain_orthonormal(self.inner_nat.weight, self.scale)
        x = self.inner_nat.forward(input)
        return x


class PassThrough(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_dim="same"
    def forward(self, x:Tensor) -> Tensor:
        return x



class TDNNF(nn.Module):
    def __init__(
        self,
        feat_dim,
        output_dim,
        bottleneck_dim,
        context_len=1,
        subsampling_factor=1,
        orthonormal_constraint=0.0,
        bypass_scale=0.66,
        bottleneck_func=PassThrough(),
    ):
        super().__init__()
        self.bottleneck_func = bottleneck_func
        self.bottleneck_outdim = bottleneck_dim
        if self.bottleneck_func.output_dim != "same":
            self.bottleneck_outdim = self.bottleneck_func.output_dim

        self.linearB = OrthonormalLinear(
            feat_dim * context_len, bottleneck_dim, scale=orthonormal_constraint
        )
        self.linearA = nn.Linear(self.bottleneck_outdim, output_dim)
        self.output_dim = torch.tensor(output_dim, requires_grad=False)
        self.bottleneck_dim = torch.tensor(bottleneck_dim, requires_grad=False)
        self.feat_dim = torch.tensor(feat_dim, requires_grad=False)
        self.subsampling_factor = torch.tensor(subsampling_factor, requires_grad=False)
        self.context_len = torch.tensor(context_len, requires_grad=False)
        self.orthonormal_constraint = torch.tensor(
            orthonormal_constraint, requires_grad=False
        )

        self.bypass_scale = torch.tensor(bypass_scale, requires_grad=False)
        self.identity_lidx = torch.tensor(0, requires_grad=False)  # Start
        self.identity_ridx = None  # End
        self.use_bypass = False
        self.indices = torch.tensor(0, requires_grad=False)
        if bypass_scale > 0.0 and feat_dim == output_dim:
            self.use_bypass = True
            if self.context_len > 1:
                if self.context_len % 2 == 1:
                    self.identity_lidx = torch.div(
                        self.context_len, 2, rounding_mode="trunc"
                    )
                    self.identity_ridx = -self.identity_lidx
                else:
                    self.identity_lidx = torch.div(
                        self.context_len, 2, rounding_mode="trunc"
                    )
                    self.identity_ridx = -self.identity_lidx + 1
                if self.context_len == 2:
                    self.identity_lidx = torch.tensor(1, requires_grad=False)  # Start
                    self.identity_ridx = None  # End

            #                               / at the subsampled rate, not in seconds
            self.indices = torch.arange(0, 16000*100, 1.5, requires_grad=False).long()

    def __repr__(self):
        return ("{}(feat_dim={}, out_dim={}, bypass_scale={}, orthonormal_constraint={}, subsampling_factor={}, context_len={}\n\t linearB={}, bottleneck_func={}, linearA={})").format(
            self.__class__.__name__,
            self.feat_dim,
            self.output_dim,
            self.bypass_scale,
            self.orthonormal_constraint,
            self.subsampling_factor,
            self.context_len,
            self.linearB,
            self.bottleneck_func,
            self.linearA,
        )

    def forward(self, input, return_bottleneck:bool=False):
        mb, T, D = input.shape
        padded_input = (
            input.reshape(mb, -1)
            .unfold(1, D * self.context_len, int(D * self.subsampling_factor))
            .contiguous()
        )
        x = self.linearB(padded_input)
        x = self.bottleneck_func(x)
        if return_bottleneck:
            return x
        x = self.linearA(x)
        if self.use_bypass:
            if self.subsampling_factor == 1.5:
                x = self.add_padd(x, input, self.indices.to(device=x.device), self.bypass_scale)
            else:
                x = (
                    x
                    + input[
                        :,
                        self.identity_lidx : self.identity_ridx : self.subsampling_factor,
                        :,
                    ]
                    * self.bypass_scale
                )
        return x

    def add_padd(self, x:torch.Tensor, input:torch.Tensor, indices:torch.Tensor, bypass_scale:torch.Tensor) -> torch.Tensor:
        y = torch.index_select(input, 1, indices[:int(input.shape[1]/1.5)]) * bypass_scale
        # pad the smaller tensor with zeros along the first dimension to match the size of the larger tensor
        if x.size(1) < y.size(1):
            x = torch.nn.functional.pad(x, [0, 0, 0, y.size(1)-x.size(1), 0, 0])
        else:
            y = torch.nn.functional.pad(y, [0, 0, 0, x.size(1)-y.size(1), 0, 0])

        # broadcast the smaller tensor to the shape of the larger tensor and add them together
        x = torch.broadcast_tensors(x, y)[0] + y
        return x



class TDNNFBatchNorm(nn.Module):
    def __init__(
        self,
        feat_dim,
        output_dim,
        bottleneck_dim,
        context_len=1,
        subsampling_factor=1,
        orthonormal_constraint=0.0,
        bypass_scale=0.66,
        bottleneck_func=PassThrough(),
    ):
        super().__init__()
        self.in_dim = feat_dim
        self.out_dim = output_dim
        self.bottleneck_dim = bottleneck_dim
        self.bottleneck_func = bottleneck_func
        self.tdnn = TDNNF(
            feat_dim,
            output_dim,
            bottleneck_dim,
            context_len=context_len,
            subsampling_factor=subsampling_factor,
            orthonormal_constraint=orthonormal_constraint,
            bypass_scale=bypass_scale,
            bottleneck_func=bottleneck_func,
        )
        self.bn = nn.BatchNorm1d(output_dim, affine=False)
        self.output_dim = torch.tensor(output_dim, requires_grad=False)

    def forward(self, input, return_bottleneck:bool=False):
        mb, T, D = input.shape
        x = self.tdnn(input, return_bottleneck=return_bottleneck)
        if return_bottleneck:
            return x
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        x = F.relu(x)
        return x


#  https://github.com/swasun/VQ-VAE-Speech/blob/3c537c17465bf59855f0b81d9265354f65016563/src/models/vector_quantizer_ema.py
class VectorQuantizerEMA(nn.Module):
    """
    Inspired from Sonnet implementation of VQ-VAE https://arxiv.org/abs/1711.00937,
    in https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py and
    pytorch implementation of it from zalandoresearch in https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb.
    Implements a slightly modified version of the algorithm presented in
    'Neural Discrete Representation Learning' by van den Oord et al.
    https://arxiv.org/abs/1711.00937
    The difference between VectorQuantizerEMA and VectorQuantizer is that
    this module uses exponential moving averages to update the embedding vectors
    instead of an auxiliary loss. This has the advantage that the embedding
    updates are independent of the choice of optimizer (SGD, RMSProp, Adam, K-Fac,
    ...) used for the encoder, decoder and other parts of the architecture. For
    most experiments the EMA version trains faster than the non-EMA version.
    Input any tensor to be quantized. Last dimension will be used as space in
    which to quantize. All other dimensions will be flattened and will be seen
    as different examples to quantize.
    The output tensor will have the same shape as the input.
    For example a tensor with shape [16, 32, 32, 64] will be reshaped into
    [16384, 64] and all 16384 vectors (each of 64 dimensions)  will be quantized
    independently.
    Args:
        embedding_dim: integer representing the dimensionality of the tensors in the
            quantized space. Inputs to the modules must be in this format as well.
        num_embeddings: integer, the number of vectors in the quantized space.
        commitment_cost: scalar which controls the weighting of the loss terms (see
            equation 4 in the paper).
        decay: float, decay for the moving averages.
        epsilon: small float constant to avoid numerical instability.
    """

    def __init__(
        self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5
    ):
        super().__init__()

        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon
        self.freeze = False

    def forward(self, inputs):
        """
        Connects the module to some inputs.
        Args:
            inputs: Tensor, final dimension must be equal to embedding_dim. All other
                leading dimensions will be flattened and treated as a large batch.

        Returns:
            loss: Tensor containing the loss to optimize.
            quantize: Tensor containing the quantized version of the input.
            perplexity: Tensor containing the perplexity of the encodings.
            encodings: Tensor containing the discrete encodings, ie which element
                of the quantized space each input element was mapped to.
            distances
        """

        # input x is of shape: [N, T, C]

        input_shape = inputs.shape
        batch_size, time, _ = input_shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)  # [T, C]

        # Compute distances between encoded audio frames and embedding vectors
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        self._device = inputs.device

        """
        encoding_indices: Tensor containing the discrete encoding indices, ie
        which element of the quantized space each input element was mapped to.
        """
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, dtype=torch.float
        ).to(self._device)
        encodings.scatter_(1, encoding_indices, 1)

        # Use EMA to update the embedding vectors
        if self.training and not self.freeze:
            # DOES NOT WORK WITH JIT MODEL, ONLY FOR EVAL MODE
            self.ema_update(flat_input, encodings)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        # TODO: Check if the more readable self._embedding.weight.index_select(dim=1, index=encoding_indices) works better

        # Loss
        e_latent_loss = torch.mean((quantized.detach() - inputs) ** 2)
        commitment_loss = self._commitment_cost * e_latent_loss
        vq_loss = commitment_loss

        quantized = inputs + (quantized - inputs).detach()

        """
        The perplexity a useful value to track during training.
        It indicates how many codes are 'active' on average.
        """
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Convert quantized from [N, T, C] (image origin: BHWC) -> [C, T, N] (image origin: BCHW)
        return (
            vq_loss,
            quantized.contiguous(),
            perplexity,
            encodings,
            distances,
            encoding_indices,
        )

    @torch.jit.unused
    def ema_update(self, flat_input, encodings):
        # DOES NOT WORK WITH JIT MODEL, ONLY FOR EVAL MODE
        self._ema_cluster_size = self._ema_cluster_size * self._decay + (
            1 - self._decay
        ) * torch.sum(encodings, 0)

        n = torch.sum(self._ema_cluster_size.data)
        self._ema_cluster_size = (
            (self._ema_cluster_size + self._epsilon)
            / (n + self._num_embeddings * self._epsilon)
            * n
        )

        dw = torch.matmul(encodings.t(), flat_input)
        self._ema_w = nn.Parameter(
            self._ema_w * self._decay + (1 - self._decay) * dw
        )

        self._embedding.weight = nn.Parameter(
            self._ema_w / self._ema_cluster_size.unsqueeze(1)
        )

    @property
    def embedding(self):
        return self._embedding


class RevGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, alpha=1.0):
        ctx.save_for_backward(input_, torch.tensor(alpha, requires_grad=False))
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None
