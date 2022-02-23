""" Neural network architectures and relevant utility functions"""
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

import torch
import torch.nn as nn
import torch.nn.functional as F
from .objf import OnlineNaturalGradient

from dataclasses import dataclass
from itertools import combinations, product
import logging
from .. import nsg


@dataclass
class NGState:
    """NGState value container"""

    alpha: float = 4.0
    num_samples_history: float = 2000.0
    update_period: int = 4


log_kaldi_warning = True


# KALDI preconditioner
def get_preconditioner_from_ngstate(ngstate):
    assert ngstate is not None
    global log_kaldi_warning
    try:
        from _pkwrap import kaldi  # lazy import (kaldi-free decoding)
    except ImportError as error:
        if log_kaldi_warning:
            logging.critical(
                "pkwrap: -- Failed to import kaldi you better not be in training mode (no backward possible) --"
            )
            log_kaldi_warning = False
        return None
        pass
    preconditioner = kaldi.nnet3.OnlineNaturalGradient()
    preconditioner.SetAlpha(ngstate.alpha)
    preconditioner.SetNumSamplesHistory(ngstate.num_samples_history)
    preconditioner.SetUpdatePeriod(ngstate.update_period)
    return preconditioner


class NaturalAffineTransform(nn.Module):
    """Linear layer wrapped in NG-SGD

    This is an implementation of NaturalGradientAffineTransform in Kaldi.
    It wraps the linear transformation with chain.OnlineNaturalGradient to
    achieve this.

    Attributes:
        feat_dim: (int, required) input dimension of the transformation
        out_dim: (int, required) output dimension of the transformation
        bias: (bool, optional) set False to not use bias. True by default.
        ngstate: a dataclass of type NGState containing the following keys
            alpha: a floating point value (default is 4.0)
            num_samples_history: a floating point value (default is 2000.)
            update_period: an integer (default is 4)
    """

    def __init__(
        self,
        feat_dim,
        out_dim,
        bias=True,
        ngstate=None,
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
        super(NaturalAffineTransform, self).__init__()
        self.feat_dim = feat_dim
        self.out_dim = out_dim
        self.ngstate = NGState()
        if self.ngstate is None:
            self.ngstate = NGState()
        # lazyinit (not required for decoding enables kaldi free execution)
        self.preconditioner_in = None
        self.preconditioner_out = None

        # For pytorch only
        self.all_preconditioner_in = {}
        self.all_preconditioner_out = {}
        self.weight = nn.Parameter(torch.Tensor(out_dim, feat_dim))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_dim))
        else:
            self.register_parameter("bias", None)
        self.init_parameters()

    def init_parameters(self):
        """Initialize the parameters (weight and bias) of the layer"""

        self.weight.data.normal_()
        self.weight.data.mul_(1.0 / pow(self.feat_dim * self.out_dim, 0.5))
        self.bias.data.normal_()

    def forward(self, input):
        """Forward pass"""

        # PyTorch only but slower
        #  if torch.cuda.device_count() > 1:
        #  if input.device not in self.all_preconditioner_in:
        #  self.all_preconditioner_in[input.device] = nsg.OnlineNaturalGradient()
        #  self.all_preconditioner_out[input.device] = nsg.OnlineNaturalGradient()
        #  return chain.OnlineNaturalGradient.apply(
        #  input,
        #  self.weight,
        #  self.bias,
        #  self.all_preconditioner_in[input.device],
        #  self.all_preconditioner_out[input.device],
        #  )

        # kaldi lazyinit (not used if OnlineNaturalGradient)
        if (
            self.training
            and self.weight.requires_grad
            and self.preconditioner_in == None
            and self.preconditioner_out == None
        ):
            self.preconditioner_in = get_preconditioner_from_ngstate(self.ngstate)
            self.preconditioner_out = get_preconditioner_from_ngstate(self.ngstate)

        return OnlineNaturalGradient.apply(
            input,
            self.weight,
            self.bias,
            self.preconditioner_in,
            self.preconditioner_out,
        )


@torch.no_grad()
def constrain_orthonormal(M, scale, update_speed=0.125):
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
    P[range(d), range(d)] -= scale2
    M.data.add_(P.mm(M), alpha=-4 * update_speed / scale2)


class OrthonormalLinear(NaturalAffineTransform):
    def __init__(
        self,
        feat_dim,
        out_dim,
        bias=True,
        scale=0.0,
        ngstate=NGState(),
    ):
        super(OrthonormalLinear, self).__init__(
            feat_dim, out_dim, bias=bias, ngstate=ngstate
        )
        self.scale = torch.tensor(scale, requires_grad=False)

    def forward(self, input):
        """Forward pass"""
        # do it before forward pass
        if self.training and self.weight.requires_grad:
            with torch.no_grad():
                constrain_orthonormal(self.weight, self.scale)
        x = super().forward(input)
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
        floating_scale=True,
        bypass_scale=0.66,
    ):
        super(TDNNF, self).__init__()
        # lets keep it context_len for now
        self.linearB = OrthonormalLinear(
            feat_dim * context_len, bottleneck_dim, scale=orthonormal_constraint
        )
        self.linearA = nn.Linear(bottleneck_dim, output_dim)
        self.output_dim = torch.tensor(output_dim, requires_grad=False)
        self.bottleneck_dim = torch.tensor(bottleneck_dim, requires_grad=False)
        self.feat_dim = torch.tensor(feat_dim, requires_grad=False)
        self.subsampling_factor = torch.tensor(subsampling_factor, requires_grad=False)
        self.context_len = torch.tensor(context_len, requires_grad=False)
        self.orthonormal_constraint = torch.tensor(
            orthonormal_constraint, requires_grad=False
        )
        self.bypass_scale = torch.tensor(bypass_scale, requires_grad=False)
        if bypass_scale > 0.0 and feat_dim == output_dim:
            self.use_bypass = True
            if self.context_len > 1:
                if self.context_len % 2 == 1:
                    #  self.identity_lidx = self.context_len // 2
                    self.identity_lidx  = torch.div(self.context_len, 2, rounding_mode='trunc')
                    self.identity_ridx = -self.identity_lidx
                else:
                    #  self.identity_lidx = self.context_len // 2
                    self.identity_lidx  = torch.div(self.context_len, 2, rounding_mode='trunc')
                    self.identity_ridx = -self.identity_lidx + 1
            else:
                self.use_bypass = False
        else:
            self.use_bypass = False

    def forward(self, input):
        mb, T, D = input.shape
        padded_input = (
            input.reshape(mb, -1)
            .unfold(1, D * self.context_len, D * self.subsampling_factor)
            .contiguous()
        )
        x = self.linearB(padded_input)
        x = self.linearA(x)
        if self.use_bypass:
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
    ):
        super(TDNNFBatchNorm, self).__init__()
        self.tdnn = TDNNF(
            feat_dim,
            output_dim,
            bottleneck_dim,
            context_len=context_len,
            subsampling_factor=subsampling_factor,
            orthonormal_constraint=orthonormal_constraint,
            bypass_scale=bypass_scale,
        )
        self.bn = nn.BatchNorm1d(output_dim, affine=False)
        self.output_dim = torch.tensor(output_dim, requires_grad=False)

    def forward(self, input):
        mb, T, D = input.shape
        x = self.tdnn(input)
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        x = F.relu(x)
        return x


class TDNNF_LD(nn.Module):
    def __init__(
        self,
        feat_dim,
        output_dim,
        bottleneck_dim,
        context_len=1,
        subsampling_factor=1,
        orthonormal_constraint=0.0,
        floating_scale=True,
        bypass_scale=0.66,
        bottleneck_ld=lambda x: x,
        bottleneck_ld_outdim=None,
    ):
        super(TDNNF_LD, self).__init__()
        if bottleneck_ld_outdim == None:
            bottleneck_ld_outdim = bottleneck_dim
        # lets keep it context_len for now
        self.bottleneck_ld = bottleneck_ld
        self.linearB = OrthonormalLinear(
            feat_dim * context_len, bottleneck_dim, scale=orthonormal_constraint
        )
        self.linearA = nn.Linear(bottleneck_ld_outdim, output_dim)
        self.output_dim = torch.tensor(output_dim, requires_grad=False)
        self.bottleneck_dim = torch.tensor(bottleneck_dim, requires_grad=False)
        self.feat_dim = torch.tensor(feat_dim, requires_grad=False)
        self.subsampling_factor = torch.tensor(subsampling_factor, requires_grad=False)
        self.context_len = torch.tensor(context_len, requires_grad=False)
        self.orthonormal_constraint = torch.tensor(
            orthonormal_constraint, requires_grad=False
        )
        self.bypass_scale = torch.tensor(bypass_scale, requires_grad=False)
        if bypass_scale > 0.0 and feat_dim == output_dim:
            self.use_bypass = True
            if self.context_len > 1:
                if self.context_len % 2 == 1:
                    self.identity_lidx = self.context_len // 2
                    self.identity_ridx = -self.identity_lidx
                else:
                    self.identity_lidx = self.context_len // 2
                    self.identity_ridx = -self.identity_lidx + 1
            else:
                self.use_bypass = False
        else:
            self.use_bypass = False

    def forward(self, input):
        mb, T, D = input.shape
        padded_input = (
            input.reshape(mb, -1)
            .unfold(1, D * self.context_len, D * self.subsampling_factor)
            .contiguous()
        )
        x = self.linearB(padded_input)
        x = self.bottleneck_ld(x)
        x = self.linearA(x)
        if self.use_bypass:
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


class TDNNFBatchNorm_LD(nn.Module):
    def __init__(
        self,
        feat_dim,
        output_dim,
        bottleneck_dim,
        context_len=1,
        subsampling_factor=1,
        orthonormal_constraint=0.0,
        bypass_scale=0.66,
        bottleneck_ld=lambda x: x,
        bottleneck_ld_outdim=None,
    ):
        super(TDNNFBatchNorm_LD, self).__init__()
        self.in_dim = feat_dim
        self.out_dim = output_dim
        self.bottleneck_dim = bottleneck_dim
        self.bottleneck_ld = bottleneck_ld
        self.tdnn = TDNNF_LD(
            feat_dim,
            output_dim,
            bottleneck_dim,
            context_len=context_len,
            subsampling_factor=subsampling_factor,
            orthonormal_constraint=orthonormal_constraint,
            bypass_scale=bypass_scale,
            bottleneck_ld=bottleneck_ld,
            bottleneck_ld_outdim=bottleneck_ld_outdim,
        )
        self.bn = nn.BatchNorm1d(output_dim, affine=False)
        self.output_dim = torch.tensor(output_dim, requires_grad=False)

    def forward(self, input):
        mb, T, D = input.shape
        x = self.tdnn(input)
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
        super(VectorQuantizerEMA, self).__init__()

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

    def forward(
        self, inputs, compute_distances_if_possible=False, record_codebook_stats=False
    ):
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

        # Compute distances between encoding vectors | n(n-1)/2 where n = T (i.e.: T = 10 -> len(encoding_distances) == 45)
        if not self.training and compute_distances_if_possible:
            _encoding_distances = [
                torch.dist(items[0], items[1], 2).to(self._device)
                for items in combinations(flat_input, r=2)
            ]
            encoding_distances = torch.tensor(_encoding_distances).to(self._device)
        else:
            encoding_distances = None

        # Compute distances between embedding vectors | n(n-1)/2 where n = num_embeddings (i.e.: num_embeddings = 2 -> len(embedding_distances) == 1)
        if not self.training and compute_distances_if_possible:
            _embedding_distances = [
                torch.dist(items[0], items[1], 2).to(self._device)
                for items in combinations(self._embedding.weight, r=2)
            ]
            embedding_distances = torch.tensor(_embedding_distances).to(self._device)
        else:
            embedding_distances = None

        # Sample nearest embedding | if T = 10 & num_embeddings == 2 -> 10*2 distance tensor
        if not self.training and compute_distances_if_possible:
            _frames_vs_embedding_distances = [
                torch.dist(items[0], items[1], 2).to(self._device)
                for items in product(flat_input, self._embedding.weight.detach())
            ]
            frames_vs_embedding_distances = (
                torch.tensor(_frames_vs_embedding_distances)
                .to(self._device)
                .view(batch_size, time, -1)
            )
        else:
            frames_vs_embedding_distances = None

        # Use EMA to update the embedding vectors
        if self.training and not self.freeze:
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

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        # TODO: Check if the more readable self._embedding.weight.index_select(dim=1, index=encoding_indices) works better

        concatenated_quantized = None
        #  concatenated_quantized = self._embedding.weight[torch.argmin(distances, dim=1).detach().cpu()] if not self.training or record_codebook_stats else None

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
            {"vq_loss": vq_loss.item()},
            encoding_distances,
            embedding_distances,
            frames_vs_embedding_distances,
            concatenated_quantized,
        )

    @property
    def embedding(self):
        return self._embedding
