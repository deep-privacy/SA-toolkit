import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
from . import nn as sann

from typing import Tuple, Union, Dict, Optional
import logging

import warnings
warnings.filterwarnings(
    "ignore", message=r'.*ComplexHalf support is experimental and many operators.*'
)
warnings.filterwarnings(
    "ignore",
    message="torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.",
)



class CoreHifiGan(torch.nn.Module):
    def __init__(
        self,
        upsample_rates=[5,4,4,2,2],
        upsample_kernel_sizes=[11,8,8,4,4],
        imput_dim=256+1,  # BN asr = 256 dim + F0 dim + ....
        upsample_initial_channel=512,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        iSTFTNetout=False,
        iSTFTNet_n_fft = 16,
    ):
        super().__init__()

        self.iSTFTNetout = iSTFTNetout
        self.post_n_fft = iSTFTNet_n_fft
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.num_kernels = len(resblock_kernel_sizes)
        self.upsample_rates = upsample_rates
        self.conv_pre = weight_norm(
            nn.Conv1d(imput_dim, upsample_initial_channel, 7, 1, padding=3)
        )

        resblock = sann.ResBlock1
        #  resblock = pkwrap.nn.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i, v in enumerate(self.ups):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        if self.iSTFTNetout:
            self.conv_post = weight_norm(nn.Conv1d(ch, self.post_n_fft + 2, 7, 1, padding=3))
        else:
            self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(sann.init_weights)
        self.conv_post.apply(sann.init_weights)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))

    def forward_resnet(self, x):
        x = self.conv_pre(x)
        for up_idx, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            xs = torch.zeros_like(x)
            for resblock_idx, resblock in enumerate(self.resblocks):
                if up_idx * self.num_kernels <= resblock_idx < (up_idx + 1) * self.num_kernels:
                    xs += resblock(x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.reflection_pad(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        return Tuple[torch.Tensor]
        Signal if not self.iSTFTNetout
        Spec and Phase if self.iSTFTNetout
        """

        x = self.forward_resnet(x)

        if self.iSTFTNetout:
            spec = torch.exp(x[:,:self.post_n_fft // 2 + 1, :])
            phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])
            return spec, phase

        return (x, torch.empty((1)))

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)



# https://github.com/rishikksh20/iSTFTNet-pytorch
class iSTFTNet(torch.nn.Module):
    def __init__(self, n_fft=800, hop_length=200, win_length=800):
        super().__init__()
        self.filter_length = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(self.win_length, periodic=True)

    def transform(self, input_data):
        forward_transform = torch.stft(
            input_data,
            self.filter_length, self.hop_length, self.win_length, window=self.window,
            return_complex=True)

        return torch.abs(forward_transform), torch.angle(forward_transform)

    @torch.autocast(device_type="cuda", enabled=False)
    def inverse(self, magnitude, phase):
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(magnitude.device))

        return inverse_transform.unsqueeze(-2)  # unsqueeze to stay consistent with conv_transpose1d implementation

    def forward(self, input_data):
        magnitude, phase = self.transform(input_data)
        reconstruction = self.inverse(magnitude, phase)
        return reconstruction
