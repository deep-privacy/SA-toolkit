# -*- coding: utf-8 -*-
#
# This file is part of SIDEKIT.
#
# SIDEKIT is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#
# SIDEKIT is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#
# SIDEKIT is free software: you can redistribute it and/or modify
# it under the terms of the GNU LLesser General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# SIDEKIT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with SIDEKIT.  If not, see <http://www.gnu.org/licenses/>.

"""
Copyright 2014-2021 Anthony Larcher
"""

import torch
import torchaudio


__author__ = "Anthony Larcher and Sylvain Meignier"
__copyright__ = "Copyright 2014-2021 Anthony Larcher and Sylvain Meignier"
__license__ = "LGPL"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


class FeatureMapScaling(torch.nn.Module):
    """

    """
    def __init__(self, nb_dim, do_add = True, do_mul = True):
        """

        :param nb_dim:
        :param do_add:
        :param do_mul:
        """
        super(FeatureMapScaling, self).__init__()
        self.fc = torch.nn.Linear(nb_dim, nb_dim)
        self.sig = torch.nn.Sigmoid()
        self.do_add = do_add
        self.do_mul = do_mul

    def forward(self, x):
        """

        :param x:
        :return:
        """
        y = torch.nn.functional.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
        y = self.sig(self.fc(y)).view(x.size(0), x.size(1), -1)

        if self.do_mul:
            x = x * y

        if self.do_add:
            x = x + y

        return x


class ResBlockWFMS(torch.nn.Module):
    """

    """
    def __init__(self, nb_filts, first=False):
        """

        :param nb_filts:
        :param first:
        """
        super(ResBlockWFMS, self).__init__()
        self.first = first

        if not self.first:
            self.bn1 = torch.nn.BatchNorm1d(num_features=nb_filts[0])

        self.lrelu = torch.nn.LeakyReLU()
        self.lrelu_keras = torch.nn.LeakyReLU(negative_slope=0.3)

        self.conv1 = torch.nn.Conv1d(in_channels=nb_filts[0],
                                     out_channels=nb_filts[1],
                                     kernel_size=3,
                                     padding=1,
                                     stride=1)

        self.bn2 = torch.nn.BatchNorm1d(num_features=nb_filts[1])

        self.conv2 = torch.nn.Conv1d(in_channels=nb_filts[1],
                                     out_channels=nb_filts[1],
                                     padding=1,
                                     kernel_size=3,
                                     stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = torch.nn.Conv1d(in_channels=nb_filts[0],
                                                   out_channels=nb_filts[1],
                                                   padding=0,
                                                   kernel_size=1,
                                                   stride=1)
        else:
            self.downsample = False

        self.mp = torch.nn.MaxPool1d(3)

        self.fms = FeatureMapScaling(nb_dim=nb_filts[1],
                                     do_add=True,
                                     do_mul=True
                                     )

    def forward(self, x):
        """

        :param x:
        :return:
        """
        identity = x

        if not self.first:
            out = self.bn1(x)
            out = self.lrelu_keras(out)
        else:
            out = x

        #out = self.conv1(x)
        out = self.conv1(out)   # modif Anthony
        out = self.bn2(out)
        out = self.lrelu_keras(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        out = self.fms(out)

        return out


class LayerNorm(torch.nn.Module):
    """

    """
    def __init__(self, features, eps=1e-6):
        """

        :param features:
        :param eps:
        """
        super(LayerNorm,self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones(features))
        self.beta = torch.nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """

        :param x:
        :return:
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class ResBlock(torch.nn.Module):
    """

    """
    def __init__(self, in_channels, out_channels, stride, is_first=False):
        """

        :param filter_size:
        :param channel_nb:
        :param is_first: boolean, True if this block ios the first of the model, if not, apply a BatchNorm layer first
        """
        super(ResBlock, self).__init__()
        self.is_first = is_first
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = self.out_channels // self.in_channels

        self.resample = None
        if not self.in_channels == self.out_channels:
            self.resample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=self.in_channels,
                                out_channels=self.out_channels,
                                kernel_size=1),
                torch.nn.BatchNorm2d(self.in_channels * self.expansion),
            )

        if not self.is_first:
            self.batch_norm1 = torch.nn.BatchNorm2d(num_features=self.in_channels)

        self.activation = torch.nn.LeakyReLU()

        self.conv1 = torch.nn.Conv2d(in_channels=self.in_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=(3,3),
                                     stride=stride,
                                     padding=1,
                                     padding_mode='zeros',
                                     dilation=1)
        self.conv2= torch.nn.Conv2d(in_channels=self.out_channels,
                                    out_channels=self.out_channels,
                                    stride=stride,
                                    kernel_size=(3,3),
                                    padding=1,
                                    padding_mode='zeros',
                                    dilation=1)

        self.batch_norm2 = torch.nn.BatchNorm2d(num_features=self.out_channels)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        identity = x

        if not self.is_first:
            out = self.activation(self.batch_norm1(x))
        else:
            out = x

        out = self.conv1(out)
        out = self.batch_norm2(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.batch_norm2(out)

        if not self.expansion == 1:
            identity = self.resample(identity)
        out += identity

        out =  self.activation(out)
        return out


class SELayer(torch.nn.Module):
    """

    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channel, channel // reduction, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(channel // reduction, channel, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        """

        :param x:
        :return:
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BasicBlock(torch.nn.Module):
    """

    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)

        self.se = SELayer(planes)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        """

        :param x:
        :return:
        """
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = torch.nn.functional.relu(out)
        return out


class Bottleneck(torch.nn.Module):
    """

    """
    def __init__(self, in_planes, planes, stride=1, expansion=4):
        super(Bottleneck, self).__init__()

        self.expansion = expansion

        self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv3 = torch.nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        """

        :param x:
        :return:
        """
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = torch.nn.functional.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = torch.nn.functional.relu(out)
        return out


class ResNet(torch.nn.Module):
    """

    """
    def __init__(self, block, num_blocks, speaker_number=10):
        super(ResNet, self).__init__()
        self.in_planes = 128
        self.speaker_number = speaker_number

        # Feature extraction
        n_fft = 2048
        win_length = None
        hop_length = 512
        n_mels = 80
        n_mfcc = 80

        # todo modify the front-end like for other architectures
        self.MFCC = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=n_mfcc, melkwargs={'n_fft': n_fft, 'n_mels': n_mels, 'hop_length': hop_length})

        self.CMVN = torch.nn.InstanceNorm1d(80)

        self.conv1 = torch.nn.Conv2d(1, 128, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(128)

        #  With block = [3, 1, 3, 1, 5, 1, 2]
        self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(block, 256, num_blocks[4], stride=1)
        self.layer6 = self._make_layer(block, 256, num_blocks[5], stride=2)
        self.layer7 = self._make_layer(block, 256, num_blocks[5], stride=1)
        self.stat_pooling = MeanStdPooling()
        self.before_embedding = torch.nn.Linear(5120, 256)

    def _make_layer(self, block, planes, num_blocks, stride):
        """

        :param block:
        :param planes:
        :param num_blocks:
        :param stride:
        :return:
        """
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        out = self.MFCC(x)
        out = self.CMVN(out)
        out = out.unsqueeze(1)
        out = torch.nn.functional.relu(self.bn1(self.conv1(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = torch.flatten(out, start_dim=1, end_dim=2)
        out = self.stat_pooling(out)
        out = self.before_embedding(out)
        return out


class PreResNet34(torch.nn.Module):
    """
    Networks that contains only the ResNet part until pooling, with NO classification layers
    """
    def __init__(self, block=BasicBlock, num_blocks=[3, 1, 3, 1, 5, 1, 2], speaker_number=10):
        super(PreResNet34, self).__init__()
        self.in_planes = 128

        self.speaker_number = speaker_number
        self.conv1 = torch.nn.Conv2d(1, 128, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.bn1 = torch.nn.BatchNorm2d(128)

        #  With block = [3, 1, 3, 1, 5, 1, 2]
        self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(block, 256, num_blocks[4], stride=1)
        self.layer6 = self._make_layer(block, 256, num_blocks[5], stride=2)
        self.layer7 = self._make_layer(block, 256, num_blocks[5], stride=1)


    def _make_layer(self, block, planes, num_blocks, stride):
        """

        :param block:
        :param planes:
        :param num_blocks:
        :param stride:
        :return:
        """
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        out = x.unsqueeze(1)
        out = torch.nn.functional.relu(self.bn1(self.conv1(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = torch.flatten(out, start_dim=1, end_dim=2)
        return out


class PreHalfResNet34(torch.nn.Module):
    """
    Networks that contains only the ResNet part until pooling, with NO classification layers
    """
    def __init__(self, block=BasicBlock, num_blocks=[3, 4, 6, 3], speaker_number=10):
        super(PreHalfResNet34, self).__init__()
        self.in_planes = 32
        self.speaker_number = speaker_number

        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3,
                               stride=(1, 1), padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(32)

        #  With block = [3, 4, 6, 3]
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=(1, 1))
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=(2, 2))


    def _make_layer(self, block, planes, num_blocks, stride):
        """

        :param block:
        :param planes:
        :param num_blocks:
        :param stride:
        :return:
        """
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        out = x.unsqueeze(1)
        out = out.contiguous(memory_format=torch.channels_last)
        out = torch.nn.functional.relu(self.bn1(self.conv1(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.contiguous(memory_format=torch.contiguous_format)
        out = torch.flatten(out, start_dim=1, end_dim=2)
        return out


class PreFastResNet34(torch.nn.Module):
    """
    Networks that contains only the ResNet part until pooling, with NO classification layers
    """
    def __init__(self, block=BasicBlock, num_blocks=[3, 4, 6, 3], speaker_number=10):
        super(PreFastResNet34, self).__init__()
        self.in_planes = 16
        self.speaker_number = speaker_number

        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=7,
                               stride=(2, 1), padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(16)

        #  With block = [3, 4, 6, 3]
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=(1, 1))


    def _make_layer(self, block, planes, num_blocks, stride):
        """

        :param block:
        :param planes:
        :param num_blocks:
        :param stride:
        :return:
        """
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        out = x.unsqueeze(1)
        out = out.contiguous(memory_format=torch.channels_last)
        out = torch.nn.functional.relu(self.bn1(self.conv1(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.contiguous(memory_format=torch.contiguous_format)
        out = torch.flatten(out, start_dim=1, end_dim=2)
        return out


def ResNet34():
    return ResNet(BasicBlock, [3, 1, 3, 1, 5, 1, 2])

# TODO create a flexible class that allows to generate different RESNET sequential classes and manage the sizes