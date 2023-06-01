"""
This file is part of SIDEKIT.
Copyright 2014-2023 Anthony Larcher
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import nn as sann

def to_channels_last(net):
    if True:
        return net.to(memory_format=torch.channels_last)
    return net

def make_layer(model, block, planes, num_blocks, stride):
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
        layers.append(block(model.in_planes, planes, stride))
        model.in_planes = planes * block.expansion
    return to_channels_last(torch.nn.Sequential(*layers))


class PreResNet34(torch.nn.Module):
    """
    Networks that contains only the ResNet part until pooling, with NO classification layers
    """
    def __init__(self, block=sann.ResNetBasicBlock, num_blocks=(3, 1, 3, 1, 5, 1, 2), speaker_number=10):
        super(PreResNet34, self).__init__()
        self.in_planes = 128

        self.speaker_number = speaker_number
        self.conv1 = to_channels_last(torch.nn.Conv2d(1,
                                     128,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     bias=False))

        self.bn1 = to_channels_last(torch.nn.BatchNorm2d(128))

        #  With block = [3, 1, 3, 1, 5, 1, 2]
        self.layer1 = make_layer(self, block, 128, num_blocks[0], stride=1)
        self.layer2 = make_layer(self, block, 128, num_blocks[1], stride=2)
        self.layer3 = make_layer(self, block, 128, num_blocks[2], stride=1)
        self.layer4 = make_layer(self, block, 256, num_blocks[3], stride=2)
        self.layer5 = make_layer(self, block, 256, num_blocks[4], stride=1)
        self.layer6 = make_layer(self, block, 256, num_blocks[5], stride=2)
        self.layer7 = make_layer(self, block, 256, num_blocks[5], stride=1)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 2)
        x = to_channels_last(x)
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return x


class PreHalfResNet34(torch.nn.Module):
    """
    Networks that contains only the HalfResNet part until pooling, with NO classification layers
    """
    def __init__(self, block=sann.ResNetBasicBlock, num_blocks=(3, 4, 6, 3), speaker_number=10):
        super(PreHalfResNet34, self).__init__()
        self.in_planes = 32
        self.speaker_number = speaker_number

        self.conv1 = to_channels_last(torch.nn.Conv2d(1,
                                     32,
                                     kernel_size=3,
                                     stride=(1, 1),
                                     padding=1,
                                     bias=False).to(memory_format=torch.channels_last))
        self.bn1 = to_channels_last(torch.nn.BatchNorm2d(32))

        #  With block = [3, 4, 6, 3]
        self.layer1 = make_layer(self, block, 32, num_blocks[0], stride=(1, 1))
        self.layer2 = make_layer(self, block, 64, num_blocks[1], stride=(2, 2))
        self.layer3 = make_layer(self, block, 128, num_blocks[2], stride=(2, 2))
        self.layer4 = make_layer(self, block, 256, num_blocks[3], stride=(2, 2))

    def forward(self, x):
        """

        :param x:
        :return:
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
            x = x.permute(0, 1, 3, 2)
        x = to_channels_last(x)
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class PreFastResNet34(torch.nn.Module):
    """
    Networks that contains only the FastResNet part until pooling, with NO classification layers
    """
    def __init__(self, block=sann.ResNetBasicBlock, num_blocks=(3, 4, 6, 3), speaker_number=10):
        super(PreFastResNet34, self).__init__()
        self.in_planes = 16
        self.speaker_number = speaker_number

        self.conv1 = to_channels_last(torch.nn.Conv2d(1,
                                     16,
                                     kernel_size=7,
                                     stride=(1, 2),
                                     padding=3,
                                     bias=False))
        self.bn1 = to_channels_last(torch.nn.BatchNorm2d(16))

        #  With block = [3, 4, 6, 3]
        self.layer1 = make_layer(self, block, 16, num_blocks[0], stride=1)
        self.layer2 = make_layer(self, block, 32, num_blocks[1], stride=(2, 2))
        self.layer3 = make_layer(self, block, 64, num_blocks[2], stride=(2, 2))
        self.layer4 = make_layer(self, block, 128, num_blocks[3], stride=(1, 1))

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 2)
        x = to_channels_last(x)
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x



class PreEcapaTDNN(nn.Module):
    """
    Implementation of 
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification".
    (https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py)

    Note that we DON'T concatenate the last frame-wise layer with non-weighted mean and standard deviation,
    because it brings little improvment but significantly increases model parameters.
    As a result, this implementation basically equals the A.2 of Table 2 in the paper.
    """
    def __init__(self, in_feature=80, channels=512):
        super().__init__()
        self.layer1 = sann.Conv1dReluBn(in_feature, channels, kernel_size=5, padding=2)
        self.layer2 = sann.SE_Res2Block(channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8)
        self.layer3 = sann.SE_Res2Block(channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8)
        self.layer4 = sann.SE_Res2Block(channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8)
        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, cat_channels, kernel_size=1)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1) + out1
        out3 = self.layer3(out1 + out2) + out1 + out2
        out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3
        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        return out
