# coding: utf-8 -*-
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
Copyright 2014-2021 Anthony Larcher, Yevhenii Prokopalo
"""


import os
import torch


os.environ['MKL_THREADING_LAYER'] = 'GNU'

__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2015-2021 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reS'


class MeanStdPooling(torch.nn.Module):
    """
    Mean and Standard deviation pooling
    """
    def __init__(self):
        """

        """
        super(MeanStdPooling, self).__init__()
        pass

    def forward(self, x):
        """

        :param x:
        :return:
        """
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)
        return torch.cat([mean, std], dim=1)


class AttentivePooling(torch.nn.Module):
    """
    Mean and Standard deviation attentive pooling
    """
    def __init__(self, num_channels, n_mels, reduction=2, global_context=False):
        """

        """
        # TODO Make global_context configurable (True/False)
        # TODO Make convolution parameters configurable
        super(AttentivePooling, self).__init__()
        in_factor = 3 if global_context else 1
        self.attention = torch.nn.Sequential(
            torch.nn.Conv1d(num_channels * (n_mels//8) * in_factor, num_channels//reduction, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_channels//reduction),
            torch.nn.Tanh(),
            torch.nn.Conv1d(num_channels//reduction, num_channels * (n_mels//8), kernel_size=1),
            torch.nn.Softmax(dim=2),
        )
        self.global_context = global_context
        self.gc = MeanStdPooling()

    def new_parameter(self, *size):
        out = torch.nn.Parameter(torch.FloatTensor(*size))
        torch.nn.init.xavier_normal_(out)
        return out

    def forward(self, x):
        """

        :param x:
        :return:
        """
        if self.global_context:
            w = self.attention(torch.cat([x, self.gc(x).unsqueeze(2).repeat(1, 1, x.shape[-1])], dim=1))
        else:
            w = self.attention(x)

        mu = torch.sum(x * w, dim=2)
        rh = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-5) )
        x = torch.cat((mu, rh),1)
        x = x.view(x.size()[0], -1)
        return x


class GruPooling(torch.nn.Module):
    """
    Pooling done by using a recurrent network
    """
    def __init__(self, input_size, gru_node, nb_gru_layer):
        """

        :param input_size:
        :param gru_node:
        :param nb_gru_layer:
        """
        super(GruPooling, self).__init__()
        self.lrelu_keras = torch.nn.LeakyReLU(negative_slope = 0.3)
        self.bn_before_gru = torch.nn.BatchNorm1d(num_features = input_size)
        self.gru = torch.nn.GRU(input_size = input_size,
                                hidden_size = gru_node,
                                num_layers = nb_gru_layer,
                                batch_first = True)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.bn_before_gru(x)
        x = self.lrelu_keras(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]

        return x
