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


import math
import numpy
import torch

from collections import OrderedDict
from torch.nn import Parameter


#from .classification import Classification

__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2015-2020 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reS'


class ArcMarginModel(torch.nn.Module):
    """

    """
    def __init__(self, args):
        super(ArcMarginModel, self).__init__()

        self.weight = Parameter(torch.FloatTensor(num_classes, args.emb_size))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = args.easy_margin
        self.m = args.margin_m
        self.s = args.margin_s

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input, label):
        """

        :param input:
        :param label:
        :return:
        """
        x = F.normalize(input)
        W = F.normalize(self.weight)
        cosine = F.linear(x, W)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


def l2_norm(input, axis=1):
    """

    :param input:
    :param axis:
    :return:
    """
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class ArcFace(torch.nn.Module):
    """

    """
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size, classnum, s=64., m=0.5):
        super(ArcFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m  # the margin value, default is 0.5
        self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)

    def forward(self, embbedings, target):
        """

        :param embbedings:
        :param target:
        :return:
        """
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel, axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings, kernel_norm)
        #         output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        # when theta not in [0,pi], use cosface instead
        keep_val = (cos_theta - self.mm)
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        # a little bit hacky way to prevent in_place operation on cos_theta
        output = cos_theta * 1.0
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, target] = cos_theta_m[idx_, target]
        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        return output


##################################  Cosface head #############################################################

class Am_softmax(torch.nn.Module):
    """

    """
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=512, classnum=51332):
        super(Am_softmax, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = 0.35  # additive margin recommended by the paper
        self.s = 30.  # see normface https://arxiv.org/abs/1704.06369

    def forward(self, embbedings, label):
        """

        :param embbedings:
        :param label:
        :return:
        """
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        phi = cos_theta - self.m
        label = label.view(-1, 1)  # size=(B,1)
        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, label.data.view(-1, 1), 1)
        index = index.byte()
        output = cos_theta * 1.0
        output[index] = phi[index]  # only change the correct predicted output
        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        return output


class ArcLinear(torch.nn.Module):
    """Additive Angular Margin linear module (ArcFace)

    Parameters
    ----------
    nfeat : int
        Embedding dimension
    nclass : int
        Number of classes
    margin : float
        Angular margin to penalize distances between embeddings and centers
    s : float
        Scaling factor for the logits
    """

    def __init__(self, nfeat, nclass, margin, s):
        super(ArcLinear, self).__init__()
        eps = 1e-4
        self.min_cos = eps - 1
        self.max_cos = 1 - eps
        self.nclass = nclass
        self.margin = margin
        self.s = s
        self.W = torch.nn.Parameter(torch.Tensor(nclass, nfeat))
        torch.nn.init.xavier_uniform_(self.W)

    def forward(self, x, target=None):
        """Apply the angular margin transformation

        Parameters
        ----------
        x : `torch.Tensor`
            an embedding batch
        target : `torch.Tensor`
            a non one-hot label batch

        Returns
        -------
        fX : `torch.Tensor`
            logits after the angular margin transformation
        """
        # the feature vectors has been normalized before calling this layer
        #xnorm = torch.nn.functional.normalize(x)
        xnorm = x
        # normalize W
        Wnorm = torch.nn.functional.normalize(self.W)
        target = target.long().view(-1, 1)
        # calculate cosθj (the logits)
        cos_theta_j = torch.matmul(xnorm, torch.transpose(Wnorm, 0, 1))
        # get the cosθ corresponding to the classes
        cos_theta_yi = cos_theta_j.gather(1, target)
        # for numerical stability
        cos_theta_yi = cos_theta_yi.clamp(min=self.min_cos, max=self.max_cos)
        # get the angle separating xi and Wyi
        theta_yi = torch.acos(cos_theta_yi)
        # apply the margin to the angle
        cos_theta_yi_margin = torch.cos(theta_yi + self.margin)
        # one hot encode  y
        one_hot = torch.zeros_like(cos_theta_j)
        one_hot.scatter_(1, target, 1.0)
        # project margin differences into cosθj
        return self.s * (cos_theta_j + one_hot * (cos_theta_yi_margin - cos_theta_yi))


class ArcMarginProduct(torch.nn.Module):
    """
    Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def change_params(self, s=None, m=None):
        """

        :param s:
        :param m:
        """
        if s is None:
            s = self.s
        if m is None:
            m = self.m
        self.s = s
        self.m = m
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input, target=None):
        """

        :param input:
        :param target:
        :return:
        """
        # cos(theta)
        cosine = torch.nn.functional.linear(torch.nn.functional.normalize(input),
                                        torch.nn.functional.normalize(self.weight))
        if target == None:
            return cosine * self.s
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, target.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        return output, cosine * self.s


class SoftmaxAngularProto(torch.nn.Module):
    """

    from https://github.com/clovaai/voxceleb_trainer/blob/3bfd557fab5a3e6cd59d717f5029b3a20d22a281/loss/angleproto.py
    """
    def __init__(self, spk_count, emb_dim=256, init_w=10.0, init_b=-5.0, **kwargs):
        super(SoftmaxAngularProto, self).__init__()

        self.test_normalize = True

        self.w = torch.nn.Parameter(torch.tensor(init_w))
        self.b = torch.nn.Parameter(torch.tensor(init_b))
        self.criterion  = torch.nn.CrossEntropyLoss()

        self.cce_backend = torch.nn.Sequential(OrderedDict([
                    ("linear8", torch.nn.Linear(emb_dim, spk_count))
                ]))

    def forward(self, x, target=None):
        """

        :param x:
        :param target:
        :return:
        """
        assert x.size()[1] >= 2

        cce_prediction = self.cce_backend(x)

        if target is None:
            return cce_prediction

        x = x.reshape(-1, 2, x.size()[-1]).squeeze(1)

        out_anchor = torch.mean(x[:, 1:, :], 1)
        out_positive = x[:,0,:]

        cos_sim_matrix = torch.nn.functional.cosine_similarity(out_positive.unsqueeze(-1),
                                                               out_anchor.unsqueeze(-1).transpose(0, 2))
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        loss = self.criterion(cos_sim_matrix, torch.arange(0,
                                                           cos_sim_matrix.shape[0],
                                                           device=x.device)) + self.criterion(cce_prediction, target)
        return loss, cce_prediction


class AngularProximityMagnet(torch.nn.Module):
    """
    from https://github.com/clovaai/voxceleb_trainer/blob/3bfd557fab5a3e6cd59d717f5029b3a20d22a281/loss/angleproto.py
    """
    def __init__(self, spk_count, emb_dim=256, batch_size=512, init_w=10.0, init_b=-5.0, **kwargs):
        super(AngularProximityMagnet, self).__init__()

        self.test_normalize = True

        self.w = torch.nn.Parameter(torch.tensor(init_w))
        self.b1 = torch.nn.Parameter(torch.tensor(init_b))
        self.b2 = torch.nn.Parameter(torch.tensor(+5.54))

        #last_linear = torch.nn.Linear(512, 1)
        #last_linear.bias.data += 1

        #self.magnitude = torch.nn.Sequential(OrderedDict([
        #            ("linear9", torch.nn.Linear(emb_dim, 512)),
        #            ("relu9", torch.nn.ReLU()),
        #            ("linear10", torch.nn.Linear(512, 512)),
        #            ("relu10", torch.nn.ReLU()),
        #            ("linear11", last_linear),
        #            ("relu11", torch.nn.ReLU())
        #        ]))

        self.cce_backend = torch.nn.Sequential(OrderedDict([
                    ("linear8", torch.nn.Linear(emb_dim, spk_count))
                ]))

        self.criterion  = torch.nn.CrossEntropyLoss()
        self.magnet_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, x, target=None):
        """

        :param x:
        :param target:
        :return:
        """
        assert x.size()[1] >= 2

        cce_prediction = self.cce_backend(x)

        if target is None:
            return x, cce_prediction

        x = x.reshape(-1, 2, x.size()[-1]).squeeze(1)
        out_anchor = torch.mean(x[:, 1:, :], 1)
        out_positive = x[:, 0, :]

        ap_sim_matrix  = torch.nn.functional.cosine_similarity(out_positive.unsqueeze(-1),out_anchor.unsqueeze(-1).transpose(0,2))
        torch.clamp(self.w, 1e-6)
        ap_sim_matrix = ap_sim_matrix * self.w + self.b1

        labels = torch.arange(0, int(out_positive.shape[0]), device=torch.device("cuda:0")).unsqueeze(1)
        cos_sim_matrix  = torch.mm(out_positive, out_anchor.T)
        cos_sim_matrix = cos_sim_matrix + self.b2
        cos_sim_matrix = cos_sim_matrix + numpy.log(1/out_positive.shape[0] / (1 - 1/out_positive.shape[0]))
        mask = (torch.tile(labels, (1, labels.shape[0])) == labels.T).float()
        batch_loss = self.criterion(ap_sim_matrix, torch.arange(0, int(out_positive.shape[0]), device=torch.device("cuda:0"))) \
            + self.magnet_criterion(cos_sim_matrix.flatten().unsqueeze(1), mask.flatten().unsqueeze(1))
        return batch_loss, cce_prediction
