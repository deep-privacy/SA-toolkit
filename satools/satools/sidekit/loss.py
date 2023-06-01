"""
This file is part of SIDEKIT.
Copyright 2014-2023 Anthony Larcher
"""


import math
import numpy
import torch
from typing import Optional

from collections import OrderedDict
from torch.nn import Parameter


class CCELoss(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, embbedings, target:Optional[torch.Tensor]=None):
        x = self.module(embbedings)
        if target is None:
            return torch.tensor(float('nan')), x
        loss = self.criterion(x, target)
        return loss, x


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
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.weight)
        self.change_params(s=s, m=m)
        self.easy_margin = easy_margin
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

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

    def forward(self, input, target:Optional[torch.Tensor]=None):
        """

        :param input:
        :param target:
        :return:
        """
        # cos(theta)
        cosine = torch.nn.functional.linear(torch.nn.functional.normalize(input),
                                        torch.nn.functional.normalize(self.weight))
        if target is None:
            return torch.tensor(float('nan')), cosine * self.s
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

        return self.criterion(output, target), cosine * self.s


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

    def forward(self, x, target:Optional[torch.Tensor]=None):
        """

        :param x:
        :param target:
        :return:
        """
        assert x.size()[1] >= 2

        cce_prediction = self.cce_backend(x)

        if target is None:
            return torch.tensor(float('nan')), cce_prediction

        x = x.reshape(-1, 2, x.size()[-1]).squeeze(1)

        out_anchor = torch.mean(x[:, 1:, :], 1)
        out_positive = x[:,0,:]

        cos_sim_matrix = torch.nn.functional.cosine_similarity(out_positive.unsqueeze(-1),
                                                               out_anchor.unsqueeze(-1).transpose(0, 2))
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        loss = self.criterion(cos_sim_matrix,
                              torch.arange(0,
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

        self.cce_backend = torch.nn.Sequential(OrderedDict([
                    ("linear8", torch.nn.Linear(emb_dim, spk_count))
                ]))

        self.criterion  = torch.nn.CrossEntropyLoss()
        self.magnet_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, x, target:Optional[torch.Tensor]=None):
        """

        :param x:
        :param target:
        :return:
        """
        assert x.size()[1] >= 2

        cce_prediction = self.cce_backend(x)

        if target is None:
            return torch.tensor(float('nan')), cce_prediction

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


class CircleMargin(torch.nn.Module):
    """Circle loss implementation with speaker prototypes
    https://arxiv.org/pdf/2002.10857.pdf

    Args:
        emb_dim (int): speaker embedding dimension
        speaker_count (int): number of speaker protoypes
        s (int): scale
        m (float): margin

    """
    def __init__(self, emb_dim, speaker_count, s=64, m=0.35, k=1) -> None:
        super(CircleMargin, self).__init__()
        self.margin = m
        self.gamma = s
        self.k = k
        self.weight = Parameter(torch.FloatTensor(speaker_count * self.k, emb_dim))
        torch.nn.init.xavier_uniform_(self.weight)
        self.soft_plus = torch.nn.Softplus()

    def forward(self, x, target:Optional[torch.Tensor]=None):
        """

        :param x:
        :param target:
        :return:
        """
        cosine = torch.nn.functional.linear(torch.nn.functional.normalize(x),
                                            torch.nn.functional.normalize(self.weight))

        cosine = cosine.reshape(cosine.shape[0], -1, self.k).max(-1)[0]

        if target is None:
            return torch.tensor(float('nan')), cosine * self.gamma

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, target.view(-1, 1), 1)

        pos = torch.masked_select(cosine, one_hot==1).unsqueeze(1)
        neg = torch.masked_select(cosine, one_hot==0).reshape(cosine.shape[0], cosine.shape[1]-1)

        alpha_p = torch.clamp_min(-pos.detach() + 1 + self.margin, min=0.)
        alpha_n = torch.clamp_min(neg.detach() + self.margin, min=0.)
        margin_p = 1 - self.margin
        margin_n = self.margin

        loss = self.soft_plus(torch.logsumexp(self.gamma * (-alpha_p * (pos - margin_p)), dim=-1)\
            + torch.logsumexp(self.gamma * (alpha_n * (neg - margin_n)), dim=-1)).mean()

        return loss, cosine * self.gamma


class CircleProto(torch.nn.Module):
    """Circle loss implementation with speaker prototypes and parwise similarities
    https://arxiv.org/pdf/2002.10857.pdf

    Args:
        emb_dim (int): speaker embedding dimension
        speaker_count (int): number of speaker protoypes
        s (int): scale
        m (float): margin

    """
    def __init__(self, in_features, out_features, s=64, m=0.40):
        super(CircleProto, self).__init__()
        
        self.margin = m
        self.gamma = s
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.weight)
        self.soft_plus = torch.nn.Softplus()


    def forward(self, x, target:Optional[torch.Tensor]=None):
        """

        :param x:
        :param target:
        :return:
        """
        cosine = torch.nn.functional.linear(torch.nn.functional.normalize(x),
                                            torch.nn.functional.normalize(self.weight))

        if target is None:
            return torch.tensor(float('nan')), cosine * self.gamma
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, target.view(-1, 1), 1)

        pos = torch.masked_select(cosine, one_hot==1).unsqueeze(1)
        neg = torch.masked_select(cosine, one_hot==0).reshape(cosine.shape[0], cosine.shape[1]-1)

        alpha_p = torch.clamp_min(-pos.detach() + 1 + self.margin, min=0.)
        alpha_n = torch.clamp_min(neg.detach() + self.margin, min=0.)
        margin_p = 1 - self.margin
        margin_n = self.margin

        loss = self.soft_plus(torch.logsumexp(self.gamma * (-alpha_p * (pos - margin_p)), dim=-1)\
            + torch.logsumexp(self.gamma * (alpha_n * (neg - margin_n)), dim=-1)).mean()

        assert x.size()[1] >= 2
        x = x.reshape(-1, 2, x.size()[-1]).squeeze(1)
        
        out_anchor = torch.mean(x[:, 1:, :], 1)
        out_positive = x[:,0,:]

        sim_matx = torch.nn.functional.cosine_similarity(out_positive.unsqueeze(-1),
                                                         out_anchor.unsqueeze(-1).transpose(0, 2))

        one_hot = torch.eye(sim_matx.shape[0], device=x.device)

        pos = torch.masked_select(sim_matx, one_hot==1).unsqueeze(1)
        neg = torch.masked_select(sim_matx, one_hot==0).reshape(sim_matx.shape[0], sim_matx.shape[1]-1)

        alpha_p = torch.clamp_min(-pos.detach() + 1 + self.margin, min=0.)
        alpha_n = torch.clamp_min(neg.detach() + self.margin, min=0.)
        margin_p = 1 - self.margin
        margin_n = self.margin

        loss += self.soft_plus(torch.logsumexp(self.gamma * (-alpha_p * (pos - margin_p)), dim=-1)\
            + torch.logsumexp(self.gamma * (alpha_n * (neg - margin_n)), dim=-1)).mean()

        return loss, cosine * self.gamma
