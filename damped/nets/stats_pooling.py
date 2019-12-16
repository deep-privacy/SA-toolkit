import torch
import torch.nn as nn


class StatsPooling(nn.Module):
    """
    StatsPooling as defined by http://www.danielpovey.com/files/2017_interspeech_embeddings.pdf

    The statistics pooling layer aggregates all T frame-level outputs from his
    variational input tensor and computes its mean and standard deviation.
    """

    def __init__(self):
        super(StatsPooling, self).__init__()

    def forward(self, varient_length_tensor):
        mean = varient_length_tensor.mean(dim=1)
        std = varient_length_tensor.std(dim=1)
        return torch.cat((mean, std), dim=1)
