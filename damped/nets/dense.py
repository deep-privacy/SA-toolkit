import torch.nn as nn


class DenseReLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DenseReLU, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.nl = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.nl(x)
        return x


class DenseEmbedding(nn.Module):
    """
    DenseEmbedding as defined by http://www.danielpovey.com/files/2017_interspeech_embeddings.pdf

    The two hidden layers with dimension ``mid_dim`` and ``out_dim``
    may be used to compute embeddings.
    """

    def __init__(self, in_dim, mid_dim, out_dim):
        super(DenseEmbedding, self).__init__()
        self.hidden1 = DenseReLU(in_dim, mid_dim)
        self.hidden2 = DenseReLU(mid_dim, out_dim)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        return x
