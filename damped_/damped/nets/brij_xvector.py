import torch
from torch import nn
import torch.nn.functional as F

"""
The author would like to thank Brij Mohan Lal Srivastava (https://brijmohan.github.io/)
for sharing his xvector implementation.
"""


class BrijSpeakerXvector(nn.Module):
    """ Speaker adversarial module
    """

    def __init__(self, odim, eprojs, hidden_size, rnn_layers, dropout_rate=0.2):
        super(BrijSpeakerXvector, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.dropout_rate = dropout_rate
        self.advnet = nn.LSTM(
            eprojs,
            hidden_size,
            self.rnn_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=False,
        )
        self.segment6 = nn.Linear(hidden_size, hidden_size)
        self.segment7 = nn.Linear(hidden_size, hidden_size)
        self.segment8 = nn.Linear(2 * hidden_size, 2 * hidden_size)

        self.bn2 = nn.BatchNorm1d(2 * hidden_size)
        self.output = nn.Linear(2 * hidden_size, odim)

    def zero_state(self, hs_pad):
        return hs_pad.new_zeros(self.rnn_layers, hs_pad.size(0), self.hidden_size)

    def forward(self, hs_pad):
        """Speaker branch forward

        Args:
            hs_pad (torch.Tensor): batch of padded hidden state sequences (B, Tmax, D)
        """

        h_0 = self.zero_state(hs_pad)
        c_0 = self.zero_state(hs_pad)

        self.advnet.flatten_parameters()  # Memory: compact weights
        out_x, (h_0, c_0) = self.advnet(hs_pad, (h_0, c_0))

        out_x = F.dropout(F.relu(self.segment6(out_x)), p=self.dropout_rate)
        out_x = F.relu(self.segment7(out_x))

        # STATS POOLING
        # out shape: B x T x D
        xv_mean = torch.mean(out_x, 1)
        xv_std = torch.std(out_x, 1)
        xv = torch.cat((xv_mean, xv_std), 1)
        # Take only the last output
        # xv = out_x[:, -1, :]

        y_hat = (
            self.segment8(xv) if hs_pad.size(0) == 1 else self.bn2(self.segment8(xv))
        )
        y_hat = self.output(y_hat)
        return y_hat
