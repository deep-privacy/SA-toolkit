import io

import kaldiio
import numpy as np
import torch


class CMVN(object):
    def __init__(
        self,
        stats,
        norm_means=True,
        norm_vars=False,
        filetype="mat",
        utt2spk=None,
        spk2utt=None,
        reverse=False,
        std_floor=1.0e-20,
    ):
        self.stats_file = stats
        self.norm_means = norm_means
        self.norm_vars = norm_vars
        self.reverse = reverse

        if isinstance(stats, dict):
            stats_dict = dict(stats)
        else:
            # Use for global CMVN
            if filetype == "mat":
                stats_dict = {None: kaldiio.load_mat(stats)}
            # Use for global CMVN
            elif filetype == "npy":
                stats_dict = {None: np.load(stats)}
            # Use for speaker CMVN
            elif filetype == "ark":
                self.accept_uttid = True
                stats_dict = dict(kaldiio.load_ark(stats))
            elif filetype == "scp":
                self.accept_uttid = True
                stats_dict = dict(kaldiio.load_scp(stats))
            else:
                raise ValueError("Not supporting filetype={}".format(filetype))

        if utt2spk is not None:
            self.utt2spk = {}
            with io.open(utt2spk, "r", encoding="utf-8") as f:
                for line in f:
                    utt, spk = line.rstrip().split(None, 1)
                    self.utt2spk[utt] = spk
        elif spk2utt is not None:
            self.utt2spk = {}
            with io.open(spk2utt, "r", encoding="utf-8") as f:
                for line in f:
                    spk, utts = line.rstrip().split(None, 1)
                    for utt in utts.split():
                        self.utt2spk[utt] = spk
        else:
            self.utt2spk = None

        # Kaldi makes a matrix for CMVN which has a shape of (2, feat_dim + 1),
        # and the first vector contains the sum of feats and the second is
        # the sum of squares. The last value of the first, i.e. stats[0,-1],
        # is the number of samples for this statistics.
        self.bias = {}
        self.scale = {}
        acc_bias = None
        acc_scale = None
        for spk, stats in stats_dict.items():
            assert len(stats) == 2, stats.shape

            count = stats[0, -1]

            # If the feature has two or more dimensions
            if not (np.isscalar(count) or isinstance(count, (int, float))):
                # The first is only used
                count = count.flatten()[0]

            mean = stats[0, :-1] / count
            # V(x) = E(x^2) - (E(x))^2
            var = stats[1, :-1] / count - mean * mean
            std = np.maximum(np.sqrt(var), std_floor)
            self.bias[spk] = torch.tensor(-mean, dtype=torch.float32)
            self.scale[spk] = torch.tensor(1 / std, dtype=torch.float32)


            if acc_bias == None:
                acc_scale = torch.zeros_like(self.scale[spk])
                acc_bias = torch.zeros_like(self.bias[spk])

            acc_bias.add_(self.bias[spk])
            acc_scale.add_(self.scale[spk])

        self.bias["generic-spk"] = acc_bias/len(stats_dict)
        self.scale["generic-spk"] = acc_scale/len(stats_dict)


    def __repr__(self):
        return (
            "{name}(stats_file={stats_file}, "
            "norm_means={norm_means}, norm_vars={norm_vars}, "
            "reverse={reverse})".format(
                name=self.__class__.__name__,
                stats_file=self.stats_file,
                norm_means=self.norm_means,
                norm_vars=self.norm_vars,
                reverse=self.reverse,
            )
        )

    def __call__(self, x, uttid=None):
        if self.utt2spk is not None and uttid != "generic-spk":
            spk = self.utt2spk[uttid]
        else:
            spk = uttid

        if not self.reverse:
            if self.norm_means:
                x = torch.add(x, self.bias[spk].to(x.device))
            if self.norm_vars:
                x = torch.multiply(x, self.scale[spk].to(x.device))

        else:
            if self.norm_vars:
                x = torch.divide(x, self.scale[spk].to(x.device))
            if self.norm_means:
                x = torch.subtract(x, self.bias[spk].to(x.device))

        return x


EPSILON = 1e-6

class UttCMVN(torch.nn.Module):
    def __init__(self, var_norm=False):
        super(UttCMVN, self).__init__()
        self.var_norm = var_norm

    def forward(self, x):
        mean = x.mean(dim=1, keepdims=True)
        if self.var_norm:
            std = torch.sqrt(x.var(dim=1, keepdims=True) + EPSILON)
        x = x - mean
        if self.var_norm:
            x /= std
        return x
