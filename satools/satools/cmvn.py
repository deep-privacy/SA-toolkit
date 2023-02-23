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


class UttCMVN(torch.nn.Module):
    def __init__(self, var_norm=False):
        super(UttCMVN, self).__init__()
        self.var_norm = var_norm

    def forward(self, x):
        mean = x.mean(dim=1).unsqueeze(1)
        if self.var_norm:
            std = torch.sqrt(x.var(dim=1) + 1e-6).unsqueeze(1)
            x = x - mean
            x /= std
        else:
            x = x - mean
        return x


class AdaptivePCMN(torch.nn.Module):
    """ Using adaptive parametric Cepstral Mean Normalization to replace traditional CMN.
        It is implemented according to [Ozlem Kalinli, etc. "Parametric Cepstral Mean Normalization 
        for Robust Automatic Speech Recognition", icassp, 2019.]
    """
    def __init__(self, input_dim, left_context=-10, right_context=10, pad=True):
        super(AdaptivePCMN, self).__init__()

        assert left_context < 0 and right_context > 0

        self.left_context = left_context
        self.right_context = right_context
        self.tot_context = self.right_context - self.left_context + 1

        kernel_size = (self.tot_context,)

        self.input_dim = input_dim
        # Just pad head and end rather than zeros using replicate pad mode 
        # or set pad false with enough context egs.
        self.pad = pad
        self.pad_mode = "replicate"

        self.groups = input_dim
        output_dim = input_dim

        # The output_dim is equal to input_dim and keep every dims independent by using groups conv.
        self.beta_w = torch.nn.Parameter(torch.randn(output_dim, input_dim//self.groups, *kernel_size))
        self.alpha_w = torch.nn.Parameter(torch.randn(output_dim, input_dim//self.groups, *kernel_size))
        self.mu_n_0_w = torch.nn.Parameter(torch.randn(output_dim, input_dim//self.groups, *kernel_size))
        self.bias = torch.nn.Parameter(torch.randn(output_dim))

        # init weight and bias. It is important
        self.init_weight()

    def init_weight(self):
        torch.nn.init.normal_(self.beta_w, 0., 0.01)
        torch.nn.init.normal_(self.alpha_w, 0., 0.01)
        torch.nn.init.normal_(self.mu_n_0_w, 0., 0.01)
        torch.nn.init.constant_(self.bias, 0.)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [B, TIME, FRAME]
        """
        inputs = inputs.permute(0, 2, 1)
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim
        assert inputs.shape[2] >= self.tot_context

        if self.pad:
            pad_input = torch.nn.functional.pad(inputs, (-self.left_context, self.right_context), mode=self.pad_mode)
        else:
            pad_input = inputs
            inputs = inputs[:,:,-self.left_context:-self.right_context]

        # outputs beta + 1 instead of beta to avoid potentially zeroing out the inputs cepstral features.
        self.beta = torch.nn.functional.conv1d(pad_input, self.beta_w, bias=self.bias, groups=self.groups) + 1
        self.alpha = torch.nn.functional.conv1d(pad_input, self.alpha_w, bias=self.bias, groups=self.groups)
        self.mu_n_0 = torch.nn.functional.conv1d(pad_input, self.mu_n_0_w, bias=self.bias, groups=self.groups)

        outputs = self.beta * inputs - self.alpha * self.mu_n_0

        outputs = inputs.permute(0, 2, 1)

        return outputs
