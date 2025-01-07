import io

import kaldiio
import numpy as np
import torch
import pickle


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
    def __init__(self, var_norm=False, keep_zeros=False):
        super().__init__()
        self.var_norm = var_norm
        self.keep_zeros = keep_zeros

    def forward(self, x):
        dim = x.dim()
        if dim == 1:
            x = x.unsqueeze(0)

        if self.keep_zeros:
            uv = x == 0
            vv = x != 0

            mean = x[vv].mean()
            if self.var_norm:
                std = torch.sqrt(x[vv].var() + 1e-6)
                x[vv] = x[vv] - mean
                x[vv] /= std
            else:
                x[vv] = x[uv] - mean

            x[uv] = 0

        else:

            mean = x.mean(dim=1).unsqueeze(1)
            if self.var_norm:
                std = torch.sqrt(x.var(dim=1) + 1e-6).unsqueeze(1)
                x = x - mean
                x /= std
            else:
                x = x - mean


        if dim == 1:
            x = x.squeeze(0)

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


import torch

class SpeakerCMVN(torch.nn.Module):
    def __init__(self):
        super(SpeakerCMVN, self).__init__()
        self.speaker_stats = {}  # Store statistics per speaker
        self.stats_computed = {}  # Track whether stats are computed per speaker
        self.computed = False
        self.pass_though_if_not_computed = False
        self.keep_zeros = True
        # Register a buffer to store the byte-serialized state of the model.
        self.register_buffer('model_state_buffer', torch.zeros((900000), dtype=torch.uint8))

    def serialize_state(self):
        """Serialize only custom attributes to a byte buffer."""
        # Serialize the relevant model components (e.g., stats and flags)
        state_dict = {
            'speaker_stats': self.speaker_stats,
            'stats_computed': self.stats_computed,
            'computed': self.computed,
            'keep_zeros': self.keep_zeros
        }
        # Serialize the state to a byte array
        serialized_state = pickle.dumps(state_dict)
        # Store the serialized state as a tensor buffer
        serialized_tensor = torch.tensor(bytearray(serialized_state), dtype=torch.uint8)
        # Pad the tensor to fit into the pre-allocated buffer if needed
        current_size = serialized_tensor.size(0)
        buffer_size = self.model_state_buffer.size(0)

        if current_size < buffer_size:
            # If the serialized tensor is smaller than the buffer size, copy it to the buffer
            self.model_state_buffer[:current_size] = serialized_tensor
            # Fill the remaining part of the buffer with zeros (no padding necessary)
            self.model_state_buffer[current_size:] = 0
        else:
            raise RuntimeError(f"Not enough place {buffer_size} < {current_size} to store bytearray of SpeakerCMVN")

    def deserialize_state(self):
        """Deserialize the byte buffer back to model components."""
        # Deserialize the model state from the buffer
        serialized_state = self.model_state_buffer.cpu().numpy().tobytes()
        state_dict = pickle.loads(serialized_state)
        # Set the deserialized values back into the model's attributes
        self.speaker_stats = state_dict['speaker_stats']
        self.stats_computed = state_dict['stats_computed']
        self.computed = state_dict['computed']
        self.keep_zeros = state_dict['keep_zeros']

    def state_dict(self, *args, **kwargs):
        self.serialize_state()
        """Override the state_dict to include the custom buffer."""
        state_dict = super().state_dict(*args, **kwargs)
        return state_dict

    def accumulate_stats(self, features, speaker_id):
        """
        Accumulate mean and variance statistics for a specific speaker.
        Args:
            features: Tensor of shape (T, D), where T is the number of frames and D is the feature dimensionality.
            speaker_id: Identifier for the speaker (string).
        """
        with torch.no_grad():
            if speaker_id not in self.speaker_stats:
                # Initialize accumulators for the speaker (as scalars)
                self.speaker_stats[speaker_id] = {
                    "sum": 0.0,           # Total sum of all feature values
                    "sum_sq": 0.0,        # Total sum of squares of feature values
                    "total_frames": 0     # Total number of frames accumulated
                }
                self.stats_computed[speaker_id] = False


            if self.keep_zeros:
                vv = features != 0
                features = features[vv]

            # Update stats for the speaker (sum and sum_sq as scalars)
            self.speaker_stats[speaker_id]["sum"] += features.sum().item()  # Sum of all values in features
            self.speaker_stats[speaker_id]["sum_sq"] += (features ** 2).sum().item()  # Sum of squared values
            self.speaker_stats[speaker_id]["total_frames"] += features.size(0)  # Number of frames processed

    def compute_global_stats(self, speaker_id):
        """
        Compute global mean and variance for a specific speaker.
        Args:
            speaker_id: Identifier for the speaker (string).
        """
        # if self.stats_computed.get(speaker_id, False):
        #     raise RuntimeError(f"Global statistics for speaker {speaker_id} have already been computed.")

        stats = self.speaker_stats[speaker_id]

        if stats["total_frames"] == 0:
            raise ValueError(f"No data accumulated for speaker {speaker_id}.")

        total_frames = stats["total_frames"]
        mean = stats["sum"] / total_frames
        var = (stats["sum_sq"] / total_frames) - (mean ** 2)
        std = torch.sqrt(torch.tensor(var + 1e-6))  # Avoid division by zero

        # Store computed stats
        stats["mean"] = torch.tensor(mean)
        stats["std"] = std
        self.stats_computed[speaker_id] = True
        self.computed = True

    def normalize(self, features, speaker_id):
        """
        Normalize features using computed mean and standard deviation for a specific speaker.
        Args:
            features: Tensor of shape (T, D).
            speaker_id: Identifier for the speaker (string).
        Returns:
            Normalized features: Tensor of shape (T, D).
        """
        if not self.stats_computed.get(speaker_id, False):
            if self.pass_though_if_not_computed:
                return features
            raise RuntimeError(f"Global statistics for speaker {speaker_id} must be computed before normalization.")

        mean = self.speaker_stats[speaker_id]["mean"].to(features.device)
        std = self.speaker_stats[speaker_id]["std"].to(features.device)
        if self.keep_zeros:
            vv = features != 0
            features[vv] = (features[vv] - mean) / std
            return features

        return (features - mean) / std

    def forward(self, features, speaker_id):
        """
        Forward method for normalization. Assumes statistics are precomputed.
        Args:
            features: Tensor of shape (T, D).
            speaker_id: Identifier for the speaker (string).
        Returns:
            Normalized features: Tensor of shape (T, D).
        """
        if self.model_state_buffer[0].item() != 0:
            self.deserialize_state()
            self.model_state_buffer[0] = 0
        if not self.computed:
            self.accumulate_stats(features, speaker_id)

        return self.normalize(features, speaker_id)


if __name__ == "__main__":
    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 2)
            self.speaker_cmvn = SpeakerCMVN()  # Register as a submodule


    a = MyModel()
    f0_norm = a.speaker_cmvn
    f0_norm.pass_though_if_not_computed = True
    for i in range(250):
        f0_norm(torch.tensor([148.1481, 158.4158, 168.4211, 173.9130, 183.9081, 186.0465, 186.0465,
            186.0465, 186.0465, 179.7753,   0.0000,   0.0000,   0.0000,   0.0000,
              0.0000,   0.0000,   0.0000, 155.3398, 150.9434, 150.9434, 150.9434,
            149.5327, 148.1481, 148.1481, 146.7890, 146.7890, 146.7890, 146.7890,
            146.7890, 146.7890, 146.7890, 146.7890, 148.1481, 148.1481, 152.3810,
              0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
              0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
              0.0000,   0.0000]), f"marie{i}")
    f0_norm(torch.tensor([148.1481, 158.4158, 168.4211, 173.9130, 183.9081, 186.0465, 186.0465,
        186.0465, 186.0465, 179.7753,   0.0000,   0.0000,   0.0000,   0.0000,
          0.0000,   0.0000,   0.0000, 155.3398, 150.9434, 150.9434, 150.9434,
        149.5327, 148.1481, 148.1481, 146.7890, 146.7890, 146.7890, 146.7890,
        146.7890, 146.7890, 146.7890, 146.7890, 148.1481, 148.1481, 152.3810,
          0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
          0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
          0.0000,   0.0000])*2, "pierre")
    for speaker_id in f0_norm.speaker_stats.keys():
        f0_norm.compute_global_stats(speaker_id)
        mean = f0_norm.speaker_stats[speaker_id]["mean"]
        std = f0_norm.speaker_stats[speaker_id]["std"]
        print(f"Speaker {speaker_id}: Mean={mean}, Std={std}")

    print(
    f0_norm(torch.tensor([158.4158, 168.4211, 173.9130, 183.9081, 186.0465, 186.0465,
        186.0465, 186.0465, 179.7753,   0.0000,   0.0000,   0.0000,   0.0000,
          0.0000,   0.0000,   0.0000, 155.3398, 150.9434, 150.9434, 150.9434,
        149.5327, 148.1481, 148.1481, 146.7890, 146.7890, 146.7890, 146.7890,
        146.7890, 146.7890, 146.7890, 146.7890, 148.1481, 148.1481, 152.3810,
          0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
          0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
          0.0000,   0.0000]), "pierre")
    )
    print(a.state_dict())
    torch.save(a.state_dict(), "/tmp/a.pkl")

    f0_norm = MyModel()
    f0_norm.load_state_dict(torch.load("/tmp/a.pkl"))
    f0_norm = f0_norm.speaker_cmvn
    print(
    f0_norm(torch.tensor([158.4158, 168.4211, 173.9130, 183.9081, 186.0465, 186.0465,
        186.0465, 186.0465, 179.7753,   0.0000,   0.0000,   0.0000,   0.0000,
          0.0000,   0.0000,   0.0000, 155.3398, 150.9434, 150.9434, 150.9434,
        149.5327, 148.1481, 148.1481, 146.7890, 146.7890, 146.7890, 146.7890,
        146.7890, 146.7890, 146.7890, 146.7890, 148.1481, 148.1481, 152.3810,
          0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
          0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
          0.0000,   0.0000]), "pierre")
    )
    print(
    f0_norm(torch.tensor([158.4158, 168.4211, 173.9130, 183.9081, 186.0465, 186.0465,
        186.0465, 186.0465, 179.7753,   0.0000,   0.0000,   0.0000,   0.0000,
          0.0000,   0.0000,   0.0000, 155.3398, 150.9434, 150.9434, 150.9434,
        149.5327, 148.1481, 148.1481, 146.7890, 146.7890, 146.7890, 146.7890,
        146.7890, 146.7890, 146.7890, 146.7890, 148.1481, 148.1481, 152.3810,
          0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
          0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
          0.0000,   0.0000]), "pierre")
    )
    print(f0_norm.state_dict())
