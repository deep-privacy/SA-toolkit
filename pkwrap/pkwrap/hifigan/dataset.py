import torch
import torchaudio

import numpy as np
import random

from . import f0


class WavList(torch.utils.data.Dataset):
    def __init__(self, wavs_paths):
        self.wavs_path = wavs_paths.split(",")

    def __len__(self):
        return len(self.wavs_path)

    def __getitem__(self, idx):
        filename = self.wavs_path[idx]
        waveform, sr = torchaudio.load(filename)
        return (waveform, filename)


def collate_fn_padd(f0_stats_file, get_func=None):
    def _func_pad(batch):
        filenames = [b[1] for b in batch]
        batch = [b[0] for b in batch]
        """
        Padds batch of variable length and compute f0
        """
        ## get sequence lengths
        lengths = torch.tensor([t.shape[1] for t in batch])
        ## padd
        batch = [torch.Tensor(t).permute(1, 0) for t in batch]
        feats = torch.nn.utils.rnn.pad_sequence(
            batch, batch_first=True, padding_value=0
        )
        feats = torch.squeeze(feats.permute(0, 2, 1))

        if len(feats.shape) == 1:
            feats = torch.unsqueeze(feats, 0)

        f0s = []
        for i, b in enumerate(batch):
            f0s.append(
                f0.get_f0(
                    b.permute(0, 1),
                    f0_stats_file=f0_stats_file,
                    cache_with_filename=filenames[i],
                )
                .squeeze(dim=1)
                .permute(1, 0),
            )

        f0spad = torch.nn.utils.rnn.pad_sequence(f0s, batch_first=True, padding_value=0)
        f0s = f0spad.permute(0, 2, 1)

        if get_func != None:
            return get_func(feats, lengths, filenames, f0s)
        return feats, lengths, filenames, f0s

    return _func_pad


def sample_interval(seqs, seq_len):
    N = max([v.shape[-1] for v in seqs])

    hops = [N // v.shape[-1] for v in seqs]
    lcm = np.lcm.reduce(hops)

    # Randomly pickup with the batch_max_steps length of the part
    interval_start = 0
    interval_end = N // lcm - seq_len // lcm

    start_step = random.randint(interval_start, interval_end)

    new_seqs = []
    for i, v in enumerate(seqs):
        start = start_step * (lcm // hops[i])
        end = (start_step + seq_len // lcm) * (lcm // hops[i])
        new_seqs += [v[..., start:end]]

    return new_seqs
