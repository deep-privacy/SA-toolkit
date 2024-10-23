import torch
from . import kaldi

from typing import Union

class WavInfo(object):
    def __init__(self, wav:torch.Tensor, name:str, filename:str):
        self.name = name
        self.filename = filename
        self.wav = wav

    def to(self, device:str="cpu"):
        self.wav = self.wav.to(device)
        return self

    def __repr__(self):
        return f"(name={self.name}, wav={self.wav.shape}, filename={self.filename})"


class WavScpDataset(torch.utils.data.Dataset):
    def __init__(self, wavs_paths, wavs_idx, load_func=None):
        if isinstance(wavs_paths, str):
            self.wavs_path = wavs_paths.split(",")
            self.wavs_idx = wavs_idx.split(",")
        else:
            self.wavs_path = wavs_paths
            self.wavs_idx = wavs_idx

        assert len(self.wavs_path) == len(self.wavs_idx)
        if load_func == None:
            self.load = kaldi.load_wav_from_scp
        else:
            self.load = load_func

    @classmethod
    def from_wav_scpfile(cls, wav_scp_file, load_func=None):
        wavs_scp = kaldi.read_wav_scp(wav_scp_file)
        return cls(list(wavs_scp.values()), list(wavs_scp.keys()), load_func=load_func)

    def __len__(self):
        return len(self.wavs_path)

    def __getitem__(self, idx):
        filename = self.wavs_path[idx]
        waveform, sr = self.load(filename)
        return WavInfo(waveform, self.wavs_idx[idx], filename)

def parse_wavinfo_wav(wavinfo: Union[WavInfo, torch.Tensor]) -> torch.Tensor:
    if torch.jit.isinstance(wavinfo, torch.Tensor):
        wav = wavinfo.detach().clone()
    else:
        wav = wavinfo.wav.detach().clone()
    return wav

