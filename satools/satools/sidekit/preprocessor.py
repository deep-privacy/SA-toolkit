"""
This file is part of SIDEKIT.
Copyright 2014-2023 Anthony Larcher
"""

import torch
import torchaudio
from contextlib import redirect_stdout
from .. import utils
from .. import augmentation


class MfccFrontEnd(torch.nn.Module):
    """
    Module that extract MFCC coefficients
    """
    def __init__(self,
                 pre_emphasis=0.97,
                 sample_rate=16000,
                 n_fft=2048,
                 f_min=133.333,
                 f_max=6855.4976,
                 win_length=1024,
                 window_fn=torch.hann_window,
                 hop_length=512,
                 power=2.0,
                 n_mels=100,
                 n_mfcc=80):

        super(MfccFrontEnd, self).__init__()

        self.pre_emphasis = pre_emphasis
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max
        self.win_length = win_length
        self.window_fn=window_fn
        self.hop_length = hop_length
        self.power=power
        self.window_fn = window_fn
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

        self.PreEmphasis = augmentation.PreEmphasis(self.pre_emphasis)

        self.melkwargs = {"n_fft":self.n_fft,
                          "f_min":self.f_min,
                          "f_max":self.f_max,
                          "win_length":self.win_length,
                          "window_fn":self.window_fn,
                          "hop_length":self.hop_length,
                          "power":self.power,
                          "n_mels":self.n_mels}

        self.MFCC = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            dct_type=2,
            log_mels=True,
            melkwargs=self.melkwargs)

        self.CMVN = torch.nn.InstanceNorm1d(self.n_mfcc)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=False):
                mfcc = self.PreEmphasis(x)
                mfcc = self.MFCC(mfcc)
                mfcc = self.CMVN(mfcc)
        return mfcc


class WavLmFrontEnd(torch.nn.Module):
    """Pretrained WavLM feature extractor (https://arxiv.org/pdf/2110.13900.pdf)
    implementation inspired from https://github.com/microsoft/UniSpeech/blob/main/downstreams/speaker_verification/models/ecapa_tdnn.py

    Args:
        update_extract (bool): allows finetuning if True
        channels_dropout (float in [0, 1]): channel dropout probability
    """

    def __init__(self, update_extract=False, channels_dropout=0.0):
        super(WavLmFrontEnd, self).__init__()
        self.feat_type = 'wavlm_large'
        # supress import error for other ssl pre-trained model than wavlm.
        with redirect_stdout(utils.StdFilterOut(ignore="can not import s3prl.", to_keep="wavlm")):
            self.feature_extract = torch.hub.load('s3prl/s3prl', self.feat_type)
        self.update_extract = update_extract
        self.feature_selection = 'hidden_states'
        self.sr = 16000
        self.feat_num = self.get_feat_num()
        self.instance_norm = torch.nn.InstanceNorm1d(1024)
        self.channels_dropout = channels_dropout
        self.feature_weight = torch.nn.Parameter(torch.zeros(self.feat_num))
        freeze_list = ['final_proj', 'label_embs_concat', 'mask_emb', 'project_q', 'quantizer']
        for name, param in self.feature_extract.named_parameters():
            for freeze_val in freeze_list:
                if freeze_val in name:
                    param.requires_grad = False
                    break

        if not self.update_extract:
            for param in self.feature_extract.parameters():
                param.requires_grad = False

    def get_feat_num(self):
        """

        :return:
        """
        self.feature_extract.eval()
        wav = [torch.randn(self.sr).to(next(self.feature_extract.parameters()).device)]
        with torch.no_grad():
            features = self.feature_extract(wav)
        select_feature = features[self.feature_selection]
        if isinstance(select_feature, (list, tuple)):
            return len(select_feature)
        else:
            return 1

    def get_feat(self, x):
        """

        :param x:
        :return:
        """
        if self.update_extract:
            x = self.feature_extract([sample for sample in x])
        else:
            with torch.no_grad():
                x = self.feature_extract([sample for sample in x])

        x = x[self.feature_selection]
        if isinstance(x, (list, tuple)):
            x = torch.stack(x, dim=0)
        else:
            x = x.unsqueeze(0)
        norm_weights = torch.nn.functional.softmax(self.feature_weight, dim=-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = (norm_weights * x).sum(dim=0)
        x = torch.transpose(x, 1, 2) + 1e-6

        x = self.instance_norm(x)

        if self.training:
            x *= torch.nn.functional.dropout(torch.ones((1, 1, x.shape[2]), device=x.device), p=self.channels_dropout)

        return x

    def forward(self, x):
        """

        :param x:
        :return:
        """

        return self.get_feat(x)

class MelSpecFrontEnd(torch.nn.Module):
    """
    Module that compute Mel spetrogramm on an audio signal
    """
    def __init__(self,
                 pre_emphasis=0.97,
                 sample_rate=16000,
                 n_fft=1024,
                 f_min=90,
                 f_max=7600,
                 win_length=400,
                 window_fn=torch.hann_window,
                 hop_length=160,
                 power=2.0,
                 n_mels=80):

        super(MelSpecFrontEnd, self).__init__()

        self.pre_emphasis = pre_emphasis
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max
        self.win_length = win_length
        self.window_fn=window_fn
        self.hop_length = hop_length
        self.power=power
        self.window_fn = window_fn
        self.n_mels = n_mels

        self.PreEmphasis = augmentation.PreEmphasis(self.pre_emphasis)

        self.melkwargs = {"n_fft":self.n_fft,
                          "f_min":self.f_min,
                          "f_max":self.f_max,
                          "win_length":self.win_length,
                          "window_fn":self.window_fn,
                          "hop_length":self.hop_length,
                          "power":self.power,
                          "n_mels":self.n_mels}

        self.MelSpec = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate,
                                                            n_fft=self.melkwargs['n_fft'],
                                                            f_min=self.melkwargs['f_min'],
                                                            f_max=self.melkwargs['f_max'],
                                                            win_length=self.melkwargs['win_length'],
                                                            hop_length=self.melkwargs['hop_length'],
                                                            window_fn=self.melkwargs['window_fn'],
                                                            power=self.melkwargs['power'],
                                                            n_mels=self.melkwargs['n_mels'])

        self.CMVN = torch.nn.InstanceNorm1d(self.n_mels)
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=5)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=False):
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                out = self.PreEmphasis(x)
                out = self.MelSpec(out)+1e-6
                out = torch.log(out)
                out = self.CMVN(out)
                if self.training:
                    out = self.freq_masking(out)
                    out = self.time_masking(out)
        return out
