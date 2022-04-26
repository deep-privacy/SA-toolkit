"""This lobe enables the integration of huggingface pretrained wav2vec2 models.

Reference: https://arxiv.org/abs/2006.11477
Reference: https://arxiv.org/abs/1904.05862
Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html
"""

import os

import torch
import torch.nn.functional as F
from torch import nn

##########################################################################
#  THOSE MODELS DONT HAVE A FIXED DOWN-SAMPLE RATE! Not sutable for tts  #
##########################################################################


class HuggingFaceWav2Vec2(nn.Module):
    """This lobe enables the integration of HuggingFace
    pretrained wav2vec2.0 models.

    Source paper: https://arxiv.org/abs/2006.11477
    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    The model can be used as a fixed feature extractor or can be finetuned. It
    will download automatically the model from HuggingFace.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "facebook/wav2vec2-large-lv60"
    sr: int
        Sampling rate for the wav2vec2 processor
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    freeze_feature_extractor :  bool (default: False)
        When freeze = False and freeze_feature_extractor True, the featue_extractor module of the model is Frozen. If False
        all the wav2vec model will be trained including featue_extractor module.
    Example
    -------
    >>> inputs = torch.rand([10, 600])
    >>> model_hub = "facebook/wav2vec2-base-960h"
    >>> model = HuggingFaceWav2Vec2(model_hub)
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([10, 1,  768])
    """

    def __init__(
        self,
        source,
        revision=None,
        sr=16000,
        freeze=True,
        freeze_feature_extractor=False,
    ):
        super().__init__()

        # We check if transformers is installed.
        try:
            from transformers import Wav2Vec2Model, Wav2Vec2Processor
            from transformers import Wav2Vec2FeatureExtractor
        except ImportError as e:
            print("Please install transformer from HuggingFace to use wav2vec2!")
            raise e

        save_me = False
        _source = source
        model_hub = source
        if os.path.exists(f"/tmp/{model_hub.replace('/', '-')}/featext_config.json"):
            _source = f"/tmp/{model_hub.replace('/', '-')}/featext_config.json"
        else:
            save_me = True

        # Download the extractor from HuggingFace.
        # The extractor is only used to retrieve the normalisation
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            _source, revision=revision
        )
        if save_me:
            self.feature_extractor.save_pretrained(
                f"/tmp/{model_hub.replace('/', '-')}/featext_config.json"
            )
        _source = source

        # sampling rate
        self.sr = sr

        save_me = False
        if os.path.exists(f"/tmp/{model_hub.replace('/', '-')}/model_config.json"):
            _source = f"/tmp/{model_hub.replace('/', '-')}/model_config.json"
        else:
            save_me = True

        try:
            # Download the model from HuggingFace.
            if revision:
                self.model = Wav2Vec2Model.from_pretrained(_source, revision=revision)
            self.model = Wav2Vec2Model.from_pretrained(_source)
        except:
            _source = source
            if revision:
                self.model = Wav2Vec2Model.from_pretrained(_source, revision=revision)
            self.model = Wav2Vec2Model.from_pretrained(_source)
            os.system(
                f"rm -rf /tmp/{model_hub.replace('/', '-')}/featext_config.json; rm -rf /tmp/{model_hub.replace('/', '-')}/model_config.json"
            )

        if save_me:
            self.model.save_pretrained(
                f"/tmp/{model_hub.replace('/', '-')}/model_config.json"
            )
        _source = source

        # We check if inputs need to be normalized w.r.t pretrained wav2vec2
        self.normalize_wav = self.feature_extractor.do_normalize

        self.freeze = freeze
        self.freeze_feature_extractor = freeze_feature_extractor
        if self.freeze:
            self.model.eval()
        else:
            self.model.train()
            if self.freeze_feature_extractor:
                self.model.feature_extractor._freeze_parameters()

    def forward(self, wav, spec_aug=False):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """

        self.model.config.apply_spec_augment = spec_aug

        # If we freeze, we simply remove all grads and features from the graph.
        if self.freeze:
            with torch.no_grad():
                return self.extract_features(wav).detach()

        return self.extract_features(wav)

    def extract_features(self, wav):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """

        if self.normalize_wav:
            wav = F.layer_norm(wav, wav.shape)

        # Extract wav2vec output
        out = self.model(
            wav,
        ).last_hidden_state

        return out
