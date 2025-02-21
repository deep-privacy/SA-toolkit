<div align="center">
<h1 align='center'>SA-toolkit</h1>
<img src="https://user-images.githubusercontent.com/7476655/232308795-90cef60d-08dd-4964-96cd-2afb4a6c03b0.jpg" width="25%">
<h2 align='center'>SA-toolkit: Speaker speech anonymization toolkit in python</h2>
</div>

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-lightgray)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deep-privacy/SA-toolkit/blob/master/SA-colab.ipynb)
[![Gradio demo](https://img.shields.io/website-up-down-yellow-red/https/hf.space/gradioiframe/Champion/SA-toolkit/+.svg?label=🤗%20Hugging%20Face-Spaces%20demo)](https://huggingface.co/spaces/Champion/SA-toolkit)

SA-toolkit is a [pytorch](https://pytorch.org/)-based library providing pipelines and basic building blocs designing speaker anonymization techniques.  
This library is the result of the work of Pierre Champion's [thesis](https://arxiv.org/abs/2308.04455).  

Features include:

- :zap: Fast anonymization with a simple [`anonymize`](https://github.com/deep-privacy/SA-toolkit/tree/master?tab=readme-ov-file#zap-anonymize-bin) script
- ASR training with a pytorch [kaldi](https://github.com/kaldi-asr/kaldi) [LF-MMI wrapper](https://github.com/idiap/pkwrap) (evaluation, and VC linguistic feature extraction)
- VC HiFi-GAN training with on-the-fly feature caching (anonymization)
- ASV training (evaluation)
- WER Utility and EER/Linkability/Cllr Privacy evaluations
- Clear and simplified egs directories
- Unified trainer/configs
- TorchScript YAAPT & TorchScript kaldi.fbank (with batch processing!)
- On the fly _only_ feature extraction
- TorchScript JIT-compatible network models

_All `data` are formatted with kaldi-like wav.scp, spk2utt, text, etc._  
_Kaldi is necessary for training the ASR models and the handy `run.pl`/`ssh.pl`/`data_split`.. scripts, but most of the actual logic is performed in python; you won't have to deal kaldi ;)_


## Installation

### :snake: conda
The best way to install the SA-toolkit is with the `install.sh` script, which setup a micromamba environment, and kaldi.  
Take a look at the script and adapt it to your cluster configuration, or leave it do it's magic.  
This install is recommended for training ASR models.

```sh
git clone https://github.com/deep-privacy/SA-toolkit
./install.sh
```

### :package: pip
Another way of installing SA-toolkit is with pip3, this will setup everything for inference/testing.  
```sh
pip3 install 'git+https://github.com/deep-privacy/SA-toolkit.git@master#egg=satools&subdirectory=satools'
```

## :zap: Anonymize bin
Once installed (with any of the above ways), you will
have access to the [`anonymize`](./satools/satools/bin/anonymize) bin in your PATH that you can use together
with a config (example: [here](./egs/vc/libritts/configs/anon_pipelines)) to anonymize a kaldi like directory.
This script can make use of multiple GPUs, for faster anonymization.

```sh
anonymize --config ./configs/anon_pipelines --directory ./data/XXX
```


## PyTorch API
### Torch HUB anonymization example

This locally installs satools with Torch HUB (the required pip dependencies are: `torch` and `torchaudio`).  
This version gives access to the python/torch model for inference/testing, but for training use `install.sh`.
You can modify `tag_version` accordingly to the available model tag [here](https://github.com/deep-privacy/SA-toolkit/releases).

```python
import torch

model = torch.hub.load("deep-privacy/SA-toolkit", "anonymization", tag_version="hifigan_bn_tdnnf_wav2vec2_vq_48_v1", trust_repo=True)
wav_conv = model.convert(torch.rand((1, 77040)), target="1069")
asr_bn = model.get_bn(torch.rand((1, 77040))) # (ASR-BN extraction for disentangled linguistic features (best with hifigan_bn_tdnnf_wav2vec2_vq_48_v1))
```

<details>
<summary><h3>Torch JIT anonymization example</h3></summary>

This version does not rely on any dependencies using [TorchScript](https://pytorch.org/docs/stable/jit.html).

```python
import torch
import torchaudio
waveform, _, text_gt, speaker, chapter, utterance = torchaudio.datasets.LIBRISPEECH("/tmp", "dev-clean", download=True)[1]
torchaudio.save(f"/tmp/clear_{speaker}-{chapter}-{str(utterance)}.wav", waveform, 16000)

model = torch.jit.load("__Exp_Path__/final.jit").eval()
wav_conv = model.convert(waveform, target="1069")
torchaudio.save(f"/tmp/anon_{speaker}-{chapter}-{str(utterance)}.wav", wav_conv, 16000)
```
Ensure you have the model [downloaded](https://github.com/deep-privacy/SA-toolkit/releases).
Check the [egs/vc](egs/vc) directory for more detail.


</details>

## VPC 2024 performances

### tag_version=`hifigan_bn_tdnnf_wav2vec2_vq_48_v1`

**VPC-B5**  

```lua
---- ASV_eval^anon results ----
 dataset split gender enrollment trial     EER
   libri  test      f       anon  anon  33.946
   libri  test      m       anon  anon  34.729

---- ASR results ----
 dataset split       asr    WER
   libri   dev      anon  4.731
   libri  test      anon  4.369
```

### tag_version=`hifigan_bn_tdnnf_600h_vq_48_v1`

**VPC-B6**  

```lua
---- ASV_eval^anon results ----
 dataset split gender enrollment trial     EER
   libri  test      f       anon  anon  21.146
   libri  test      m       anon  anon  21.137

---- ASR results ----
 dataset split       asr    WER
   libri   dev      anon  9.693
   libri  test      anon  9.092
```

### tag_version=`hifigan_bn_tdnnf_wav2vec2_vq_48_v1+f0-transformation=quant_16_awgn_2`

**Add F0 transformations to B5**  

*With a stronger attacker than the VPC one (a better ASV model), the F0 transformation does not
get a higher EER than B5. (the VPC 2024 attack model is sensible to
F0 modification).*

```lua
---- ASV_eval^anon results ----
 dataset split gender enrollment trial     EER
   libri  test      f       anon  anon  42.151
   libri  test      m       anon  anon  40.755

---- ASR results ----
 dataset split       asr    WER
   libri   dev      anon  5.306
   libri  test      anon  4.814
```

### tag_version=`hifigan_inception_bn_tdnnf_wav2vec2_train_600_vq_48_v1+f0-transformation=quant_16_awgn_2`

*Experiment where libritts speech data is converted to a single speaker (using
an anonymization system), then used as training data for another anonymization
system.*  
ASR bottleneck extractor fine-tuned on librispeech 600 (rather than 100 like the
above).


```lua
---- ASV_eval^anon results ----
 dataset split gender enrollment trial     EER
   libri  test      f       anon  anon  35.765
   libri  test      m       anon  anon  35.195

---- ASR results ----
 dataset split       asr    WER
   libri   dev      anon  4.693
   libri  test      anon  4.209
```

## Model training

Checkout the READMEs of _[egs/asr/librispeech](egs/asr/librispeech)_ / _[egs/vc/libritts](egs/vc/libritts)_ / _[egs/asv/voxceleb](./egs/asv/voxceleb)_ .

## Evaluation

It is prefered to use the Voice-Privacy-Challenge-2024 evaluation tool as this SA-toolkit library was used for two baselines (B5 and B6)

## Citation

This library is the result of the work of Pierre Champion's thesis.  
If you found this library useful in academic research, please cite:

```bibtex
@phdthesis{champion2023,
    title={Anonymizing Speech: Evaluating and Designing Speaker Anonymization Techniques},
    author={Pierre Champion},
    year={2023},
    school={Université de Lorraine - INRIA Nancy},
    type={Thesis},
}
```

(Also consider starring the project on GitHub.)

## Acknowledgements
* Idiap' [pkwrap](https://github.com/idiap/pkwrap)
* Jik876's [HifiGAN](https://github.com/jik876/hifi-gan)
* A.Larcher's [Sidekit](https://git-lium.univ-lemans.fr/speaker/sidekit)
* Organazers of the [VoicePrivacy](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2022) Challenge

## License
Most of the software is distributed under Apache 2.0 License (http://www.apache.org/licenses/LICENSE-2.0); the parts distributed under other licenses are indicated by a `LICENSE` file in related directories.
