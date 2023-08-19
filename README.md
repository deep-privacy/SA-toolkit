<div align="center">
<h1 align='center'>SA-toolkit</h1>
<img src="https://user-images.githubusercontent.com/7476655/232308795-90cef60d-08dd-4964-96cd-2afb4a6c03b0.jpg" width="25%">
<h2 align='center'>SA-toolkit: Speaker speech anonymization toolkit in python</h2>
</div>

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-lightgray)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deep-privacy/SA-toolkit/blob/master/SA-colab.ipynb)
[![Gradio demo](https://img.shields.io/website-up-down-yellow-red/https/hf.space/gradioiframe/Champion/SA-toolkit/+.svg?label=🤗%20Hugging%20Face-Spaces%20demo)](https://huggingface.co/spaces/Champion/SA-toolkit)

SA-toolkit is a [pytorch](https://pytorch.org/)-based library providing pipelines and basic building blocs for evaluating and designing speaker anonymization techniques.

Features include:

- ASR training with a pytorch [kaldi](https://github.com/kaldi-asr/kaldi) [LF-MMI wrapper](https://github.com/idiap/pkwrap) (evaluation, and VC linguistic feature)
- VC HiFi-GAN training with on-the-fly feature caching (anonymization)
- ASV training (evaluation)
- WER Utility and EER/Linkability/Cllr Privacy evaluations
- Clear and simplified egs directories
- Unified trainer/configs
- TorchScript YAAPT & TorchScript kaldi.fbank (with batch processing!)
- On the fly _only_ feature extraction
- 100% TorchScript JIT-compatible network models

_All `data` are formatted with kaldi-like wav.scp, spk2utt, text, etc._  
_Kaldi is necessary for training the ASR models and the handy `run.pl`/`ssh.pl`/`data_split`.. scripts, but most of the actual logic is performed in python; you won't have to deal kaldi ;)_


## Installation

The best way to install the toolkit is with the `install.sh` script, which setup a miniconda environment, and kaldi.
Take a look at the script and adapt it to your cluster configuration, or leave it do it's magic.

```sh
git clone https://github.com/deep-privacy/SA-toolkit
./install.sh
```

## Quick Torch HUB anonymization example

This locally installs satools (the required pip dependencies are: `torch`, `torchaudio`, `soundfile` and `configargparse`).  
This version gives access to the python/torch model for inference/testing, but for training use `install.sh`.
You can modify `tag_version` accordingly to the available model tag [here](https://github.com/deep-privacy/SA-toolkit/releases).

```python
import torch

model = torch.hub.load("deep-privacy/SA-toolkit", "anonymization", tag_version="hifigan_bn_tdnnf_wav2vec2_vq_48_v1", trust_repo=True)
wav_conv = model.convert(torch.rand((1, 77040)), target="1069")
```

## Quick JIT anonymization example

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

## Quick evaluation example

```sh
cd egs/anon/vctk
./local/eval.py --config configs/eval_clear  # eval privacy/utility of the signals
```
Ensure you have the corresponding evaluation model trained or [downloaded](https://github.com/deep-privacy/SA-toolkit/releases).

## Model training

Checkout the READMEs of _[egs/asr/librispeech](egs/asr/librispeech)_ / _[egs/asv/voxceleb](./egs/asv/voxceleb)_ / _[egs/vc/libritts](egs/vc/libritts)_.

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

## Evaluation choices
_As outlined in the thesis, selecting the appropriate target identities for voice conversion is crucial for privacy evaluation. We strongly encourage the use of any-to-one voice conversion as it provides the greatest level of guarantee regarding unlinkable speech generation and facilitates proper training of a white-box ASV evaluation model. Additionally, this approach is easy to comprehend (everyone should sounds like a single identity) and enables using one-hot encoding for target identity representation, which is simpler than x-vectors while still highly effective for utility preservation.  
Furthermore, the thesis identifies a limitation in the current utility evaluation process. We believe that the best solution for proper assessment of utility is through subjective listening, which allows for accurate evaluation of any mispronunciations produced by the VC system._
