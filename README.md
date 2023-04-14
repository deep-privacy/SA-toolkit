<div align="center">
<h1 align='center'>SA-toolkit</h1>
<img src="https://user-images.githubusercontent.com/7476655/232167135-d6e82df7-5a3e-4d6b-b624-3503e0fafc79.png" width="17%">
<h2 align='center'>SA-toolkit: Speaker speech anonymization toolkit in python</h2>
</div>

SA-toolkit is a [pytorch](https://pytorch.org/)-based library providing evaluation pipelines and basic building blocs and for evaluating and designing speaker anonymization techniques.

Features include:

- ASR training with a pytorch [kaldi](https://github.com/kaldi-asr/kaldi) [LF-MMI wrapper](https://github.com/idiap/pkwrap) (evaluation, and VC linguistic feature)
- VC HiFi-GAN training with on-the-fly feature caching (anonymization)
- ASV training (evaluation)
- Clear and simplified egs directories
- Unified trainer/configs
- On the fly _only_ feature extraction
- 100% JIT-compatible network

_All `data` structure is formatted with kaldi-like wav.scp, spk2utt, text, etc. Kaldi is necessary, but most of the actual logic is performed in python; it should not bother you ;)_


## Installation

The best way to install the toolkit is through the `install.sh`, which setup a miniconda environment.
```sh
git clone https://github.com/deep-privacy/SA-toolkit
./install.sh
```

## Quick evaluation example

```sh
$ cd egs/anon/vctk
$./local/eval.py --config configs/eval_clear  # eval privacy/utility of the signal
```
Ensure you have the corresponding evaluation model trained or [downloaded](https://github.com/deep-privacy/SA-toolkit/releases).

## Model training

Checkout the READMEs of _[egs/asr/librispeech](egs/asr/librispeech)_ / _[egs/asv/voxceleb](./egs/asv/voxceleb)_ / _[egs/vc/libritts](egs/vc/libritts)_.

## Citation

This library is the result of the work of Pierre Champion's thesis.
If you found this library useful in academic research, please cite [(arXiv link)](https://arxiv.org/abs/???).

```bibtex
@phdthesis{champion2023,
    title={{A}nonymizing {S}peech: {E}valuating and {D}esigning {S}peaker {A}nonymization {T}echniques},
    author={Pierre Champion},
    year={2023},
    school={Université de Lorraine},
}
```

(Also consider starring the project on GitHub.)

## Acknowledgements
* Idiap' [pkwrap](https://github.com/idiap/pkwrap)
* Jik876's [HifiGAN](https://github.com/jik876/hifi-gan)
* A.Larcher's [Sidekit](https://git-lium.univ-lemans.fr/speaker/sidekit)
* Organazers of the [VoicePrivacy](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2022) Challenge

## License
Most of the software is distributed under Apache 2.0 License (http://www.apache.org/licenses/LICENSE-2.0), the parts of the software that is distributed under other licenses is indicated by a `LICENSE` file in the relatated direcotry.

## Evaluation choices
_As discussed in the thesis, one key element to proper privacy evaluation is selecting the target identities for voice conversion.
We want to push the community to use any-to-one voice conversion because it is the type of VC with the most guaranty regarding unlinkable speech generation and proper training of a white-box asv evaluation model.
In addition, it is simple to understand (everyone should sounds-like one identity) and allows using one-hot encoding as the target identity representation, which is more straightforward than x-vectors while having good utility.
Also discussed in the thesis, the current utility evaluation has a limitation, the use of subjective listening to asses the mispronunciation that the vc system produce is the best option in our opinion._
