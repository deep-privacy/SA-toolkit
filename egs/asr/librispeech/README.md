Acoustic model / ASR
===

This folder contains librispeech recipes to train ASR / linguistic feature extractor.

To run the recipe:

```bash
# Activate your miniconda env and kaldi env
. ./path.sh

#  Change the path to librispeech database in `configs/local.conf` and/or use `local/download_libri.sh`
./local/chain/prepare_data.sh --train_set train_clean_100
./local/chain/prepare_data.sh --train_set train_clean_360
./local/chain/prepare_data.sh --train_set train_600 # train-clean-100 + train-other-500

# Train with archi and data defined in configs and local/chain/tuning/ (configs: model_file)
local/chain/train.py --conf configs/...
```


### Results train-clean-100 (Wav2vec2 large voxpopuli west_germanic_v2)

```sh
Test    Clean   Other    Exp                                Config
%WER     3.70    8.60    exp/bn_tdnnf_wav2vec2_t100_aug     configs/tdnnf_bn_wav2vec2
# Fine-tune with vector quantization at the bottleneck extraction layer
%WER     4.77   11.84    exp/bn_tdnnf_wav2vec2_vq_48        configs/tdnnf_bn_wav2vec2_vq
```

### Results train-clean-100 (fbanks)
```sh
Test    Clean   Other    Exp                          Config
%WER     5.76   18.55    exp/bn_tdnnf_t100_aug        configs/tdnnf_bn
# Fine-tune with vector quantization at the bottleneck extraction layer
%WER     7.38   24.03    exp/bn_tdnnf_t100_vq_512     configs/tdnnf_bn_vq
%WER     7.80   24.94    exp/bn_tdnnf_t100_vq_256     configs/tdnnf_bn_vq
%WER     8.54   27.33    exp/bn_tdnnf_t100_vq_64      configs/tdnnf_bn_vq
%WER     9.24   28.69    exp/bn_tdnnf_t100_vq_48      configs/tdnnf_bn_vq
```

### Results train-600 (fbanks)
```sh
Test    Clean   Other    Exp                         Config
%WER     5.04   12.14    exp/bn_tdnnf_t600_aug       configs/tdnnf_bn_data_large
# Fine-tune with vector quantization at the bottleneck extraction layer
%WER     7.35   17.99    exp/bn_tdnnf_t100_vq_64     configs/tdnnf_bn_data_large_vq
```

### Results train-clean-360 (fbanks)
```sh
Test    Clean   Other    Exp                             Config
%WER     4.85   14.89    exp/asr_eval_tdnnf_t360_aug     configs/tdnnf_asr_eval
```


### JIT model (extract_bn or posterior with forward)

```python3
import torch
import torchaudio
waveform, _, text_gt, speaker, chapter, utterance = torchaudio.datasets.LIBRISPEECH("/tmp", "dev-clean", download=True)[0]
model = torch.jit.load("__Exp_Path__/final.jit")
model = model.eval()

model.extract_bn(waveform) # asrbn feature (BATCH, SEQ, FEAT)
post,_ = model(waveform)   # asr posterior (BATCH, SEQ, NbClass) NbClass = 3280 for left-biphone
```
