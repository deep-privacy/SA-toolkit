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
./local/chain/prepare_data.sh --train_set train_600

# Train with archi and data defined in configs and local/chain/tuning/ (configs: model_file)
local/chain/train.py --conf configs/...
```

### Results train-clean-100 (Wav2vec2 large voxpopuli west_germanic_v2)
```sh
        Clean  Other
%WER    3.70   8.60     exp/bn_tdnnf_wav2vec2_t100_aug
# With vector quantization at the bottleneck extraction layer
%WER    4.77   11.84    exp/bn_tdnnf_wav2vec2_vq_48
```

### Results train-clean-100 (fbanks)
```sh
        Clean  Other
%WER    5.76   18.55    exp/bn_tdnnf_t100_aug
# With vector quantization at the bottleneck extraction layer
%WER    7.38   24.03    exp/bn_tdnnf_t100_vq_512
%WER    7.80   24.94    exp/bn_tdnnf_t100_vq_256
%WER    8.54   27.33    exp/bn_tdnnf_t100_vq_64
%WER    9.24   28.69    exp/bn_tdnnf_t100_vq_48
```
