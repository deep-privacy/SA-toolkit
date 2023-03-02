# README

This folder contains librispeech recipes to train with the SA-toolkit.

To run the recipe:

```bash
# Activate your miniconda env and kaldi env
. ./path.sh

#  Change the path to librispeech database in `conf/local.conf` and/or use `local/download_libri.sh`
./local/chain/e2e/prepare_data.sh --train_set train_clean_100
./local/chain/e2e/prepare_data.sh --train_set train_clean_360
./local/chain/e2e/prepare_data.sh --train_set train_600

# Train with archi and data defined in configs and local/chain/e2e/tuning/ (configs: model_file)
local/chain/train.py --conf configs/...
```
