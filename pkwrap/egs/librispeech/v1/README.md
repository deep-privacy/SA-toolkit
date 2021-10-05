# README

This folder contains librispeech recipes to train with pkwrap.

To run the recipe first setup the data folders:

```
. ./path.sh
# run this only if links are not already there
./make_links.sh
```

## E2E Chain recipe

To run the 100h setup, follow the steps below

1. Change the path to librispeech database in ``conf/local.conf``

2. Prepare data directories with

```
local/chain/e2e/prepare_data.sh
```

3. Train (and test ``dev_clean``) using the configuration below

```
local/chain/train.py --stage 4 --conf configs/tdnnf_e2e
```

The model is stored in ``exp/chain/e2e_tdnnf/``. Once the entire script finishes successfully, the expected WER on ``dev_clean`` is

Add results to git:
```
git add e2e_tdnnf*/decode_dev_clean_fbank_hires_iterfinal_fg/scoringDetails/wer_details/* -f
```
