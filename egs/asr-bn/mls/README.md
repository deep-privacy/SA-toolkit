# README

This folder contains mls recipes to train with pkwrap.

To run the recipe first activate your miniconda env and kaldi env:

```
./local/make_links.sh `pwd`/../../../kaldi
. ../../../env.sh
. ../path.sh
```

## E2E Chain recipe

To run the 1000h setup (train split of mls), follow the steps below

1. Change the path to mls database in ``configs/local.conf`` (downloadable here : http://openslr.org/94/)

2. Prepare data directories with

```
local/chain/e2e/prepare_data.sh
local/chain/e2e/get_egs.sh --data ./data/mls_train_sp_fbank_hires --fstdir ./exp/chain/e2e_biphone_tree --dir exp/chain/e2e_tdnnf/fst_egs
```

3. Train (and test ``dev``) using the configuration below

```
local/chain/train.py --stage 4 --conf configs/tdnnf_e2e_wav2vec2_fairseq_hibitrate
```


```
find exp/chain/e2e_wav2vec2_fairseq_hibitrate/  -not -path "*decode*" -not -name "*[0-9].pt" -not -path "*log*" -not -path "*runs*" -not -path "*egs*"  -exec zip asr_models.zip {} +
```

The model is stored in ``exp/chain/e2e_wav2vec2_fairseq_hibitrate/``. Once the entire script finishes successfully, the expected WER on ``dev_clean`` is

Add results to git:
```
git add e2e_wav2vec2_fairseq_hibitrate/decode_dev_clean_fbank_hires_iterfinal_fg/scoringDetails/wer_details/* -f
```