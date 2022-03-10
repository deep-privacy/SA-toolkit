# README

This folder contains librispeech recipes to train with pkwrap.

To run the recipe first activate your miniconda env and kaldi env:

```
cd ../../../../../
. ./env.sh
cd -
. ./path.sh
```

## E2E Chain recipe

To run the 100h setup, follow the steps below

1. Change the path to librispeech database in ``conf/local.conf``

2. Prepare data directories with

```
local/chain/e2e/prepare_data.sh
local/chain/e2e/get_egs.sh --data ./data/train_clean_100_sp_fbank_hires --fstdir ./exp/chain/e2e_biphone_tree --dir exp/chain/e2e_tdnnf/fst_egs
```

3. Train (and test ``dev_clean``) using the configuration below

```
local/chain/train.py --stage 4 --conf configs/tdnnf_e2e
```


```
find exp/chain/e2e_tdnnf/ exp/chain/e2e_tdnnf_vq_16 exp/chain/e2e_tdnnf_vq_32 exp/chain/e2e_tdnnf_vq_64 exp/chain/e2e_tdnnf_vq_48 exp/chain/e2e_tdnnf_vq_128 exp/chain/e2e_tdnnf_vq_256 exp/chain/e2e_tdnnf_vq_512 exp/chain/e2e_tdnnf_vq_1024  -not -path "*decode*" -not -name "*[0-9].pt" -not -path "*log*" -not -path "*runs*" -not -path "*egs*"  -exec zip asr_models.zip {} +
```

The model is stored in ``exp/chain/e2e_tdnnf/``. Once the entire script finishes successfully, the expected WER on ``dev_clean`` is

Add results to git:
```
git add e2e_tdnnf*/decode_dev_clean_fbank_hires_iterfinal_fg/scoringDetails/wer_details/* -f
```

### Share models
```bash
find exp/chain/e2e_tdnnf/ exp/chain/e2e_tdnnf_vq_16 exp/chain/e2e_tdnnf_vq_32 exp/chain/e2e_tdnnf_vq_64 exp/chain/e2e_tdnnf_vq_48 exp/chain/e2e_tdnnf_vq_128 exp/chain/e2e_tdnnf_vq_256 exp/chain/e2e_tdnnf_vq_512 exp/chain/e2e_tdnnf_vq_1024  -not -path "*decode*" -not -name "*[0-9].pt" -not -path "*log*" -not -path "*runs*" -not -path "*egs*"  -exec zip asr_models.zip {} +
```

```
a=exp/chain/e2e_tdnnf_wav2vec_fairseq_hibitrate; zip -r e2e_tdnnf_wav2vec_fairseq_hibitrate.zip $a/den.fst $a/final.mdl $a/0.trans_mdl $a/final.pt $a/normalization.fst $a/num_pdfs $a/tree
```
