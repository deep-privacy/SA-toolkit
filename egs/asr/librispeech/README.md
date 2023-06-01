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

_Up to 5 gpus can be used for training, you can use `ssh.pl` to distribute training on multiple node or max_concurrent_jobs config[exp] option to sequence the training, (using natural gradient and parameter averaging)._


### Results train-clean-100 (fbanks)
```sh
Test    Clean   Other    Exp                          Config
%WER     5.76   18.55    exp/bn_tdnnf_100h_aug        configs/tdnnf_bn
# Fine-tune with vector quantization at the bottleneck extraction layer
%WER     7.38   24.03    exp/bn_tdnnf_100h_vq_512     configs/tdnnf_bn_vq
%WER     7.80   24.94    exp/bn_tdnnf_100h_vq_256     configs/tdnnf_bn_vq
%WER     8.54   27.33    exp/bn_tdnnf_100h_vq_64      configs/tdnnf_bn_vq
%WER     9.24   28.69    exp/bn_tdnnf_100h_vq_48      configs/tdnnf_bn_vq
```


### Results train-clean-100 (Wav2vec2 large voxpopuli west_germanic_v2)

```sh
Test    Clean   Other    Exp                                Config
%WER     3.70    8.60    exp/bn_tdnnf_wav2vec2_100h_aug     configs/tdnnf_bn_wav2vec2
# Fine-tune with vector quantization at the bottleneck extraction layer
%WER     4.77   11.84    exp/bn_tdnnf_wav2vec2_vq_48        configs/tdnnf_bn_wav2vec2_vq
```

### Results train-600 (fbanks)
```sh
Test    Clean   Other    Exp                         Config
%WER     5.04   12.14    exp/bn_tdnnf_600h_aug       configs/tdnnf_bn_data_large
# Fine-tune with vector quantization at the bottleneck extraction layer
%WER     7.35   17.99    exp/bn_tdnnf_600h_vq_64     configs/tdnnf_bn_data_large_vq
```

### Results train-clean-360 (fbanks)
```sh
Test    Clean   Other    Exp                             Config
%WER     4.85   14.89    exp/asr_eval_tdnnf_360h_aug     configs/tdnnf_asr_eval
%WER     4.93   15.52    exp/asr_eval_tdnnf_360h         configs/tdnnf_asr_eval
```


### JIT model (extract_bn or acoustic likelihoods with forward)

```python3
import torch
import torchaudio
waveform, _, text_gt, speaker, chapter, utterance = torchaudio.datasets.LIBRISPEECH("/tmp", "dev-clean", download=True)[0]
model = torch.jit.load("__Exp_Path__/final.jit")
model = model.eval()

model.extract_bn(waveform)     # asrbn feature (BATCH, SEQ, FEAT)
loglikes,_ = model(waveform)   # asr loglikes (BATCH, SEQ, NbClass) NbClass = 3280 for left-biphone
```

### Decode loglikes with kaldi in python

The `transition`, `HCLG`, `words_txt`, ... files are also available in the model [releases](https://github.com/deep-privacy/SA-toolkit/releases).

```python3
import satools

txt, words, alignment, latt = satools.chain.decoder.kaldi_decode(
  loglikes,
  trans_model="exp/chain/asr_eval_tdnnf_360h/0.trans_mdl",
  HCLG="exp/chain/e2e_train_clean_360/e2e_biphone_tree/graph_tgsmall/HCLG.fst",
  words_txt="exp/chain/e2e_train_clean_360/e2e_biphone_tree/graph_tgsmall/words.txt")
print("Decoded text:", txt)

txt, words, alignment, latt_res = satools.chain.decoder.kaldi_lm_rescoring(
  latt,
  trans_model = "exp/chain/asr_eval_tdnnf_360h/0.trans_mdl",
  G_old = "data/lang_lp_test_tgsmall/G.fst",
  G_new = "data/lang_lp_test_fglarge/G.carpa",
  words_txt = "exp/chain/e2e_train_clean_360/e2e_biphone_tree/graph_tgsmall/words.txt")
print("LM rescored text:", txt)
```
