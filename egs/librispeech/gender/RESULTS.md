# RESULTS

| Subset          | Female spkrs | Male spkrs | Total spkrs |
|-----------------|--------------|------------|-------------|
| all             | 1201         | 1283       | 2484        |
| dev-clean       | 20           | 20         | 40          |
| test-clean      | 20           | 20         | 40          |
| dev-other       | 16           | 17         | 33          |
| test-other      | 17           | 16         | 33          |
| train-clean-100 | 125          | 126        | 251         |
| train-960       | 1128         | 1210       | 2338        |

## trained on 60% of train-clean-100 (see ../spk_identif/README.md)

## Gender
**The Gender branch wasn't used in the ESPnet computation graph.** The Gender
branch was used as a indicator on how much Gender information is hidden in a
classic ASR encoder.

## Environments 

##### `damped`:
- date: `Mon Jan 20 10:20:40 2020 +0100`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `10733370339d72a1ea931b643de3e9e7824e8d29`
  - Commit date: `Mon Jan 20 10:20:40 2020 +0100`


## Baseline on a trained ESPnet (No modification made to backpropagation)
```log
BrijSpeakerXvector(
  (advnet): LSTM(1024, 512, num_layers=3, batch_first=True, dropout=0.2)
  (segment6): Linear(in_features=512, out_features=512, bias=True)
  (segment7): Linear(in_features=512, out_features=512, bias=True)
  (segment8): Linear(in_features=1024, out_features=1024, bias=True)
  (bn2): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (output): Linear(in_features=1024, out_features=2, bias=True)
)
```

### Test Other
The speakers are from *Test Other* and train_clean_100 are disjoint.
```log
Accuracy:  0.8389
        Precision    Recall    Fscore
Female   0.823845  0.885529  0.853574
Male     0.858917  0.786353  0.821035

                  Predicted
     True  Female   Male 
    Female    0.9    0.1 
      Male    0.2    0.8 
```

### 20% of train-clean-100
```log
Accuracy:  0.9641
        Precision    Recall    Fscore
Female   0.946154  0.984000  0.964706
Male     0.983471  0.944444  0.963563

                  Predicted
     True  Female   Male 
    Female    1.0    0.0 
      Male    0.1    0.9 
```

## ASR
## Environments 
**Using forked version of ESPnet with damped weights sharing functions.**

##### `ESPnet`:
- date: `Mon Jan 20 10:33:17 CET 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.4`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `696e2e2cfeaa6f586666120752e0080607d80362`
  - Commit date: `Mon Jan 20 10:21:42 2020 +0100`

**ESPnet was modified according to this [diff](https://github.com/espnet/espnet/compare/e88a477cb72be7e5a03595ead5c233f8d211f6b6...deep-privacy:e53e1718d0943b9efeeec45ce8ae0e15367f2fdf)**

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dev_clean_model.acc.best_decode_lm|2703|54402|96.2|3.4|0.4|0.5|4.3|43.6|
|decode_dev_other_model.acc.best_decode_lm|2864|50948|89.0|9.8|1.2|1.8|12.8|68.1|
|decode_test_clean_model.acc.best_decode_lm|2620|52576|96.1|3.5|0.4|0.6|4.4|44.0|
|decode_test_other_model.acc.best_decode_lm|2939|52343|87.9|10.7|1.3|1.8|13.9|72.1|

Those scores are very close to the one reported by the ESPnet team: [source](https://github.com/espnet/espnet/blob/3b83007b43b79c7c0730f45b06783bd478ce87e7/egs/librispeech/asr1/RESULTS.md#pytorch-vgg-3blstm-1024-units-bpe-5000-latest-rnnlm-training-with-tuned-decoding-ctc_weight05-lm_weight07-dropout-02)
