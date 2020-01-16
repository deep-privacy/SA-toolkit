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

## trained on train_960

## Gender
**The Gender branch wasn't used in the ESPnet computation graph.** The Gender
branch was used as a indicator on how much Gender information is hidden in a
classic ASR encoder.

## Environments 
**Using forked version of ESPnet with damped weights sharing functions.**

- date: `Wed Jan 15 12:08:01 CET 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `7a278a22c68930b63109f9847ee61896cd5f8d5c`
  - Commit date: `Wed Jan 15 12:13:09 2020 +0100`


## Baseline on a trained ESPnet (No modification made to backpropagation)
```log
GenderNet(
  (lstm): LSTM(1024, 782, batch_first=True)
  (fc1): Linear(in_features=782, out_features=2, bias=True)
  (softmax): Softmax()
)
```

### Test Other
```log
          Predicted
     True  Female   Male
    Female    0.8    0.2
      Male    0.2    0.8

Accuracy:  0.8054
        Precision    Recall    Fscore
Female   0.800746  0.778665  0.789551
Male     0.809256  0.828956  0.818987
```

### Test Clean
```log
          Predicted
     True  Female   Male
    Female    0.8    0.2
      Male    0.1    0.9

Accuracy:  0.8355
        Precision    Recall    Fscore
Female   0.920914  0.754500  0.829442
Male     0.769906  0.926889  0.841135
```

## ASR
## Environments 
**Using forked version of ESPnet with damped weights sharing functions.**

- date: `Wed Jan 15 12:08:01 CET 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.4`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `e53e1718d0943b9efeeec45ce8ae0e15367f2fdf`
  - Commit date: `Wed Jan 15 12:07:51 2020 +0100`

**ESPnet was modified according to this [diff](https://github.com/espnet/espnet/compare/e88a477cb72be7e5a03595ead5c233f8d211f6b6...deep-privacy:e53e1718d0943b9efeeec45ce8ae0e15367f2fdf)**

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dev_clean_model.acc.best_decode_lm|2703|54402|96.2|3.4|0.4|0.5|4.3|43.6|
|decode_dev_other_model.acc.best_decode_lm|2864|50948|89.0|9.8|1.2|1.8|12.8|68.1|
|decode_test_clean_model.acc.best_decode_lm|2620|52576|96.1|3.5|0.4|0.6|4.4|44.0|
|decode_test_other_model.acc.best_decode_lm|2939|52343|87.9|10.7|1.3|1.8|13.9|72.1|

Those scores are very close to the one reported by the ESPnet team: [source](https://github.com/espnet/espnet/blob/3b83007b43b79c7c0730f45b06783bd478ce87e7/egs/librispeech/asr1/RESULTS.md#pytorch-vgg-3blstm-1024-units-bpe-5000-latest-rnnlm-training-with-tuned-decoding-ctc_weight05-lm_weight07-dropout-02)
