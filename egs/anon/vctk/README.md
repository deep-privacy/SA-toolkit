Voice Privacy vctk eval
===

To run the recipe:

```bash
# Activate your miniconda env
. ./path.sh

# prepare data (You will need a password which is provided by registering)
./local/data_prep_vpc.sh

# eval privacy/utility of the clear signals
./local/eval.py --config configs/eval_clear
```

## Voice privacy [registration](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2022#general-information)

To get access to evaluation datasets and models, please send an email to organisers@lists.voiceprivacychallenge.org with “VoicePrivacy-2022 registration" as the subject line.  
The mail body should include:
```markdown
* (i) the contact person;
* (ii) affiliation;
* (iii) country;
* (iv) status (academic/nonacademic).
```

## Results clear speech (non-anonymized)

```cfg
# Config
./local/eval.py --config configs/eval_clear

# Models:
asr_egs = ../../asr/librispeech
asr_exp = asr_eval_tdnnf_360h

asv_egs = ../../asv/voxceleb
asv_exp = asv_eval_vox1_resnet

# Result:
satools INFO: Printing best WER without rescoring exp/eval_clear/asr_decode_vctk_test_asr_eval_tdnnf_360h_final_iter...
satools INFO:  %WER 26.92 [ 25018 / 92922, 7415 ins, 3873 del, 13730 sub ]
satools INFO: Printing best WER with rescoring exp/eval_clear/asr_decode_vctk_test_asr_eval_tdnnf_360h_final_iter_fg...
satools INFO:  %WER 21.97 [ 20418 / 92922, 6835 ins, 2739 del, 10844 sub ]
satools INFO: Printing ASV metrics...
satools INFO:  %EER: 1.14 ±  0.205, Min Cllr: 0.045, linkability: 0.971
satools INFO: Printing ASV AS-NORM metrics...
satools INFO:  %EER: 1.049 ± 25.348, Min Cllr: 0.03, linkability: 0.981
```
