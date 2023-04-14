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

* (i) the contact person;
* (ii) affiliation;
* (iii) country;
* (iv) status (academic/nonacademic).
