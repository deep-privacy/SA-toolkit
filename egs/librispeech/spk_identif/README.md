# SPK identif

The speakers in the original LibriSpeech train/dev/test splits are disjoint.

The original train-960 subset is split into three sub-subset to train and
evaluate speaker identification on all possible speakers.
The evaluation of the automatic speech recognition system is performed using the
original and unchanged test-clean/test-other.

To create this split in ESPnet, `python splitjson_spk.py --dev 2 --test 2 ./dump/train_960/deltafalse/data_unigram5000.json` is used (run in to root of ESPnet recipe).
