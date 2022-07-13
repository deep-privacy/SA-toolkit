#!/usr/bin/env bash

set -e

source configs/local.conf

cd ../../../../sidekit/egs/librispeech

# Create Librispeech train data
python3 ./local/dataprep_librispeech.py --from $corpus --filter-dir train-clean-100 --make-train-csv --out-csv list/libri_train.csv # set --filter-dir if you want to select specific part of Librispeech (default is train-clean-360)

# Create Librispeech test files in kaldi format
python3 ./local/dataprep_kaldi_test_datasets.py --save-path ./data/asv_test_libri --from $corpus

create_sidekit_test_files.py --enrolls ./data/asv_test_libri/enrolls_test --trials ./data/asv_test_libri/trials_test --utt2spk ./data/asv_test_libri/enrolls_utt2spk --out-dir ./list/asv_test_libri --out-file-prefix libri_test

cd -

echo "-- DONE --"

grep "^^[0-9]*\s+\|.*$1" -E  $corpus/SPEAKERS.TXT | grep train-clean-100 | awk -F'|' '{gsub(/[ ]+/, ""); print $1, NR-1}' > ./data/spk2id
