#!/bin/bash

. ./configs/local.conf


for data in train-clean-100 dev-clean; do
  echo "Audio resample $data"
  python3 ./local/preprocess.py \
    --srcdir $corpus/$data/ \
    --outdir data/${data}_wavs_16khz --pad
done
