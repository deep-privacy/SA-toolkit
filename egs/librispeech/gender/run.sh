#!/bin/bash

. ./path.sh || exit 1;

# configuration
stage=2
stop_stage=100

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "stage 0: Meta-Data Download"
  bash local/download_and_untar.sh
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "stage 1: Data preparation"
  bash local/data_prep.sh
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "stage 2: Training the '$(basename `pwd`)' branch"
  trainer.py \
    --config $(pwd)/conf/conf.py
fi
