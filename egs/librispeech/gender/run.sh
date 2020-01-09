#!/bin/bash

. ./path.sh || exit 1;

# configuration
stage=2 # start at stabe 0 to Download Meta-Data
stop_stage=100

# misc
log_interval=5

# model parameters
conf=conf.py

# eval
snapshot=$conf/snapshot.ep.35
label="0 1"
label_name="Female Male"

# The 'to_rank' value to provide to damped.disturb.DomainTask
task_rank=1

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
    --config $(pwd)/conf/$conf \
    --task-rank $task_rank \
    --label $label \
    --label-name $label_name \
    --log-interval $log_interval
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "stage 3: Evaluating the '$(basename `pwd`)' branch"
  evaluator.py \
    --config $(pwd)/conf/$conf \
    --task-rank $task_rank \
    --label $label \
    --label-name $label_name \
    --snapshot $(pwd)/exp/$snapshot
fi
