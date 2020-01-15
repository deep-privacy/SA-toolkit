#!/bin/bash

. ./path.sh || exit 1;

# configuration
stage=2 # start at stabe 0 to Download Meta-Data
stop_stage=100

# model/trainer conf
conf=conf.py
resume=
# resume=$(pwd)/exp/$conf/GenderNet.best.acc.ckpt

# misc
log_interval=300
log_path=$(pwd)/exp/$conf

# eval
snapshot=$(pwd)/exp/$conf/GenderNet.best.acc.ckpt

# task related
label="0 1"
label_name="Female Male"

# The 'to_rank' value to provide to damped.disturb.DomainTask
task_rank=1
gpu_device=1

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
  mkdir -p $log_path
  trainer.py \
    --config $(pwd)/conf/$conf \
    --task-rank $task_rank \
    --log-interval $log_interval \
    --gpu-device $gpu_device \
    --resume $resume \
    | tee $log_path/train.log
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "stage 3: Evaluating the '$(basename `pwd`)' branch"
  evaluator.py \
    --config $(pwd)/conf/$conf \
    --task-rank $task_rank \
    --label $label \
    --label-name $label_name \
    --snapshot $snapshot \
    --gpu-device $gpu_device \
    | tee -a $log_path/eval.log
fi
