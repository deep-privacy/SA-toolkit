#!/bin/bash

_all_args="$@"

. ./path.sh || exit 1;

# Trainer/Evaluator global conf #
####

# exp tag
tag="spk_iden_reco_lstm_eproj_no_back"

# configuration
stage=2 # start at stabe 0 to Download Meta-Data
stop_stage=100

# model/trainer conf (net archi, task related)
conf=conf.py
resume=
resume=BrijSpeakerXvector.best.acc.ckpt

# misc
log_interval=300

# eval
snapshot=BrijSpeakerXvector.best.acc.ckpt

# task related
label="$(cat data/spk2id | awk '{ print $2 }' | tr '\n' ' ')"
label_name="$label"

# The 'task_rank' value to provide to damped.disturb.DomainTask
task_rank=2
gpu_device=1
world_size=2 # MUST be incremented to match the number of damped.disturb.DomainTask + master!
master_ip="0.0.0.0" # address of the tool that was damped.disturb-ed

# undefined (not defined above) arguments string are placed in the $other
# variable without evaluation. ('--not-def 50 --not-def-2 100' will be placed
# in $other).
# $other is used to bypass trainer/evaluator and is accecible in the 'conf' python file
. utils/parse_options.sh || exit 1;

# Model CONF
####
# The model hyperparameters can be modified using the $other
# variable. Only available through the command line parameters.

tag+="__"
tag+="rank=$task_rank"
tag+="_"
tag+=$(echo "$other" | sed "s/--\(\S*\)\s/\1=/g"  | sed "s/\s/_/g")

log_path=$(pwd)/exp/$tag

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

_date=$(date +"%y-%d-%m %Hh")


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "stage 0: Meta-Data Download"
  bash local/download_and_untar.sh
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "stage 1: Data preparation"
  # Only on 'train' subset!
  bash local/data_prep.sh
  exit 1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "stage 2: Training the '$(basename `pwd`)' branch"
  mkdir -p $log_path

  echo -e "-------\nNew run: $_date\n" >> $log_path/train.cmd
  echo -e "-------\nNew run: $_date\n" >> $log_path/train.log
  echo "$_all_args" >> $log_path/train.cmd

  if [[ -n "$resume" ]]; then
    resume=$log_path/$resume
  fi

  trainer.py \
    --config $(pwd)/conf/$conf \
    --exp-path $log_path \
    --task-rank $task_rank \
    --log-interval $log_interval \
    --gpu-device $gpu_device \
    --world-size $world_size \
    --master-ip $master_ip \
    --resume $resume \
    $other \
    | tee -a $log_path/train.log
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "stage 3: Evaluating the '$(basename `pwd`)' branch"

  echo -e "-------\nNew run: $_date\n" >> $log_path/eval.cmd
  echo -e "-------\nNew run: $_date\n" >> $log_path/eval.log
  echo "$_all_args" >> $log_path/eval.cmd

  snapshot="$log_path/$snapshot"

  evaluator.py \
    --config $(pwd)/conf/$conf \
    --task-rank $task_rank \
    --label $label \
    --label-name $label_name \
    --snapshot $snapshot \
    --gpu-device $gpu_device \
    --world-size $world_size \
    --master-ip $master_ip \
    $other \
    | tee -a $log_path/eval.log
fi
