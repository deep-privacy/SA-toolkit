#!/bin/bash


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# WARN: Don't forget to have the right DAMPED_N_DOMAIN venv in ESPnet (run.sh)
# branches type
branches=( "gender" "spk_identif" "spk_identif" )
# input dimensions of each above branches
branches_eproj=( 1024 1024 1024 )
# branches task rank of each above branches
# Carefully crafted value also defined in ESPnet
branches_rank=( 1 2 3 )
# on which GPUs to run branches
branches_gpu=( 2 3 1 )
branches_conf_args=( "" "--grad-reverse true" "" )

world_size=$((${#branches[@]} + 1))
master_ip="0.0.0.0" # address of the tool that was damped.disturb-ed
# master_ip="172.16.64.9"

stage=2
stop_stage=100

# misc
log_interval=1200

. gender/utils/parse_options.sh || exit 1; # ! Using gender utils !

NC=`tput sgr0`
pids=() # initialize pids

for (( i = 0; i < ${#branches[@]}; i++ )); do
  (
    cd ./${branches[i]}/
    ./run.sh \
      --stage $stage \
      --stop-stage $stop_stage \
      --log-interval $log_interval \
      --eproj ${branches_eproj[i]}   \
      --gpu-device ${branches_gpu[i]} \
      --world-size $world_size \
      --master-ip $master_ip \
      --task-rank ${branches_rank[i]} \
      ${branches_conf_args[i]} \
      | sed "s/^/`tput setaf $i`[${branches_rank[i]} - ${branches[i]}]${NC} /"
  ) &
  pids+=($!) # store background pids
done

i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
[ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
