#!/bin/bash


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# study only the first layer of vggblstm
elayers=0
world_size=$((2 + 1))
# Out VGG2L -> 2688 D (in p/rnn/encoders.py)
# Out LSTM_l3-tanh-fc -> 1024 D (in p/2e_asr.py)
eprojs=( 2688 1024 )
master_ip="0.0.0.0" # address of the tool that was damped.disturb-ed
# master_ip="172.16.64.9"

stage=2
stop_stage=100

# misc
log_interval=1200

. gender/utils/parse_options.sh || exit 1; # ! using gender utils !

pids=() # initialize pids

# !WARNN carefully crafted for vggblstm with 3 encoder layers
# same offsets are also defined in espnet/nets/chainer_backend/rnn/encoders.py
offset_gender=3
offset_spk=4

gpu_device=0

# Gender
for (( i = 0; i < $elayers; i++ )); do
  (
    cd ./gender/
    ./run.sh \
      --stage $stage \
      --stop-stage $stop_stage \
      --log-interval $log_interval \
      --eproj ${eprojs[i]}   \
      --gpu-device $gpu_device \
      --world-size $world_size \
      --master-ip $master_ip \
      --task-rank $((offset_gender + i)) \
    | sed "s/^/[$((offset_gender + i)) - gender] /"
  ) &
  pids+=($!) # store background pids
done

gpu_device=1

# spk_identif
for (( i = 0; i < $elayers; i++ )); do
  (
    cd ./spk_identif/
    ./run.sh \
      --stage $stage \
      --stop-stage $stop_stage \
      --log-interval $log_interval \
      --eproj ${eprojs[i]}   \
      --gpu-device $gpu_device \
      --world-size $world_size \
      --master-ip $master_ip \
      --task-rank $((offset_spk + i)) \
    | sed "s/^/[$((offset_spk + i)) - spk_identif] /"
  ) &
  pids+=($!) # store background pids
done

# The two last layers are placed outside of the encoder (rank 1->gen and 2->spk)
# They share the same input features as the output of the last layer of the encoder.

gpu_device=2

(
  cd ./gender/
  ./run.sh \
    --stage $stage \
    --stop-stage $stop_stage \
    --log-interval $log_interval \
    --eproj ${eprojs[-1]}   \
    --gpu-device $gpu_device \
    --world-size $world_size \
    --master-ip $master_ip \
    --task-rank 1 \
    | sed "s/^/[1 - gender] /"
) &
pids+=($!) # store background pids

gpu_device=3

(
  cd ./spk_identif/
  ./run.sh \
    --stage $stage \
    --stop-stage $stop_stage \
    --log-interval $log_interval \
    --eproj ${eprojs[-1]}   \
    --gpu-device $gpu_device \
    --world-size $world_size \
    --master-ip $master_ip \
    --task-rank 2 \
    | sed "s/^/[2 - spk_identif] /"
) &

i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
[ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
