#!/bin/bash


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

elayers=3
# Out VGG2L -> 2688 D (in p/rnn/encoders.py)
# Out LSTM_l1 -> 50 D
# Out LSTM_l2 -> 50 D
# Out LSTM_l3 -> 50 D (in p/2e_asr.py)
eprojs=( 2688 50 50 50 )
world_size=9
stage=2
stop_stage=100

. gender/utils/parse_options.sh || exit 1; # ! using gender utils !

pids=() # initialize pids

# !WARNN carefully crafted for vggblstm with 3 encoder layers
# same offsets are also defined in espnet/nets/chainer_backend/rnn/encoders.py
offset_gender=3
offset_spk=6

# Gender
for (( i = 0; i < $elayers; i++ )); do
  (
    cd ./gender/
    ./run.sh \
      --stage $stage \
      --stop-stage $stop_stage \
      --eproj ${eprojs[i]}   \
      --log-interval 1 \
      --world-size $world_size \
      --task-rank $((offset_gender + i)) \
    | sed "s/^/[$((offset_gender + i)) - gender] /"
  ) &
  pids+=($!) # store background pids
done


# spk_identif
for (( i = 0; i < $elayers; i++ )); do
  (
    cd ./spk_identif/
    ./run.sh \
      --stage $stage \
      --stop-stage $stop_stage \
      --eproj ${eprojs[i]}   \
      --log-interval 1 \
      --world-size $world_size \
      --task-rank $((offset_spk + i)) \
    | sed "s/^/[$((offset_spk + i)) - spk_identif] /"
  ) &
  pids+=($!) # store background pids
done

# The two last layers are placed outside of the encoder (rank 1->gen and 2->spk)
# They share the same input features as the output of the last layer of the encoder.

(
  cd ./gender/
  ./run.sh \
    --stage $stage \
    --stop-stage $stop_stage \
    --eproj ${eprojs[-1]}   \
    --log-interval 1 \
    --world-size $world_size \
    --task-rank 1 \
    | sed "s/^/[1 - gender] /"
) &
pids+=($!) # store background pids

(
  cd ./spk_identif/
  ./run.sh \
    --stage $stage \
    --stop-stage $stop_stage \
    --eproj ${eprojs[-1]}   \
    --log-interval 1 \
    --world-size $world_size \
    --task-rank 2 \
    | sed "s/^/[2 - spk_identif] /"
) &

i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
[ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
