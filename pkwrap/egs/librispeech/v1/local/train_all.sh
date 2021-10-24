#!/bin/bash

# frontend
if [ "$(hostname)" == "lst" ]; then
    for dim in 16 32 256 384 512 768
    do
        sbatch -p gpu -c 40 --gres gpu:rtx8000:3 -N 1 --mem 128G --constraint=noexcl --time 7:30:00 -o .log_parrallel_tdnnf_vq_$dim.out --job-name tdnnf_vq_$dim --wrap="bash .parallel.sh $dim"
    done
    for dim in 48 64 128
    do
        sbatch -p gpu -c 40 --gres gpu:rtx6000:3 -N 1 --mem 128G --constraint=noexcl --time 7:30:00 -o .log_parrallel_tdnnf_vq_$dim.out --job-name tdnnf_vq_$dim --wrap="bash .parallel.sh $dim"
    done

    exit 0
fi

dim="$1"
shift
echo "TDNNF_VQ{,spkdiff} Run for dim: $dim"

cd ~/lab/asr-based-privacy-preserving-separation
. ./env.sh
cd pkwrap/egs/librispeech/v1/
. ./path.sh

dset="data\/dev_clean_fbank_hires"
(
# Wait for lock on /var/lock/.myscript.exclusivelock (fd 200) for 20 seconds
flock -x -w 20 200 || exit 1
sed -i -E "/^test_set.*$/s/test_set\s=.*$/test_set\ =\ $dset/" configs/tdnnf_e2e_vq
sed -i -E "/^dirname|^model_args/s/[0-9]{2,}/$dim/" configs/tdnnf_e2e_vq
cat configs/tdnnf_e2e_vq  | grep "^dirname"
cp configs/tdnnf_e2e_vq configs/tdnnf_e2e_vq_$dim
) 200>.parallel.exclusivelock
local/chain/train.py --stage 4 --conf configs/tdnnf_e2e_vq_$dim
(
flock -x -w 20 200 || exit 1
sed -i -E "/^test_set.*$/s/test_set\s=.*$/test_set\ =\ $dset/" configs/tdnnf_e2e_vq_spkdelta
sed -i -E "/^dirname|^model_args|^init_weight_model/s/[0-9]{2,}/$dim/" configs/tdnnf_e2e_vq_spkdelta
cat configs/tdnnf_e2e_vq_spkdelta  | grep "^dirname"
cp configs/tdnnf_e2e_vq_spkdelta configs/tdnnf_e2e_vq_spkdelta_$dim
) 200>.parallel.exclusivelock
local/chain/train.py --stage 4 --conf configs/tdnnf_e2e_vq_spkdelta_$dim

for dset in "data\/test_clean_fbank_hires" "data\/test_other_fbank_hires"
do
    (
    flock -x -w 20 200 || exit 1
    sed -i -E "/^test_set.*$/s/test_set\s=.*$/test_set\ =\ $dset/" configs/tdnnf_e2e_vq
    sed -i -E "/^dirname|^model_args/s/[0-9]{2,}/$dim/" configs/tdnnf_e2e_vq
    cat configs/tdnnf_e2e_vq  | grep "^dirname"
    cp configs/tdnnf_e2e_vq configs/tdnnf_e2e_vq_$dim
    ) 200>.parallel.exclusivelock
    local/chain/train.py --stage 7 --conf configs/tdnnf_e2e_vq_$dim

    (
    flock -x -w 20 200 || exit 1
    sed -i -E "/^test_set.*$/s/test_set\s=.*$/test_set\ =\ $dset/" configs/tdnnf_e2e_vq_spkdelta
    sed -i -E "/^dirname|^model_args|^init_weight_model/s/[0-9]{2,}/$dim/" configs/tdnnf_e2e_vq_spkdelta
    cat configs/tdnnf_e2e_vq_spkdelta  | grep "^dirname"
    cp configs/tdnnf_e2e_vq_spkdelta configs/tdnnf_e2e_vq_spkdelta_$dim
    ) 200>.parallel.exclusivelock
    local/chain/train.py --stage 7 --conf configs/tdnnf_e2e_vq_spkdelta_$dim
done

# sqg | grep pchamp | awk '{print $1}' | xargs scancel; \rm .log_parrallel*
