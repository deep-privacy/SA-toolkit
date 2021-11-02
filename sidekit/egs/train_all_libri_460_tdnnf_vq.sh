#!/bin/bash

# frontend
if [ "$(hostname)" == "lst" ]; then
    for dim in 16 32 48 64 128 256 384 512 768
    do
        # sbatch -p gpu -c 20 --gres gpu:rtx8000:2 -N 1 --mem 32G --constraint=noexcl --time 25:30:00 -o .log_parrallel_xvec_tdnnf_vq_$dim.out --job-name xvec_tdnnf_vq_$dim --wrap="bash ./train_all_libri_460_tdnnf_vq.sh $dim"
        sbatch -p non-k40 -c 49 --gres gpu:2 -C "gpu24gb|gpu48gb" -N 1 --mem 24G --constraint=noexcl --time 35:30:00 -o .log_parrallel_xvec_tdnnf_vq_$dim.out --job-name xvec_tdnnf_vq_$dim --wrap="bash ./train_all_libri_460_tdnnf_vq.sh $dim"
    done
    exit 0
fi

dim="$1"
shift
echo "Xvec TDNNF_VQ Run for dim: $dim"

cd ~/lab/asr-based-privacy-preserving-separation
. ./env.sh
cd sidekit/egs

cd libri460_fast_vq

(
flock -x -w 20 200 || exit 1
sed -i -E "/^export\spkwrap_exp_dir|export\spkwrap_vq_dim|trainingcfg/s/[0-9]{2,}/$dim/" train.sh
sed -i -E "/^log_file|tmp_model_name|best_model_name/s/[0-9]{2,}/$dim/" cfg/training.yaml
cat train.sh
cp train.sh train.sh_$dim
cp cfg/training.yaml cfg/training_$dim.yaml
) 200>.parallel.exclusivelock
./train.sh_$dim

# sqg | grep pchamp | awk '{print $1}' | xargs scancel; \rm .log_parrallel*
