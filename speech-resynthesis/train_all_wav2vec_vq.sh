#!/bin/bash

# frontend
if [ "$(hostname)" == "lst" ]; then

    for dim in -1 64 32 48 128 16
    do
        sbatch -p non-k40 -c 42 --gres gpu:3 -C "gpu48gb" --exclude=spk1,raid02 -N 1 --mem 130G --constraint=noexcl --time 80:30:00 -o .log_parrallel_hifi_wav2vec_vq_$dim.out --job-name hifi_wav2vec_vq$dim --wrap="bash ./train_all_wav2vec_vq.sh $dim"
    done
    sleep 5

    for dim in 256 512 1024
    do
        sbatch -p non-k40 -c 42 --gres gpu:2 -C "gpu48gb" --exclude=spk1,raid02 -N 1 --mem 130G --constraint=noexcl --time 80:30:00 -o .log_parrallel_hifi_wav2vec_vq_$dim.out --job-name hifi_wav2vec_vq$dim --wrap="bash ./train_all_wav2vec_vq.sh $dim"
    done
    exit 0
fi

dim="$1"
shift
echo "Run for dim: $dim"

cd ~/lab/asr-based-privacy-preserving-separation
. ./env.sh
cd speech-resynthesis

ngpu=$(python3 -c "import torch; print(torch.cuda.device_count())")
mibgpu=$(python3 -c "import torch; print(torch.cuda.get_device_properties(0).total_memory // 1024 ** 2)")

batch_size=$(python - << EOF
if $ngpu > 2 and $mibgpu>40000: print(240)
if $ngpu == 2 and $mibgpu>40000: print(150)
if $ngpu > 2 and $mibgpu<40000: print(96)
if $ngpu == 2 and $mibgpu<40000: print(74)
EOF
)


if [ "$dim" == "-1" ]; then
    env pkwrap_model="local/chain/e2e/tuning/tdnnf_wav2vec.py" \
        pkwrap_exp_dir="exp/chain/e2e_tdnnf_wav2vec/" \
        pkwrap_vq_dim="-1" \
        python -m torch.distributed.launch --nproc_per_node $ngpu train.py \
        --checkpoint_path checkpoints/lj_vq_tdnnf_asr_wav2vec \
        --config configs/LJSpeech/vq_tdnnf_asr_wav2vec.json --batch_size $batch_size

    exit 0
fi

env pkwrap_model="local/chain/e2e/tuning/tdnnf_wav2vec_vq.py" \
    pkwrap_exp_dir="exp/chain/e2e_tdnnf_wav2vec_vq_$dim/" \
    pkwrap_vq_dim="$dim" \
    python -m torch.distributed.launch --nproc_per_node $ngpu train.py \
    --checkpoint_path checkpoints/lj_vq_tdnnf_asr_wav2vec_vq_$dim \
    --config configs/LJSpeech/vq_tdnnf_asr_wav2vec.json --batch_size $batch_size


# sqg | grep pchamp | awk '{print $1}' | xargs scancel; \rm .log_parrallel*

# dim=8
# env pkwrap_model="local/chain/e2e/tuning/tdnnf_vq_bd.py" \
    # pkwrap_exp_dir="exp/chain/e2e_tdnnf_vq_bd_sizeco_$dim/" \
    # pkwrap_vq_dim="$dim" \
    # python -m torch.distributed.launch --nproc_per_node $ngpu train.py \
    # --checkpoint_path checkpoints/lj_vq_tdnnf_asr_vq_$dim \
    # --config configs/LJSpeech/vq_tdnnf_asr.json
