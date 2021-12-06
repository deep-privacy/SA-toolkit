#!/bin/bash

# frontend
if [ "$(hostname)" == "lst" ]; then

    for dim in 16 128 48 256 768 64 32
    do
        sbatch -p non-k40 -c 42 --gres gpu:3 -C "gpu48gb" -N 1 --mem 130G --constraint=noexcl --time 80:30:00 -o .log_parrallel_hifi_vq_$dim.out --job-name asr_hifi_vq_$dim --wrap="bash ./train_all_vq.sh $dim"
    done

    dim=-1
    sbatch -p non-k40 -c 42 --gres gpu:2 -w trad10 -C "gpu48gb" -N 1 --mem 130G --constraint=noexcl --time 80:30:00 -o .log_parrallel_hifi_vq_$dim.out --job-name asr_hifi_vq_$dim --wrap="bash ./train_all_vq.sh $dim"
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
if $ngpu >= 2 and $mibgpu>40000: print(128)
if $ngpu == 2 and $mibgpu<40000: print(64)
if $ngpu > 2 and $mibgpu<40000: print(128)

EOF
)


if [ "$dim" == "-1" ]; then
    python -m torch.distributed.launch --nproc_per_node $ngpu train.py \
    --checkpoint_path checkpoints/lj_vq_tdnnf_asr \
    --batch_size 96 \
    --config configs/LJSpeech/vq_tdnnf_asr.json
    exit 0
fi

env pkwrap_model="local/chain/e2e/tuning/tdnnf_vq.py" \
    pkwrap_exp_dir="exp/chain/e2e_tdnnf_vq_sizeco_$dim/" \
    pkwrap_vq_dim="$dim" \
    python -m torch.distributed.launch --nproc_per_node $ngpu train.py \
    --checkpoint_path checkpoints/lj_vq_tdnnf_asr_vq_$dim \
    --config configs/LJSpeech/vq_tdnnf_asr.json

# sqg | grep pchamp | awk '{print $1}' | xargs scancel; \rm .log_parrallel*

# dim=8
# env pkwrap_model="local/chain/e2e/tuning/tdnnf_vq_bd.py" \
    # pkwrap_exp_dir="exp/chain/e2e_tdnnf_vq_bd_sizeco_$dim/" \
    # pkwrap_vq_dim="$dim" \
    # python -m torch.distributed.launch --nproc_per_node $ngpu train.py \
    # --checkpoint_path checkpoints/lj_vq_tdnnf_asr_vq_$dim \
    # --config configs/LJSpeech/vq_tdnnf_asr.json
