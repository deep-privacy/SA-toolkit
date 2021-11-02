#!/bin/bash

export NUM_NODES=1
export NUM_GPUS_PER_NODE=2
export NODE_RANK=0
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))

export pkwrap_model="local/chain/e2e/tuning/tdnnf_vq.py"
export pkwrap_exp_dir="exp/chain/e2e_tdnnf_vq_sizeco_256/"
export pkwrap_bn_dim="256"
export pkwrap_vq_dim="256"
trainingcfg="cfg/training_256.yaml"

# env DEV4S=True WORLD_SIZE=1 ipython3 train_xtractor.py # DEV

python3 -m torch.distributed.launch \
       --nproc_per_node=$NUM_GPUS_PER_NODE \
       --nnodes=$NUM_NODES \
       --node_rank $NODE_RANK \
       train_xtractor.py --training $trainingcfg
