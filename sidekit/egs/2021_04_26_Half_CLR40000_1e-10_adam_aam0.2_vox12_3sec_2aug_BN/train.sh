#!/bin/bash

export NUM_NODES=1
export NUM_GPUS_PER_NODE=2
export NODE_RANK=0
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))

# export DEV4S=True; ipython3 train_xtractor.py # DEV

env CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch \
       --nproc_per_node=$NUM_GPUS_PER_NODE \
       --nnodes=$NUM_NODES \
       --node_rank $NODE_RANK \
       train_xtractor.py
