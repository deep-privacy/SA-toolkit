#!/bin/bash
#SBATCH -N1
#SBATCH -c 30
#SBATCH --mem=80G
#SBATCH -J half_BN_resnet_ASR_BN
#SBATCH -p gpu
#SBATCH --gres gpu:rtx8000:3
#SBATCH --constraint=noexcl
#SBATCH --time 4-00

cd ../../..
. ./env.sh
cd -
bash train.sh
