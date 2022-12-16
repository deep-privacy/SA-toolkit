# python3 -m torch.distributed.launch --nproc_per_node 3 local/tuning/hifi_gan.py
for dim in 16 32 48 64 128 256 512 1024
do
ngpu=$(python3 -c "import torch; print(torch.cuda.device_count())")
  python3 -m torch.distributed.launch --nproc_per_node $ngpu local/tuning/hifi_gan.py \
    --checkpoint_path exp/hifigan_vq_${dim}_finetuned \
    --asr_tdnnf_model local/chain/e2e/tuning/tdnnf_vq_bd.py \
    --asr_tdnnf_exp_path exp/chain/e2e_tdnnf_vq_${dim}/ \
    --asr_tdnnf_vq ${dim} \
    --training_epochs 300 \
    --cold_restart  \
    --init_weight_model ./exp/hifigan/g_best
  exit 0

done
exit 0

# 16 Done
# 32 Done
# 48 Done
# 64 Done
# 128 Done
# 256 Done
# 512 Done
# 1024 Done
