Hifi-GAN Synthesis from Vector quantized features
===


## Data
1. Download MLS dataset from [here](https://www.openslr.org/94/) into ```data/mls``` folder.
2. Preprocess data with padding

```bash
python ./local/preprocess.py \
--srcdir data/mls/train \
--outdir data/mls/wavs \
--pad
```

## Training

### Train HifiGAN model

```
python3 local/get_f0_stats_hifi_gan_w2w2.py \
    --srcdir ./data/mls/wavs/
    --outstats ./data/mls/stats.json

python3 -m torch.distributed.launch --nproc_per_node 2 ./local/tuning/hifi_gan_wav2vec2.py \
    --batch_size 40 \
    --bn_dataset mls \
    --data_dir data/mls/wavs/ \
    --f0_stats data/mls/stats.json
```

# DP models
```bash

### Train HifiGAN model TDNNF bn + onehot
```bash
python -m torch.distributed.launch --nproc_per_node 2 local/tuning/hifi_gan_tdnnf.py --batch_size 40 --no-caching


python3 -m torch.distributed.launch --nproc_per_node 2 \
  local/tuning/hifi_gan_tdnnf.py \
  --batch_size 40 \
  --no-caching \
  --asrbn_tdnnf_vq 128 \
  --checkpoint_path exp/hifigan_vq_128 \
  --asrbn_tdnnf_model local/chain/e2e/tuning/tdnnf_vq_bd.py \
  --asrbn_tdnnf_exp_path exp/chain/e2e_tdnnf_vq_128/ \
  --cold_restart  \
  --init_weight_model exp/hifigan_tdnnf/g_00111000

python3 -m torch.distributed.launch --nproc_per_node 2 \
  local/tuning/hifi_gan_tdnnf.py \
  --batch_size 40 \
  --no-caching \
  --asrbn_tdnnf_vq 256 \
  --checkpoint_path exp/hifigan_vq_256 \
  --asrbn_tdnnf_model local/chain/e2e/tuning/tdnnf_vq_bd.py \
  --asrbn_tdnnf_exp_path exp/chain/e2e_tdnnf_vq_256/ \
  --cold_restart  \
  --init_weight_model exp/hifigan_tdnnf/g_00111000


python3 -m torch.distributed.launch --nproc_per_node 2 \
  local/tuning/hifi_gan_tdnnf.py \
  --batch_size 40 \
  --no-caching \
  --asrbn_tdnnf_vq 128 \
  --checkpoint_path exp/hifigan_vq_128 \
  --asrbn_tdnnf_model local/chain/e2e/tuning/tdnnf_vq_bd.py \
  --asrbn_tdnnf_exp_path exp/chain/e2e_tdnnf_vq_128/ \
  --init_weight_model exp/hifigan_vq_128/g_00042000; sleep 60; python3 -m torch.distributed.launch --nproc_per_node 2 \
  local/tuning/hifi_gan_tdnnf.py \
  --batch_size 40 \
  --no-caching \
  --asrbn_tdnnf_vq 256 \
  --checkpoint_path exp/hifigan_vq_256 \
  --asrbn_tdnnf_model local/chain/e2e/tuning/tdnnf_vq_bd.py \
  --asrbn_tdnnf_exp_path exp/chain/e2e_tdnnf_vq_256/ \
  --init_weight_model exp/hifigan_vq_256/g_00042000
```


### Train HifiGAN model
```bash
ngpu=$(python3 -c "import torch; print(torch.cuda.device_count())")
python -m torch.distributed.launch --nproc_per_node $ngpu local/tuning/hifi_gan_wav2vec2.py
```


## Convert
```
# Create spk2target mapping
python3 create_random_target.py \
    --target-list data/mls/stats.json \
    --in-wavscp ../asr-bn/mls/data/mls_french/test_kaldi/wav.scp \
    --in-utt2spk ../asr-bn/mls/data/mls_french/test_kaldi/utt2spk --same-spk "10065" \
    > target-mapping

python3 ./convert.py \
    --model-type wav2vec2_mls \
    --num-workers 4 \
    --batch-size 64  \
    --out generated_mls_test_137000 \
    --in-wavscp /home/test/wav.scp \
    --f0-stats "$(cat data/mls/stats.json)" \
    --target_id target-mapping
