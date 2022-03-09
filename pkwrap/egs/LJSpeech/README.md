Hifi-GAN Synthesis from Vector quantized features
===


### Data

#### For LJSpeech:
1. Download LJSpeech dataset from [here](https://keithito.com/LJ-Speech-Dataset/) into ```data/LJSpeech-1.1``` folder.
2. Downsample audio from 22.05 kHz to 16 kHz and pad

```bash
python ./local/preprocess.py \
--srcdir data/LJSpeech-1.1/wavs \
--outdir data/LJSpeech-1.1/wavs_16khz \
--pad
```

## Training

#### F0 Quantizer Model
To train F0 quantizer model, use the following command:
```bash
ngpu=$(python3 -c "import torch; print(torch.cuda.device_count())")
python -m torch.distributed.launch --nproc_per_node $ngpu local/tuning/f0_quant.py \
--checkpoint_path checkpoints/lj_f0_vq
```

### Train HifiGAN model
```bash
ngpu=$(python3 -c "import torch; print(torch.cuda.device_count())")
python -m torch.distributed.launch --nproc_per_node $ngpu local/tuning/hifi_gan.py
```

### VQ models
```bash
tail e2e_tdnnf_vq_*/decode_test_clean*final_fg/scoringDetails/best_wer | grep -E "*e2e_tdnnf_vq_[0-9]{2,}" | grep "WER" | awk '{print $1 "\t"$2 "\t" $14}' | cut -d/ -f1,3 | sort -k4,4 -n -t"_"
# %WER    15.90   exp/e2e_tdnnf_vq_16
# %WER    9.81    exp/e2e_tdnnf_vq_32
# %WER    8.68    exp/e2e_tdnnf_vq_48
# %WER    8.50    exp/e2e_tdnnf_vq_64
# %WER    8.04    exp/e2e_tdnnf_vq_128
# %WER    7.63    exp/e2e_tdnnf_vq_256
# %WER    7.60    exp/e2e_tdnnf_vq_512
# %WER    7.16    exp/e2e_tdnnf_vq_1024

python3 -m torch.distributed.launch --nproc_per_node $ngpu local/tuning/hifi_gan.py \
   --checkpoint_path exp/hifigan_vq_${dim}_finetuned \
    --asrbn_tdnnf_model local/chain/e2e/tuning/tdnnf_vq_bd.py \
    --asrbn_tdnnf_exp_path exp/chain/e2e_tdnnf_vq_${dim}/ \
    --asrbn_tdnnf_vq ${dim} \
    --training_epochs 300 \
    --cold_restart  \
    --init_weight_model ./exp/hifigan/g_best
```

### DP models
```bash
tail e2e_tdnnf_dp*/decode_dev_clean*final_fg/scoringDetails/best_wer | grep "WER" | awk '{print $1 "\t"$2 "\t" $14}' | cut -d/ -f1,3 | sort -k5,5 -n -t"e"
# %WER    30.38   exp/e2e_tdnnf_dp_e160000
# %WER    19.38   exp/e2e_tdnnf_dp_e180000
# %WER    8.87    exp/e2e_tdnnf_dp_e200000
# %WER    8.28    exp/e2e_tdnnf_dp_e220000 <- done
# %WER    8.06    exp/e2e_tdnnf_dp_e240000
# %WER    7.14    exp/e2e_tdnnf_dp_e260000
# %WER    6.97    exp/e2e_tdnnf_dp_e280000
# %WER    6.73    exp/e2e_tdnnf_dp_e300000

python3 -m torch.distributed.launch --nproc_per_node $ngpu local/tuning/hifi_gan.py \
    --checkpoint_path exp/hifigan_dp_220000_finetuned \
    --asrbn_tdnnf_model local/chain/e2e/tuning/tdnnf_dp.py \
    --asrbn_tdnnf_exp_path exp/chain/e2e_tdnnf_dp_e220000/ \
    --asrbn_tdnnf_dp 220000 \
    --training_epochs 300 \
    --cold_restart  \
    --init_weight_model ./exp/hifigan/g_best
```

# Wav2vec2 vq
```bash
python3 -m torch.distributed.launch --nproc_per_node $ngpu local/tuning/hifi_gan.py \
    --checkpoint_path exp/hifigan_wav2vec2_highbitrate \
    --asrbn_tdnnf_model local/chain/e2e/tuning/tdnnf_wav2vec_hibitrate_vq.py \
    --asrbn_tdnnf_exp_path exp/chain/e2e_tdnnf_wav2vec_hibitrate_vq_256/ \
    --asrbn_tdnnf_vq 256 \
    --hifigan_upsample_rates "5,4,4,2,2" \
    --batch_size 8 \
    --asrbn_interpol_bitrate 320
```

### convert
```bash
python convert.py

for dim in 16 32 48 64 128 256 512 1024
do
 python3 ./convert.py --num-workers 4 --batch-size 64 --vq-dim $dim --out generated_train-clean-360_vq_$dim --in /lium/home/pchampi/lab/asr-based-privacy-preserving-separation/pkwrap/egs/librispeech/v1/corpora/LibriSpeech/train-clean-360
done
```

### Share
```
zip hifigan_model.zip -r **/hifigan_vq_*/g_00075000
```



#### For LibriTTS:
```
python3 ./local/preprocess.py \
  --srcdir [..]/corpora/LibriTTS/train-clean-100/ \
  --outdir data/LibriTTS/wavs_16khz --pad

python3 local/get_f0_stats_hifi_gan_w2w2_libriTTS.py \
  --srcdir ./data/LibriTTS/wavs_16khz/ 
  --outstats ./data/LibriTTS/stats.json

python3 -m torch.distributed.launch --nproc_per_node 2 \
  ./local/tuning/hifi_gan_wav2vec2.py \
  --batch_size 40 \
  --no-caching

python3 -m torch.distributed.launch --nproc_per_node 2 \
  ./local/tuning/hifi_gan_wav2vec2.py \
  --batch_size 40 \
  --no-caching \
  --asrbn_tdnnf_vq 256 \
  --checkpoint_path exp/hifigan_w2w2_vq_256 \
  --asrbn_tdnnf_model local/chain/e2e/tuning/tdnnf_wav2vec_fairseq_hibitrate_vq.py \
  --asrbn_tdnnf_exp_path exp/chain/e2e_tdnnf_wav2vec_fairseq_hibitrate_vq_256/ \
  --cold_restart  \
  --init_weight_model ./exp/hifigan_w2w2/g_best


python3 -m torch.distributed.launch --nproc_per_node 2 \
  ./local/tuning/hifi_gan_wav2vec2.py \
  --batch_size 40 \
  --no-caching \
  --asrbn_tdnnf_vq 512 \
  --checkpoint_path exp/hifigan_w2w2_vq_512 \
  --asrbn_tdnnf_model local/chain/e2e/tuning/tdnnf_wav2vec_fairseq_hibitrate_vq.py \
  --asrbn_tdnnf_exp_path exp/chain/e2e_tdnnf_wav2vec_fairseq_hibitrate_vq_512/ \
  --cold_restart  \
  --init_weight_model ./exp/hifigan_w2w2/g_best

python3 -m torch.distributed.launch --nproc_per_node 2 \
  ./local/tuning/hifi_gan_wav2vec2.py \
  --batch_size 40 \
  --no-caching \
  --asrbn_tdnnf_vq 512 \
  --checkpoint_path exp/hifigan_w2w2_vq_512 \
  --asrbn_tdnnf_model local/chain/e2e/tuning/tdnnf_wav2vec_fairseq_hibitrate_vq.py \
  --asrbn_tdnnf_exp_path exp/chain/e2e_tdnnf_wav2vec_fairseq_hibitrate_vq_512/ \
  --init_weight_model exp/hifigan_w2w2_vq_512/g_00038000


python3 -m torch.distributed.launch --nproc_per_node 2 \
  ./local/tuning/hifi_gan_wav2vec2.py \
  --batch_size 40 \
  --no-caching \
  --asrbn_tdnnf_vq 128 \
  --checkpoint_path exp/hifigan_w2w2_vq_128 \
  --asrbn_tdnnf_model local/chain/e2e/tuning/tdnnf_wav2vec_fairseq_hibitrate_vq.py \
  --asrbn_tdnnf_exp_path exp/chain/e2e_tdnnf_wav2vec_fairseq_hibitrate_vq_128/ \
  --cold_restart  \
  --init_weight_model ./exp/hifigan_w2w2/g_best


python3 -m torch.distributed.launch --nproc_per_node 2 \
  ./local/tuning/hifi_gan_wav2vec2.py \
  --batch_size 40 \
  --no-caching \
  --asrbn_tdnnf_vq 128 \
  --checkpoint_path exp/hifigan_w2w2_vq_128 \
  --asrbn_tdnnf_model local/chain/e2e/tuning/tdnnf_wav2vec_fairseq_hibitrate_vq.py \
  --asrbn_tdnnf_exp_path exp/chain/e2e_tdnnf_wav2vec_fairseq_hibitrate_vq_128/ \
  --init_weight_model exp/hifigan_w2w2_vq_128/g_00042000
```


### Train HifiGAN model No Wav2vec
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
  --init_weight_model exp/hifigan_tdnnf/g_00045000
```
