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

### convert
```bash
python convert.py

for dim in 16 32 48 64 128 256 512 1024
do
 python3 ./convert.py --num-workers 4 --batch-size 64 --vq-dim $dim --out generated_train-clean-360_vq_$dim --in /lium/home/pchampi/lab/asr-based-privacy-preserving-separation/pkwrap/egs/librispeech/v1/corpora/LibriSpeech/train-clean-360
done
```
