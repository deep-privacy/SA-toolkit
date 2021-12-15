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
