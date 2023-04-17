Hifi-GAN VC
===

To run the recipe:

```bash
# Activate your miniconda env
. ./path.sh

#  Change the path to librispeech database in `configs/local.conf` and/or use `local/download_libri.sh`
./local/prepare_data.sh

# Train with archi and data defined in configs and local/tuning/ (configs: model_file)
local/train.py --conf configs/...

# Change asrbn extractor with
asrbn_model=bn_tdnnf_100h_vq_64 local/train.py --conf configs/hifigan

# Create final (.pt and .jit) models (based on the quality of the audio generated)
asrbn_model=bn_tdnnf_100h_vq_64 local/train.py --conf configs/hifigan --stage 10 --final-model ./exp/bn_tdnnf_100h_vq_64/g_00111000.pt
```

### TODO
The model performs f0 normalization over the utterance for each utterance, it would be much better to do it on a per speaker basis.
Adapt the code of: [link](https://github.com/deep-privacy/SA-toolkit/blob/482055bdb61f285e77115fb73e4a2af337ab9e89/pkwrap/egs/LJSpeech/local/get_f0_stats_hifi_gan_w2w2_libriTTS.py#L26) to new jit yaapt extractor and adapt the model / CMVN function.


### JIT model convert/anonymize speech

```python3
import torch
import torchaudio
waveform, _, text_gt, speaker, chapter, utterance = torchaudio.datasets.LIBRISPEECH("/tmp", "dev-clean", download=True)[1]
torchaudio.save(f"/tmp/clear_{speaker}-{chapter}-{str(utterance)}.wav", waveform, 16000)
model = torch.jit.load("__Exp_Path__/final.jit")
# model = torch.jit.load("exp/hifigan_bn_tdnnf_100h_aug/final.jit")
model = model.eval()

wav_conv = model.convert(waveform, target="1069")
torchaudio.save(f"/tmp/anon_{speaker}-{chapter}-{str(utterance)}.wav", wav_conv, 16000)

# Modify some feature (like the F0 which can be useful for better anonymization on some models)
f0, asrbn, spk_id = model.extract_features(waveform, target="1069")
f0[f0 != 0] += torch.randn(f0[f0 != 0].size()) * 20 # you may want to use something else ;)
wav_convv = model._forward(f0, asrbn, spk_id).squeeze(0)
torchaudio.save(f"/tmp/anon2_{speaker}-{chapter}-{str(utterance)}.wav", wav_convv, 16000)

# list the possible targets:
target_s = set(model.utt2spk.values())
```
