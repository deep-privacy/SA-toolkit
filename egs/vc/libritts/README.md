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
asrbn_model=bn_tdnnf_t100_vq_64 local/train.py --conf configs/hifigan
```

### JIT model convert/anonymize speech

```python3
import torch
import torchaudio
waveform, _, text_gt, speaker, chapter, utterance = torchaudio.datasets.LIBRISPEECH("/tmp", "dev-clean", download=True)[0]
torchaudio.save(f"/tmp/clear_{speaker}-{chapter}-{str(utterance)}.wav", waveform, 16000)
model = torch.jit.load("__Exp_Path__/final.jit")
model = model.eval()

wav_conv = model.convert(waveform, target="1069")
torchaudio.save(f"/tmp/anon_{speaker}-{chapter}-{str(utterance)}.wav", wav_conv, 16000)

# or to modify some feature (like the F0 which is usefull for better anonymization in some cases)
f0, asrbn, spk_id = model.extract_features(waveform, target="1069")
f0 *= torch.randn(f0.size()) # you may want to use something more powerfull
wav_convv = model._forward(f0, asrbn, spk_id).squeeze(0)
torchaudio.save(f"/tmp/anon2_{speaker}-{chapter}-{str(utterance)}.wav", wav_convv, 16000)
```
