[Sidekit](https://git-lium.univ-lemans.fr/speaker/sidekit) ASV Voxceleb 1
===

To run the recipe:

```bash
# Activate your miniconda env
. ./path.sh

# Download dataset to data
./local/data_prep.py --save-path ./data --download

# Create train CSV list from directory (recursive search of wavs) (Change `--from data` if you already have downloaded the wavs)
./local/data_prep.py  --from ./data --make-train-data # set --filter-dir if your data dir structure differ from the '--download' one (e.g.: voxceleb1/wav/)

# Create test data
./local/data_prep.py  --from ./data --make-test-data # set --filter-dir if your data dir structure differ from the '--download' one (e.g.: voxceleb1_test/wav/)

# Train
./local/train.py  --config configs/...
```

### Results train-Voxceleb 1 (fbanks)
```sh
Test | as-norm    Voxceleb-0             Exp                              Config
EER/Link          1.1/0.91 | 1.0/0.92    exp/asv_eval_half_resnet_vox1    configs/half_resnet
```

### JIT model

```python3
import torch
import torchaudio
waveform, _, text_gt, speaker, chapter, utterance = torchaudio.datasets.LIBRISPEECH("/tmp", "dev-clean", download=True)[0]
model = torch.jit.load("__Exp_Path__/final.jit")
model = model.eval()

x_vector = model(waveform)
```
