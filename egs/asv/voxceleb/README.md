[Sidekit](https://git-lium.univ-lemans.fr/speaker/sidekit) ASV Voxceleb 1
===

To run the recipe:

```bash
# Activate your miniconda env
. ./path.sh

# Download dataset to data
./local/data_prep.py --save-path ./data --download

# Create train data (recursive search of wavs) (Change `--from data` if you already have downloaded the wavs)
./local/data_prep.py  --from ./data --make-train-data # set --filter-dir if your data dir structure differ from the '--download' one (e.g.: voxceleb1/wav/)

# Create test data
./local/data_prep.py  --from ./data --make-test-data

# Train
./local/train.py  --config configs/...
```

### Results train-Voxceleb 1 (fbanks)
```sh
Test                Voxceleb-0               Exp                                Config
EER / min cllr      2.593 ± 0.0   / 0.106    exp/asv_eval_vox1_ecapa_tdnn       configs/ecapa_tdnn
EER / min cllr      2.089 ± 0.408 / 0.105    exp/asv_eval_vox1_ecapa_tdnn_ft    configs/ecapa_tdnn_fine_tune
EER / min cllr      2.413 ± 0.101 / 0.101    exp/asv_eval_vox1_resnet           configs/resnet
```

_Note: On VCTK, the resnet model seems to be better._  
_Note: ecapa_tdnn converges faster._

### JIT model

```python3
import torch
import torchaudio
waveform, _, text_gt, speaker, chapter, utterance = torchaudio.datasets.LIBRISPEECH("/tmp", "dev-clean", download=True)[0]
model = torch.jit.load("__Exp_Path__/final.jit")
model = model.eval()

_, x_vector = model(waveform)
```



[Sidekit](https://git-lium.univ-lemans.fr/speaker/sidekit) ASV Voxceleb 2
===

Training on vox 2 and evaluating on vox 1
To run the recipe:

```bash
# Activate your miniconda env
. ./path.sh

# Download & convert dataset to data
./local/data_prep.py --save-path ./data --download --with-vox2
./local/data_prep.py --from ./data --convert --with-vox2

# Create train data (recursive search of wavs) (Change `--from data` if you already have downloaded the wavs)
./local/data_prep.py --from ./data --with-vox2  --make-train-data  --filter-dir  "voxceleb2/"

# Create test data
./local/data_prep.py --from ./data --with-vox2 --make-test-data

# Train
./local/train.py  --config configs/...
```

[Sidekit](https://git-lium.univ-lemans.fr/speaker/sidekit) ASV Voxceleb 12
===

Training on vox 1&2 and evaluating on vox 1
To run the recipe:

```bash
# Activate your miniconda env
. ./path.sh

# Download & convert dataset to data
./local/data_prep.py --save-path ./data --download --with-vox2
./local/data_prep.py --from ./data --convert --with-vox2

# Create train data (recursive search of wavs) (Change `--from data` if you already have downloaded the wavs)
./local/data_prep.py --from ./data --with-vox2  --make-train-data

# Create test data
./local/data_prep.py --from ./data --with-vox2 --make-test-data

# Train
./local/train.py  --config configs/...
```
