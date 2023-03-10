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
```
