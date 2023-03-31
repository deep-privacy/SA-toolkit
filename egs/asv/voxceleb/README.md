[Sidekit](https://git-lium.univ-lemans.fr/speaker/sidekit) ASV Voxceleb 1
===

To run the recipe:

```bash
# Activate your miniconda env
. ./path.sh

# Download dataset to data
./local/data_prep.py --save-path ./data --download

# Create train CSV list from directory (recursive search of wavs) (Change `--from data` if you already have downloaded the wavs)
./local/data_prep.py  --from ./data --make-train-data # set --filter-dir if your data dir structure differ from the '--download' one (e.g.: voxceleb1/dev/wav/)

# Create test data
./local/data_prep.py  --from ./data --make-test-data # set --filter-dir if your data dir structure differ from the '--download' one (e.g.: voxceleb1_test/dev/wav/)
```
