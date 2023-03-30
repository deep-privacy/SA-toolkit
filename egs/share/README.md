Shared data for other task
===

Scripts to be executed from `egs/share`

### Data prep data-augmentation

```bash
# Activate your env
[...]

./share/dataprep_aug.py --save-path ./data --download

./share/dataprep_aug.py --from ./data/RIRS_NOISES --make-csv-augment-reverb
./share/dataprep_aug.py --from ./data/musan_split --make-csv-augment-noise
```
