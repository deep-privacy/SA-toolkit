#!/bin/bash

set -e

. ./configs/local.conf

sets="train-clean-100 dev-clean"

KALDI_ROOT=`pwd`/../../../kaldi
if [ ! -L ./utils ]; then
  echo "Kaldi root: ${KALDI_ROOT}"
  ./local/make_links.sh $KALDI_ROOT || exit 1
  echo "Succesfuly created ln links"
fi

for part in $sets; do
  local/data_prep.sh $corpus/$part/ data/${part//-/_}
done

for part in $sets; do
  echo "Audio resample $part"
  python3 ./local/preprocess.py \
    --srcdir $corpus/$part/ \
    --outdir data/${part//-/_}/wavs_16khz --pad
done

for part in $sets; do
  cp data/${part//-/_}/wav.scp data/${part//-/_}/wav_original_SR.scp

  cat data/${part//-/_}/wav_original_SR.scp | \
    sed "s/\(.*\)\ .*/\1 data\/${part//-/_}\/wavs_16khz\/\1.wav/g" > data/${part//-/_}/wav.scp
done

utils/subset_data_dir.sh --per-spk ./data/dev_clean 2 ./data/dev_clean_reduced_2utt
