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
  echo "Data prep $part"
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
    sed "s/\([^-]*\)-\(.*\)\ .*/\1-\2 data\/${part//-/_}\/wavs_16khz\/\2.wav/g" > data/${part//-/_}/wav.scp
done

for part in $sets; do
  ./utils/data/get_utt2dur.sh --nj $(nproc) data/${part//-/_}

  awk '{ printf "%s %i\n", $1, 16000 * $2 }' data/${part//-/_}/utt2dur > data/${part//-/_}/utt2len
done

local/filterlen_data_dir.sh --max-length 96000 ./data/dev_clean ./data/dev_clean_max_size
utils/subset_data_dir.sh --per-spk ./data/dev_clean_max_size 2 ./data/dev_clean_reduced
