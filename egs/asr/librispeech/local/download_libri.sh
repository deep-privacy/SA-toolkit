#!/bin/bash

data_url_librispeech=www.openslr.org/resources/12  # Link to download LibriSpeech corpus
data_url_libritts=www.openslr.org/resources/60     # Link to download LibriTTS corpus

corpora=corpora

mkdir -p $corpora

printf "Stage 3: Downloading LibriSpeech data sets \n"
for part in train-other-500 train-clean-360 train-clean-100 dev-clean test-clean dev-other test-other; do
  local/download_and_untar.sh $corpora $data_url_librispeech $part LibriSpeech || exit 1;
done
