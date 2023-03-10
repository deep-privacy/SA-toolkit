#!/bin/bash

data_url_libritts=www.openslr.org/resources/60     # Link to download LibriTTS corpus

corpora=corpora

mkdir -p $corpora


mkdir -p ${datadir}
for part in dev-clean test-clean train-clean-100 ; do
    local/download_and_untar.sh --remove-archive ${corpora} ${data_url_libritts} ${part}
done
