#!/bin/bash

cd ~/lab/asr-based-privacy-preserving-separation
. ./env.sh
cd pkwrap/egs/librispeech/v1/
. ./path.sh

  for dim in 16 32 48 64 128 256 384 512 768
  do
    for delta in "" "_spkdelta"
    do
      for dset in "data\/test_clean_fbank_hires" "data\/test_other_fbank_hires" "data\/dev_clean_fbank_hires"
      do
        file="./exp/chain/e2e_tdnnf_vq${delta}_sizeco_${dim}/decode_${dset/data\\\//}_iterfinal_final"
        if [ -d $file ]; then
          if [ ! -f "${file}_fg/best_wer" ]; then
            echo "$file decoding error"


            (
            flock -x -w 20 200 || exit 1
            sed -i -E "/^test_set.*$/s/test_set\s=.*$/test_set\ =\ $dset/" configs/tdnnf_e2e_vq$delta
            sed -i -E "/^dirname|^model_args|^init_weight_model/s/[0-9]{2,}/$dim/" configs/tdnnf_e2e_vq$delta
            cat configs/tdnnf_e2e_vq$delta  | grep "^dirname"
            cp configs/tdnnf_e2e_vq$delta configs/tdnnf_e2e_vq${delta}_${dim}_decode
            ) 200>.parallel.exclusivelock
            local/chain/train.py --stage 7 --conf configs/tdnnf_e2e_vq${delta}_${dim}_decode

          fi
        fi
      done
    done
  done
