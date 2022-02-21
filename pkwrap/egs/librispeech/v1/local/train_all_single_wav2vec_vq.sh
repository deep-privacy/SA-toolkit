model="configs/tdnnf_e2e_wav2vec2_hibitrate_vq"
for dim in 64 1024 16 32 48 128  512 256
do

  dset="data\/test_clean_fbank_hires"

  sed -i -E "/^test_set.*$/s/test_set\s=.*$/test_set\ =\ $dset/" $model
  sed -i -E "/^dirname|^model_args|^init_weight_model/s/[0-9]{2,}/$dim/" $model
  cat $model  | grep "^dirname"
  local/chain/train.py --stage 4 --conf $model

    for dset in "data\/test_other_fbank_hires"
    do
        sed -i -E "/^test_set.*$/s/test_set\s=.*$/test_set\ =\ $dset/" $model
        local/chain/train.py --stage 8 --conf $model
    done

done
exit 0
