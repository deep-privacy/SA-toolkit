# local/chain/train.py --stage 4 --conf configs/tdnnf_e2e

dset="data\/dev_clean_fbank_hires"
sed -i -E "/^test_set.*$/s/test_set\s=.*$/test_set\ =\ $dset/" configs/tdnnf_e2e_vq
# for dim in 64 128 256 384 512 768
for dim in 128 256 384 512 768
do
    cp "/lium/raid01_b/pchampi/lab/asr-based-privacy-preserving-separation/pkwrap/egs/librispeech/v1/exp/chain/e2e_tdnnf_vq_sizeco_$dim/final.pt" "/lium/raid01_b/pchampi/lab/asr-based-privacy-preserving-separation/pkwrap/egs/librispeech/v1/exp/chain/e2e_tdnnf_vq_sizeco_$dim/final.back.pt"
    sed -i -E "/^dirname|^model_args/s/[0-9]{2,}/$dim/" configs/tdnnf_e2e_vq
    local/chain/train.py --stage 4 --conf configs/tdnnf_e2e_vq
done
exit 0

for dset in "data\/test_clean_fbank_hires" "data\/test_other_fbank_hires"
do
    sed -i -E "/^test_set.*$/s/test_set\s=.*$/test_set\ =\ $dset/" configs/tdnnf_e2e_vq
    for dim in 64 128 256 384 512 768
    do
        sed -i -E "/^dirname|^model_args/s/[0-9]{2,}/$dim/" configs/tdnnf_e2e_vq
        local/chain/train.py --stage 7 --conf configs/tdnnf_e2e_vq
    done

    sed -i -E "/^test_set.*$/s/test_set\s=.*$/test_set\ =\ $dset/" configs/tdnnf_e2e
    local/chain/train.py --stage 7 --conf configs/tdnnf_e2e

done
