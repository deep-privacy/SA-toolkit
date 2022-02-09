#!/usr/bin/env bash

pkwrap_libri_path="$HOME/lab/asr-based-privacy-preserving-separation/pkwrap/egs/librispeech/v1/"

./local/compute_metrics.sh \
    --pkwrap_model "local/chain/e2e/tuning/tdnnf_wav2vec_vq_lds.py" \
    --pkwrap_bn_dim 256 \
    --pkwrap_vq_dim 48 \
    --pkwrap_exp_dir "exp/chain/e2e_tdnnf_wav2vec_lds_vq_48/" \
    --asv_model ../../egs/libri460_fast_vq/model/best_model_vq_w2_48.pt_epoch486_EER_33.97_ACC_1.87.pt 2> /dev/null

./local/compute_metrics.sh \
    --pkwrap_model "local/chain/e2e/tuning/tdnnf_wav2vec.py" \
    --pkwrap_bn_dim 256 \
    --pkwrap_vq_dim -1 \
    --pkwrap_exp_dir "exp/chain/e2e_tdnnf_wav2vec/" \
    --asv_model ../../egs/libri460_fast_vq/model/best_model_w2.pt_epoch56_EER_30.28_ACC_25.14.pt 2> /dev/null

exit

exp=e2e_tdnnf
for suff in dev_clean_fbank_hires test_clean_fbank_hires test_other_fbank_hires; do
    if [ ! -f "${pkwrap_libri_path}/exp/chain/$exp/decode_${suff}_iterfinal_final_fg/scoringDetails/wer_details/wer_bootci" ]; then
        cd $pkwrap_libri_path
        . ./path.sh
        ./local/wer_detail.sh \
        --dataDir ./data/$suff \
        --decodeDir ${pkwrap_libri_path}/exp/chain/$exp/decode_${suff}_iterfinal_final_fg \
        --langDir data/lang_lp_test_fglarge
        cd -
    fi
    echo "WER bootci $(echo $suff | cut -d_ -f1,2 ): $(cat ${pkwrap_libri_path}/exp/chain/$exp/decode_${suff}_iterfinal_final_fg/scoringDetails/wer_details/wer_bootci)"
done


for dim in 1024 16 32 48 64 128 256 512; do

    ./local/compute_metrics.sh \
        --pkwrap_model "local/chain/e2e/tuning/tdnnf_vq_bd.py" \
        --pkwrap_bn_dim 256 \
        --pkwrap_vq_dim $dim \
        --pkwrap_exp_dir "exp/chain/e2e_tdnnf_vq_$dim/" \
        --asv_model ../../egs/libri460_fast_vq/model/best_model_vq_$dim.pt 2> /dev/null

    exp=e2e_tdnnf_vq_$dim
    for suff in test_clean_fbank_hires test_other_fbank_hires; do
        if [ ! -f "${pkwrap_libri_path}/exp/chain/$exp/decode_${suff}_iterfinal_final_fg/scoringDetails/wer_details/wer_bootci" ]; then
            cd $pkwrap_libri_path
            . ./path.sh
            ./local/wer_detail.sh \
            --dataDir ./data/$suff \
            --decodeDir ${pkwrap_libri_path}/exp/chain/$exp/decode_${suff}_iterfinal_final_fg \
            --langDir data/lang_lp_test_fglarge
            cd -
        fi
        echo "WER bootci $(echo $suff | cut -d_ -f1,2 ): $(cat ${pkwrap_libri_path}/exp/chain/$exp/decode_${suff}_iterfinal_final_fg/scoringDetails/wer_details/wer_bootci)"
    done

done
