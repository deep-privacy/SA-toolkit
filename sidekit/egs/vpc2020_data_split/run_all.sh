

for dim in 16 32 48 64 128 256 384 512 768 ;do

    ./local/compute_metrics.sh \
        --pkwrap_model "local/chain/e2e/tuning/tdnnf_vq_spkdelta.py" \
        --pkwrap_bn_dim 512 \
        --pkwrap_vq_dim $dim \
        --pkwrap_exp_dir "exp/chain/e2e_tdnnf_vq_spkdelta_sizeco_$dim/" \
        --asv_model ../../egs/libri460_fast_vq_spkdelta_l2norm/model/best_model_vq_$dim.pt 2> /dev/null

done
