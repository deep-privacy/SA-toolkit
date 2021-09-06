  steps/online/nnet2/extract_ivectors_online.sh --nj $nj --cmd "$train_cmd" \
    $data ${ivec_extr} $ivect || exit 1


local/chain/train.py --stage 4 --conf configs/tdnnf_e2e_vq
