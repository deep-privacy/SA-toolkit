#!/usr/bin/env bash

echo $@
echo "--"

# asv_model="../../egs/libri460_fast2/model/best_model.pt_epoch240_EER_10.57_ACC_91.51.pt"
asv_model="../../egs/libri460_fast2/model/best_model.pt"

# pkwrap_model="local/chain/e2e/tuning/tdnnf_vq.py"
pkwrap_model="local/chain/e2e/tuning/tdnnf_vq_spkdelta.py"
# pkwrap_bn_dim="256"
pkwrap_bn_dim="512"
pkwrap_vq_dim="16"

# pkwrap_exp_dir="exp/chain/e2e_tdnnf_vq_sizeco_$pkwrap_vq_dim/"
pkwrap_exp_dir="exp/chain/e2e_tdnnf_vq_spkdelta_sizeco_$pkwrap_vq_dim/"

# asv_model="../../egs/libri460_fast_vq/model/best_model_vq_$pkwrap_vq_dim.pt"
asv_model="../../egs/libri460_fast_vq_spkdelta/model/best_model_vq_$pkwrap_vq_dim.pt"

. ./local/parse_options.sh


export pkwrap_model=$pkwrap_model
export pkwrap_bn_dim=$pkwrap_bn_dim
export pkwrap_vq_dim=$pkwrap_vq_dim
export pkwrap_exp_dir=$pkwrap_exp_dir

asv_test=()
# librispeech
# for suff in 'dev' 'test'; do
for suff in 'test'; do
  # Baseline on clear speech
  asv_test+=("libri_${suff}_enrolls,libri_${suff}_trials_f")
  asv_test+=("libri_${suff}_enrolls,libri_${suff}_trials_m")

  # asv_test+=("vctk_${suff}_enrolls,vctk_${suff}_trials_f_common")
  # asv_test+=("vctk_${suff}_enrolls,vctk_${suff}_trials_m_common")
done

for asv_row in "${asv_test[@]}"; do
  while IFS=',' read -r enroll trial; do
      for data_dir in "$enroll" "$trial"; do
        \rm ./data/$data_dir/x_vector.scp
      done
  done <<< "$asv_row"
done

for asv_row in "${asv_test[@]}"; do
    while IFS=',' read -r enroll trial; do
        printf 'ASV: %s\n' "$enroll - $trial"

        for data_dir in "$enroll" "$trial"; do
          if [[ ! -f ./data/$data_dir/x_vector.scp ]]; then
            >&2 echo -e "Extracting x-vectors of $data_dir"
            python3 ./local/extract_xvectors.py \
              --model $asv_model \
              --wav-scp ./data/$data_dir/wav.scp \
              --out-scp ./data/$data_dir/x_vector.scp || exit 1
          fi
          if [[ ! "$(wc -l < ./data/$data_dir/wav.scp)" -eq "$(wc -l < ./data/$data_dir/x_vector.scp)" ]]; then >&2 echo -e "\nWarning: Something went wrong during the x-vector extraction!\nPlease redo the extraction:\n\trm ./data/$data_dir/x_vector.scp\n" && exit 2; fi
        done

        python3 ./local/compute_spk_cosine.py \
          ./data/$trial/trials \
          ./data/$enroll/utt2spk \
          ./data/$trial/x_vector.scp \
          ./data/$enroll/x_vector.scp \
          ./data/$trial/cosine_score_$enroll.txt || exit 1

        PYTHONPATH=$(realpath ../../../anonymization_metrics) \
        python3 ./local/compute_metrics.py \
          -k ./data/$trial/trials \
          -s ./data/$trial/cosine_score_$enroll.txt || exit 1

    done <<< "$asv_row"
done
