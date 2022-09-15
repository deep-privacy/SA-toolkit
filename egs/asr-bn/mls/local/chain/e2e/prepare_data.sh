#!/bin/bash

set -e

stage=-1
train_set=mls_train
affix=tdnnf

corpus=
lm_url=www.openslr.org/resources/11

# option related to tree/phone-lm
shared_phones='true'
phones_type='biphone'

# Options related to egs generation for training
get_egs_stage=-10
cmvn_opts=""
left_context=0
right_context=0
frames_per_iter=3000000
frame_subsampling_factor=3

. ./utils/parse_options.sh
. configs/local.conf
. ../path.sh

KALDI_ROOT=`pwd`/../../../kaldi
if [ ! -L ./utils ]; then
  echo "Kaldi root: ${KALDI_ROOT}"
  ./make_links.sh $KALDI_ROOT || exit 1
  echo "Successfuly created ln links"
fi

export LD_LIBRARY_PATH="$(pwd)/lib:$LD_LIBRARY_PATH"
# Grid parameters for Kaldi scripts
export train_cmd="run.pl"
export cpu_cmd="run.pl"
export decode_cmd="run.pl"
export mkgraph_cmd="run.pl"


# Setting directory names
new_lang=data/lang_e2e_${phones_type}
treedir=exp/chain/e2e_${phones_type}_tree  # it's actually just a trivial tree (no tree building)
dir=exp/chain/e2e_${affix}

required_scripts="score.sh prepare_dict.sh format_lms.sh"
for script_name in $required_scripts; do
    if [ ! -f local/$script_name ]; then
        cp $KALDI_ROOT/egs/librispeech/s5/local/$script_name local/
        if [ $script_name = "prepare_dict.sh" ]; then
          # For prepare_dict.sh, change librispeech reference to mls reference
          sed -i 's/librispeech/mls/g' local/$script_name
          sed -i 's/<UNK>/<unk>/g' local/$script_name
        fi
    fi
done

if [ $stage -le -1 ]; then
  echo "$0: Downloading lm"
  # Checking if ngram is installed. Useful for pruning lm
  if [ ! ngram ]; then
    echo "Error : ngram not found. Please run kaldi/tools/extras/install_srilm.sh to install it."
    exit 1
  fi

  dst_lm_dir=data/local/lm
  mkdir -p $dst_lm_dir
  wget https://dl.fbaipublicfiles.com/mls/mls_lm_french.tar.gz -P $dst_lm_dir
  tar -xzf $dst_lm_dir/mls_lm_french.tar.gz -C $dst_lm_dir --strip-components=1
  rm $dst_lm_dir/mls_lm_french.tar.gz
  # Prune 3-gram lm
  ngram -prune 0.0000003 -lm $dst_lm_dir/"3-gram_lm.arpa" -write-lm $dst_lm_dir/"3-gram_lm.pruned.3e-7.arpa"
  ngram -prune 0.0000001 -lm $dst_lm_dir/"3-gram_lm.arpa" -write-lm $dst_lm_dir/"3-gram_lm.pruned.1e-7.arpa"
  # Zip all lm
  for lm_model in 3-gram_lm.arpa 3-gram_lm.pruned.1e-7.arpa 3-gram_lm.pruned.3e-7.arpa 5-gram_lm.arpa; do
    gzip $dst_lm_dir/$lm_model
  done
  # Convert vocab_counts into vocabulary file
  cat $dst_lm_dir/vocab_counts.txt | awk '{print $1}' > $dst_lm_dir/mls-vocab.txt
fi

if [ $stage -le 0 ]; then
  # format the data as Kaldi data directories
  for part in train dev test; do
    data_name=mls_$part
    if [ ! -d data/${data_name} ]; then
        local/mls_to_kaldi.py \
          --mls_root $corpus \
          --data_split $part \
          --out_dir data/${data_name}_fbank_hires
    fi
  done
fi

if [ $stage -le 1 ]; then
  mkdir -p data/local/lm_less_phones
  if [ ! -L data/local/lm_less_phones/lm_tglarge.arpa.gz ]; then
    ln -rs data/local/lm/3-gram_lm.arpa.gz data/local/lm_less_phones/lm_tglarge.arpa.gz
  fi
  if [ ! -L data/local/lm_less_phones/lm_tgmed.arpa.gz ]; then
    ln -rs data/local/lm/3-gram_lm.pruned.1e-7.arpa.gz data/local/lm_less_phones/lm_tgmed.arpa.gz
  fi
  if [ ! -L data/local/lm_less_phones/lm_tgsmall.arpa.gz ]; then
    ln -rs data/local/lm/3-gram_lm.pruned.3e-7.arpa.gz data/local/lm_less_phones/lm_tgsmall.arpa.gz
  fi
  if [ ! -L data/local/lm_less_phones/lm_fglarge.arpa.gz ]; then
    ln -rs data/local/lm/5-gram_lm.arpa.gz data/local/lm_less_phones/lm_fglarge.arpa.gz
  fi
  echo "$0: Preparing lexicon"
  cp data/local/lm/mls-vocab.txt data/local/lm_less_phones/
  if [ ! -f data/local/lm/mls-lexicon.txt ]; then
    echo "Please create lexicon for mls dataset from 'data/local/lm/mls-vocab.txt' in 'data/local/lm/mls-lexicon.txt'"
    exit 1
  fi
  cp data/local/lm/mls-lexicon.txt data/local/lm_less_phones/

  echo "$0: Preparing dictionary"
  local/prepare_dict.sh --stage 3 --nj 30 --cmd "$cpu_cmd" \
    data/local/lm_less_phones data/local/lm_less_phones data/local/dict_lp

  echo "$0: Preparing lang"
  utils/prepare_lang.sh \
    --position_dependent_phones false \
    --share_silence_phones true \
    data/local/dict_lp \
    "<unk>" data/local/lang_tmp_lp data/lang_lp

  echo "$0: Formatting LMs"
  local/format_lms.sh --src-dir data/lang_lp data/local/lm_less_phones
  # Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
  utils/build_const_arpa_lm.sh data/local/lm_less_phones/lm_tglarge.arpa.gz \
    data/lang_lp data/lang_lp_test_tglarge
  utils/build_const_arpa_lm.sh data/local/lm_less_phones/lm_fglarge.arpa.gz \
    data/lang_lp data/lang_lp_test_fglarge
fi

if [ $stage -le 2 ]; then
  utils/data/get_utt2dur.sh data/${train_set}_fbank_hires
  utils/data/perturb_speed_to_allowed_lengths.py 12 \
    data/${train_set}_fbank_hires \
    data/${train_set}_sp_fbank_hires
  utils/fix_data_dir.sh data/${train_set}_sp_fbank_hires

  for part in ${train_set}_sp; do
    datadir=${part}_fbank_hires
    # Extracting 80 dim filter bank features
    mkdir -p data/feats/fbank
    steps/make_fbank.sh --fbank-config configs/fbank_hires.conf \
      --cmd "$cpu_cmd" --nj 50 data/${datadir} \
      data/feats/fbank/${datadir} data/feats/fbank/${datadir}/data || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir} \
      data/feats/fbank/${datadir} data/feats/fbank/${datadir}/data || exit 1;
    utils/fix_data_dir.sh data/${datadir} || exit 1
  done
fi

# feature extraction ends here
if [ $stage -le 3 ]; then
  bash shutil/chain/check_lang.sh data/lang_lp $new_lang
fi

if [ $stage -le 4 ]; then
  echo 'Estimating a phone language model for the denominator graph...'
  bash shutil/chain/estimate_e2e_phone_lm.sh --cmd "$cpu_cmd" \
    data/lang_lp $treedir \
    data/${train_set}_sp_fbank_hires $shared_phones $phones_type $new_lang
fi

if [ $stage -le 5 ]; then
  mkdir -p ${dir}/configs
  mkdir -p ${dir}/init
  cp -r $treedir/tree $dir/
  cp $treedir/phones.txt $dir/
  cp $treedir/phone_lm.fst $dir/
  cp $treedir/0.trans_mdl $dir/
  echo 'Making denominator fst for training'
  bash shutil/chain/make_e2e_den_fst.sh \
    --cmd "$cpu_cmd" $treedir $dir
fi


# Generate a decoding graph to decode the validation data
# for early stopping
if [ $stage -le 6 ]; then
  cp $dir/0.trans_mdl $dir/final.mdl
  utils/lang/check_phones_compatible.sh \
    data/lang_lp_test_tgsmall/phones.txt $new_lang/phones.txt
  utils/mkgraph.sh \
    --self-loop-scale 1.0 --remove-oov data/lang_lp_test_tgsmall \
    $dir $treedir/graph_tgsmall || exit 1;
  rm $dir/final.mdl
fi