#!/bin/bash

set -e
KALDI_ROOT=`pwd`/../../../kaldi
if [ ! -L ./utils ]; then
  echo "Kaldi root: ${KALDI_ROOT}"
  ./local/make_links.sh $KALDI_ROOT || exit 1
  echo "Succesfuly created ln links"
fi

# Grid parameters for Kaldi scripts
export train_cmd="run.pl"
export cpu_cmd="run.pl"
export decode_cmd="run.pl"
export mkgraph_cmd="run.pl"

. ./path.sh
export LD_LIBRARY_PATH="$(pwd)/lib:$LD_LIBRARY_PATH"

stage=-1
# train_600 or train_clean_100 or train_clean_360
train_set=train_clean_100

corpus=
lm_url=www.openslr.org/resources/11

# option related to tree/phone-lm
shared_phones='true'
phones_type='biphone'

num_utts_subset=1400    # number of utterances in validation and training
                        # subsets used for shrinkage and diagnostics.
frames_per_iter=10000000 # each iteration of training, see this many frames per
                        # job, measured at the sampling rate of the features
                        # used.  This is just a guideline; it will pick a number
                        # that divides the number of samples in the entire data.


. ./utils/parse_options.sh
. configs/local.conf

if [[ "$train_set" == "train_600" &&  "$frames_per_iter" == 10000000 ]]; then
    frames_per_iter=30000000
fi

if [[ "$train_set" == "train_clean_360" &&  "$frames_per_iter" == 10000000 ]]; then
    frames_per_iter=20000000
fi

# Setting directory names
new_lang=data/lang_e2e_${phones_type}
dir=exp/chain/e2e_${train_set}
treedir=$dir/e2e_${phones_type}_tree  # it's actually just a trivial tree (no tree building)

required_scripts="download_lm.sh score.sh data_prep.sh prepare_dict.sh format_lms.sh"
for script_name in $required_scripts; do
    if [ ! -f local/$script_name ]; then
        cp $KALDI_ROOT/egs/librispeech/s5/local/$script_name local/
    fi
done

if [ $stage -le -1 ]; then
  echo "-- Stage -1 --"
  local/download_lm.sh $lm_url data/local/lm
fi

required_lm_files="3-gram.arpa.gz 3-gram.pruned.1e-7.arpa.gz 3-gram.pruned.3e-7.arpa.gz 4-gram.arpa.gz"
for lm_file in $required_lm_files; do
    if [ ! -f data/local/lm/$lm_file ]; then
        echo "$0: Did not find $lm_file in data/local/lm. Make sure it exists."
        echo "$0: Else run this script with --stage -1 to download the LM."
        exit 1
    fi
done

if [ $stage -le 0 ]; then
  echo "-- Stage 0 --"
  parts="test-clean dev-clean test-other dev-other train-clean-100"
  if [[ "$train_set" == "train_600" ]]; then
    parts="$parts train-other-500"
  fi
  if [[ "$train_set" == "train_clean_360" ]]; then
    parts="$parts train_clean_360"
  fi
  # format the data as Kaldi data directories
  for part in $parts ; do
    # use underscore-separated names in data directories.
    echo "Data prep $part"
    data_name=$(echo $part | sed s/-/_/g)
    if [ ! -f data/${data_name}/wav.scp ]; then
        local/data_prep.sh $corpus/$part data/${data_name}
    fi
  done

  if [[ "$train_set" == "train_600" ]]; then
    utils/combine_data.sh data/train_600 data/train_clean_100 data/train_other_500
  fi
fi

if [[ $stage -le 1 &&  ! -f "data/local/lm_less_phones/lm_tglarge.arpa.gz" ]]; then
  echo "-- Stage 1 --"
  mkdir -p data/local/lm_less_phones
  ln -rs data/local/lm/3-gram.arpa.gz data/local/lm_less_phones/lm_tglarge.arpa.gz
  ln -rs data/local/lm/3-gram.pruned.1e-7.arpa.gz data/local/lm_less_phones/lm_tgmed.arpa.gz
  ln -rs data/local/lm/3-gram.pruned.3e-7.arpa.gz data/local/lm_less_phones/lm_tgsmall.arpa.gz
  ln -rs data/local/lm/4-gram.arpa.gz data/local/lm_less_phones/lm_fglarge.arpa.gz
  echo "$0: Preparing lexicon"
  cp data/local/lm/librispeech-vocab.txt data/local/lm_less_phones/
  cat data/local/lm/librispeech-lexicon.txt | sed -e 's/[0,1,2]//g' > \
    data/local/lm_less_phones/librispeech-lexicon.txt

  echo "$0: Preparing dictionary"
  local/prepare_dict.sh --stage 3 --nj 30 --cmd "$cpu_cmd" \
    data/local/lm_less_phones data/local/lm_less_phones data/local/dict_lp

  echo "$0: Preparing lang"
  utils/prepare_lang.sh \
    --position_dependent_phones false \
    --share_silence_phones true \
    data/local/dict_lp \
    "<UNK>" data/local/lang_tmp_lp data/lang_lp

  echo "$0: Formatting LMs"
  local/format_lms.sh --src-dir data/lang_lp data/local/lm_less_phones
  # Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
  utils/build_const_arpa_lm.sh data/local/lm_less_phones/lm_tglarge.arpa.gz \
    data/lang_lp data/lang_lp_test_tglarge
  utils/build_const_arpa_lm.sh data/local/lm_less_phones/lm_fglarge.arpa.gz \
    data/lang_lp data/lang_lp_test_fglarge
fi

nj=$(nproc)
num_spk=$(wc -l < data/${train_set}/spk2utt)
[ $nj -gt $num_spk ] && nj=$num_spk

if [ $stage -le 2 ]; then
  echo "-- Stage 2 --"
  utils/data/get_utt2dur.sh --nj $nj data/${train_set}
  utils/data/perturb_speed_to_allowed_lengths.py 12 \
    data/${train_set} \
    data/${train_set}_sp
  utils/fix_data_dir.sh data/${train_set}_sp

fi

if [ $stage -le 3 ]; then
  echo "-- Stage 3 --"
  bash shutil/chain/check_lang.sh data/lang_lp $new_lang
fi

if [ $stage -le 4 ]; then
  echo "-- Stage 4 --"
  echo 'Estimating a phone language model for the denominator graph...'
  bash shutil/chain/estimate_e2e_phone_lm.sh --cmd "$cpu_cmd" \
    data/lang_lp $treedir \
    data/${train_set}_sp $shared_phones $phones_type $new_lang
fi

if [ $stage -le 5 ]; then
  echo "-- Stage 5 --"
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
  echo "-- Stage 6 --"
  cp $dir/0.trans_mdl $dir/final.mdl
  utils/lang/check_phones_compatible.sh \
    data/lang_lp_test_tgsmall/phones.txt $new_lang/phones.txt
  utils/mkgraph.sh \
    --self-loop-scale 1.0 --remove-oov data/lang_lp_test_tgsmall \
    $dir $treedir/graph_tgsmall || exit 1;
  rm $dir/final.mdl
fi

dir=$dir/fst_egs
fstdir=$treedir
data=data/${train_set}_sp
mkdir -p $dir $dir/info

if [ $stage -le 7 ]; then
  echo "-- Stage 7 --"
  utils/data/get_utt2dur.sh $data

  frames_per_eg=$(cat $data/allowed_lengths.txt | tr '\n' , | sed 's/,$//')

  # requires computing fbank.. but the model does it
  # [ ! -f "$data/utt2len" ] && feat-to-len scp:$data/feats.scp ark,t:$data/utt2len

  # use a good enough approximation
  [ ! -f "$data/utt2len" ] && awk '{ printf "%s %i\n", $1, 99.9 * $2 }' $data/utt2dur > $data/utt2len

  cat $data/utt2len | \
    awk '{print $1}' | \
    utils/shuffle_list.pl 2>/dev/null | head -$num_utts_subset > $dir/valid_uttlist


  len_uttlist=`wc -l $dir/valid_uttlist | awk '{print $1}'`
  if [ $len_uttlist -lt $num_utts_subset ]; then
    echo "Number of utterances which have length at least $frames_per_eg is really low. Please check your data." && exit 1;
  fi

  if [ -f $data/utt2uniq ]; then  # this matters if you use data augmentation.
    # because of this stage we can again have utts with lengths less than
    # frames_per_eg
    echo "File $data/utt2uniq exists, so augmenting valid_uttlist to"
    echo "include all perturbed versions of the same 'real' utterances."
    mv $dir/valid_uttlist $dir/valid_uttlist.tmp
    utils/utt2spk_to_spk2utt.pl $data/utt2uniq > $dir/uniq2utt
    cat $dir/valid_uttlist.tmp | utils/apply_map.pl $data/utt2uniq | \
      sort | uniq | utils/apply_map.pl $dir/uniq2utt | \
      awk '{for(n=1;n<=NF;n++) print $n;}' | sort  > $dir/valid_uttlist
    rm $dir/uniq2utt $dir/valid_uttlist.tmp
  fi

  # awk -v mf_len=222 '{if ($2 == mf_len) print $1}' | \
  cat $data/utt2len | \
    awk '{print $1}' | \
     utils/filter_scp.pl --exclude $dir/valid_uttlist | \
     utils/shuffle_list.pl 2>/dev/null | head -$num_utts_subset > $dir/train_subset_uttlist
  len_uttlist=`wc -l $dir/train_subset_uttlist | awk '{print $1}'`
  if [ $len_uttlist -lt $num_utts_subset ]; then
    echo "Number of utterances which have length at least $frames_per_eg is really low. Please check your data." && exit 1;
  fi

  num_frames=$(awk -v s=0.01 '{n += $2} END{printf("%.0f\n", (n / s))}' <$data/utt2dur)
  echo $num_frames > $dir/info/num_frames

  num_fst_jobs=$(cat $fstdir/num_jobs) || exit 1;
  for id in $(seq $num_fst_jobs); do cat $fstdir/fst.$id.scp; done > $fstdir/fst.scp

  utils/filter_scp.pl <(cat $dir/valid_uttlist) \
    <$fstdir/fst.scp >$dir/fst_valid.scp

  utils/filter_scp.pl <(cat $dir/train_subset_uttlist) \
    <$fstdir/fst.scp >$dir/fst_train_diagnostic.scp

  utils/filter_scp.pl --exclude $dir/valid_uttlist \
    <$fstdir/fst.scp >$dir/fst_train.scp


  # the + 1 is to round up, not down... we assume it doesn't divide exactly.
  num_archives=$[$num_frames/$frames_per_iter+1]

  echo $num_archives >$dir/info/num_archives

  utils/shuffle_list.pl $dir/fst_train.scp > $dir/fst_train_shuffle.scp

  for n in $(seq $num_archives); do
    split_scp="$split_scp $dir/fst_train.$n.scp"
  done

  utils/split_scp.pl $dir/fst_train_shuffle.scp $split_scp || exit 1;
fi
echo "Done"
