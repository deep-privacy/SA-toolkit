#!/bin/bash


num_utts_subset=1400    # number of utterances in validation and training
                        # subsets used for shrinkage and diagnostics.

frames_per_iter=3000000 # each iteration of training, see this many frames per
                        # job, measured at the sampling rate of the features
                        # used.  This is just a guideline; it will pick a number
                        # that divides the number of samples in the entire data.

. ./utils/parse_options.sh

data=$1
fstdir=$2
dir=$3

mkdir -p $dir $dir/info

utils/data/get_utt2dur.sh $data

frames_per_eg=$(cat $data/allowed_lengths.txt | tr '\n' , | sed 's/,$//')

[ ! -f "$data/utt2len" ] && feat-to-len scp:$data/feats.scp ark,t:$data/utt2len

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

num_frames=$(steps/nnet2/get_num_frames.sh $data)
echo $num_frames > $dir/info/num_frames

num_fst_jobs=$(cat $fstdir/num_jobs) || exit 1;
for id in $(seq $num_fst_jobs); do cat $fstdir/fst.$id.scp; done > $fstdir/fst.scp

utils/filter_scp.pl <(cat $dir/valid_uttlist) \
  <$fstdir/fst.scp >$dir/fst_valid.scp

utils/filter_scp.pl <(cat $dir/train_subset_uttlist) \
  <$fstdir/fst.scp >$dir/fst_train_diagnositc.scp

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
