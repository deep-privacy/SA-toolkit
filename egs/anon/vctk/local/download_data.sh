#!/bin/sh

set -e

KALDI_ROOT=`pwd`/../kaldi
if [ ! -L ./utils ]; then
  echo "Kaldi root: ${KALDI_ROOT}"
  ./share/make_links.sh $KALDI_ROOT || exit 1
  echo "Succesfuly created ln links"
fi

if [ $# != 1 ]; then
  echo "Usage: "
  echo "  $0 [options] <data-set> <password>"
  exit 1;
fi

data_set=$1
pass=$2
expo_dir=data/$data_set

dir=$expo_dir
if [ ! -f $dir/wav.scp ]; then
  [ -d $dir ] && rm -r $dir
  if [ ! -f  $data_set.tar.gz ]; then
    sshpass -p "$pass" sftp getdata@voiceprivacychallenge.univ-avignon.fr <<EOF
cd /challengedata/corpora
get $data_set.tar.gz
bye
EOF
  fi
  echo "  Unpacking $data_set data set..."
  tar -xf $data_set.tar.gz || exit 1
  [ ! -f $dir/text ] && echo "File $dir/text does not exist" && exit 1
  cut -d' ' -f1 $dir/text > $dir/text1
  cut -d' ' -f2- $dir/text | sed -r 's/,|!|\?|\./ /g' | sed -r 's/ +/ /g' | awk '{print toupper($0)}' > $dir/text2
  paste -d' ' $dir/text1 $dir/text2 > $dir/text
  rm $dir/text1 $dir/text2
  utils/fix_data_dir.sh $dir || exit 1
  utils/validate_data_dir.sh --no-feats $dir || exit 1
fi

echo '  Done'
