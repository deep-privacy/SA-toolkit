#!/bin/sh

set -e

KALDI_ROOT=`pwd`/../../../kaldi
if [ ! -L ./utils ]; then
  echo "Kaldi root: ${KALDI_ROOT}"
  ./local/make_links.sh $KALDI_ROOT || exit 1
  echo "Succesfuly created ln links"
fi

if [ $# != 2 ]; then
  echo "Usage: "
  echo "  $0 [options] <data-set> <password>"
  exit 1;
fi

data_set=$1
pass=$2
expo_dir=data/$data_set
from_github=true

dir=$expo_dir
[ -d $dir ] && rm -r $dir
if [ ! -f  $data_set.tar.gz ]; then

  if ! $from_github; then
    echo "Downloading from sftp"
    sshpass -p "$pass" sftp getdata@voiceprivacychallenge.univ-avignon.fr <<EOF
cd /challengedata/corpora
get $data_set.tar.gz
bye
EOF
  else
    echo "Downloading from github release"
    wget -q --show-progress https://github.com/deep-privacy/SA-toolkit/releases/download/vctk_test_data/vctk_test.tar.gz.gpg
    gpg --batch --passphrase "$pass" vctk_test.tar.gz.gpg
    \rm -rf vctk_test.tar.gz.gpg
  fi

fi
echo "Unpacking $data_set data set..."
tar -xf $data_set.tar.gz || exit 1
[ ! -f $dir/text ] && echo "File $dir/text does not exist" && exit 1
cut -d' ' -f1 $dir/text > $dir/text1
cut -d' ' -f2- $dir/text | sed -r 's/,|!|\?|\./ /g' | sed -r 's/ +/ /g' | awk '{print toupper($0)}' > $dir/text2
paste -d' ' $dir/text1 $dir/text2 > $dir/text
rm $dir/text1 $dir/text2
utils/fix_data_dir.sh $dir || exit 1
utils/validate_data_dir.sh --no-feats $dir || exit 1

echo '  Done'
