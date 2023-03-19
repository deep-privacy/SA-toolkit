#!/bin/bash

set -e


. ./path.sh

# eval_sets='libri vctk'
# eval_subsets='dev test'

eval_sets='vctk'
eval_subsets='test'

echo "Enter getdata@voiceprivacychallenge.univ-avignon.fr password : "
read -s password
echo

for dset in $eval_sets ; do
  for suff in $eval_subsets; do
    printf "${GREEN}\n Downloading ${dset}_${suff} set...${NC}\n"
    share/download_data.sh ${dset}_${suff} ${password} || exit 1
  done
done

temp=$(mktemp)

for suff in $eval_subsets; do
  for name in data/vctk_$suff/{enrolls_mic2,trials_f_common_mic2,trials_f_mic2,trials_m_common_mic2,trials_m_mic2}; do
    [ ! -f $name ] && echo "File $name does not exist" && exit 1
  done

  dset=data/vctk_$suff
  utils/subset_data_dir.sh --utt-list $dset/enrolls_mic2 $dset ${dset}_enrolls || exit 1
  cp $dset/enrolls_mic2 ${dset}_enrolls/enrolls || exit 1

  cut -d' ' -f2 $dset/trials_f_mic2 | sort | uniq > $temp
  utils/subset_data_dir.sh --utt-list $temp $dset ${dset}_trials_f || exit 1
  cp $dset/trials_f_mic2 ${dset}_trials_f/trials || exit 1

  cut -d' ' -f2 $dset/trials_f_common_mic2 | sort | uniq > $temp
  utils/subset_data_dir.sh --utt-list $temp $dset ${dset}_trials_f_common || exit 1
  cp $dset/trials_f_common_mic2 ${dset}_trials_f_common/trials || exit 1

  utils/combine_data.sh ${dset}_trials_f_all ${dset}_trials_f ${dset}_trials_f_common || exit 1
  cat ${dset}_trials_f/trials ${dset}_trials_f_common/trials > ${dset}_trials_f_all/trials

  cut -d' ' -f2 $dset/trials_m_mic2 | sort | uniq > $temp
  utils/subset_data_dir.sh --utt-list $temp $dset ${dset}_trials_m || exit 1
  cp $dset/trials_m_mic2 ${dset}_trials_m/trials || exit 1

  cut -d' ' -f2 $dset/trials_m_common_mic2 | sort | uniq > $temp
  utils/subset_data_dir.sh --utt-list $temp $dset ${dset}_trials_m_common || exit 1
  cp $dset/trials_m_common_mic2 ${dset}_trials_m_common/trials || exit 1

  utils/combine_data.sh ${dset}_trials_m_all ${dset}_trials_m ${dset}_trials_m_common || exit 1
  cat ${dset}_trials_m/trials ${dset}_trials_m_common/trials > ${dset}_trials_m_all/trials

  utils/combine_data.sh ${dset}_trials_all ${dset}_trials_f_all ${dset}_trials_m_all || exit 1
  cat ${dset}_trials_f_all/trials ${dset}_trials_m_all/trials > ${dset}_trials_all/trials
done
rm $temp
