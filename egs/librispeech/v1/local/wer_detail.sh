#!/bin/bash

# echo ./local/wer_detail.sh $@

dataDir=$1
decodeDir=$2
langDir=$3

cmd=run.pl

word_ins_penalty=0.0,0.5,1.0
min_lmwt=7
max_lmwt=17

. parse_options.sh || exit 1;

symtab=$langDir/words.txt
outDir=$decodeDir/scoringDetails


mkdir -p $outDir
cat $dataDir/text | cat > $decodeDir/scoring/test_filt.txt || exit 1;
for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
  for lmwt in $(seq $min_lmwt $max_lmwt); do
    # adding /dev/null to the command list below forces grep to output the filename
    grep WER $decodeDir/wer_${lmwt}_${wip} /dev/null
  done | utils/best_wer.sh  >& $outDir/best_wer || exit 1
done

best_wer_file=$(awk '{print $NF}' $outDir/best_wer)
best_wip=$(echo $best_wer_file | awk -F_ '{print $NF}')
best_lmwt=$(echo $best_wer_file | awk -F_ '{N=NF-1; print $N}')

if [ -z "$best_lmwt" ]; then
  echo "$0: we could not get the details of the best WER from the file $decodeDir/wer_*.  Probably something went wrong."
  exit 1;
fi

mkdir -p $outDir/penalty_$best_wip/log
$cmd LMWT=$best_lmwt $outDir/penalty_$best_wip/log/best_path.LMWT.log \
      lattice-scale --inv-acoustic-scale=LMWT "ark:gunzip -c $decodeDir/lat.*.gz|" ark:- \| \
      lattice-add-penalty --word-ins-penalty=$best_wip ark:- ark:- \| \
      lattice-best-path --word-symbol-table=$symtab ark:- ark,t:- \| \
      utils/int2sym.pl -f 2- $symtab \| \
      cat '>' $outDir/penalty_$best_wip/LMWT.txt || exit 1;

mkdir -p $outDir/wer_details
echo $best_lmwt > $outDir/wer_details/lmwt # record best language model weight
echo $best_wip > $outDir/wer_details/wip # record best word insertion penalty

$cmd $outDir/log/stats1.log \
  cat $outDir/penalty_$best_wip/$best_lmwt.txt \| \
  align-text --special-symbol="'***'" ark:$decodeDir/scoring/test_filt.txt ark:- ark,t:- \|  \
  utils/scoring/wer_per_utt_details.pl --special-symbol "'***'" \| tee $outDir/wer_details/per_utt \|\
   utils/scoring/wer_per_spk_details.pl $dataDir/utt2spk \> $outDir/wer_details/per_spk || exit 1;

$cmd $outDir/log/stats2.log \
  cat $outDir/wer_details/per_utt \| \
  utils/scoring/wer_ops_details.pl --special-symbol "'***'" \| \
  sort -b -i -k 1,1 -k 4,4rn -k 2,2 -k 3,3 \> $outDir/wer_details/ops || exit 1;

$cmd $outDir/log/wer_bootci.log \
  compute-wer-bootci --mode=present \
    ark:$decodeDir/scoring/test_filt.txt ark:$outDir/penalty_$best_wip/$best_lmwt.txt \
    '>' $outDir/wer_details/wer_bootci || exit 1;
