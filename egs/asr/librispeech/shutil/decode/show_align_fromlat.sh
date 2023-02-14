#!/usr/bin/env bash

set -e


# oldlm=./data/lang_lp_test_tgsmall/G.fst
# newlm=./data/lang_lp_test_fglarge/G.carpa
# oldlmcommand="fstproject --project_output=true $oldlm |"

# lattice-lmrescore --lm-scale=-1.0 \
    # "ark:gunzip -c $1|" "$oldlmcommand" ark:-  | \
    # lattice-lmrescore-const-arpa --lm-scale=1.0 \
    # ark:- "$newlm" "ark,t:|gzip -c>$1.res"

# bypass rescore
cp $1 $1.res

exp_dir="exp/chain/e2e_tdnnf/"

# Word alin
lattice-align-words-lexicon \
    ./data/lang_lp/phones/align_lexicon.int \
    $exp_dir/0.trans_mdl \
    "ark:gunzip -c $1.res|" ark:-  \
    | lattice-1best ark:- ark:- |  nbest-to-ctm  ark:- $1.res_align.ctm

python3 shutil/decode/convert_ctm.py -i $1.res_align.ctm \
    -w exp/chain/e2e_biphone_tree/graph_tgsmall/words.txt \
    -o $1.res_align.ctm_out_ctm

# cat $TMPFILE/out_ctm


# Phone alin

zcat $1.res  > $1.res_lat
lattice-1best --acoustic-scale=1 ark:$1.res_lat ark:$1.res_lat_1best.lats
nbest-to-linear ark:$1.res_lat_1best.lats ark:$1.res_lat_ali
ali-to-phones --ctm-output $exp_dir/0.trans_mdl \
    ark:$1.res_lat_ali \
    $1.phone_alined.ctm

python3 shutil/decode/convert_ctm.py -i $1.phone_alined.ctm \
    -w ./data/lang_lp/phones.txt \
    -o $1.out_phone_ctm

# cat $TMPFILE/out_phone_ctm

copy-int-vector ark,t:$1.res_lat_ali ark,t:$1.transids.txt
show-transitions ./data/lang_lp/phones.txt $exp_dir/0.trans_mdl > $1.transitions.txt
python3 ./shutil/decode/map_kaldi_transitionids.py --input $1.transids.txt --input_transitions $1.transitions.txt --output $1.out_state_seq
