#!/usr/bin/env bash

set -e

mkdir -p ./exp/align/
TMPFILE=$(mktemp -d ./exp/align/d_XXX) || exit 1

export show_bn=1

model="local/chain/e2e/tuning/tdnnf_vq.py  --codebook-size 16"
exp_dir="exp/chain/e2e_tdnnf_vq_sizeco_16/"

# model="local/chain/e2e/tuning/tdnnf.py"
# exp_dir="exp/chain/e2e_tdnnf/"

echo $model

$model \
    --dir $exp_dir \
    --mode decode --use-gpu True --gpu-id 1 \
    --decode-feats test_one.scp $exp_dir/final.pt \
    | shutil/decode/latgen-faster-mapped.sh \
    --beam 1 \
    exp/chain/e2e_biphone_tree/graph_tgsmall/words.txt \
    $exp_dir/0.trans_mdl \
    exp/chain/e2e_biphone_tree/graph_tgsmall/HCLG.fst \
    $TMPFILE/lat.gz


oldlm=./data/lang_lp_test_tgsmall/G.fst
newlm=./data/lang_lp_test_fglarge/G.carpa
oldlmcommand="fstproject --project_output=true $oldlm |"

lattice-lmrescore --lm-scale=-1.0 \
    "ark:gunzip -c $TMPFILE/lat.gz|" "$oldlmcommand" ark:-  | \
    lattice-lmrescore-const-arpa --lm-scale=1.0 \
    ark:- "$newlm" "ark,t:|gzip -c>$TMPFILE/lat_rescore.gz"

# bypass rescore
# cp $TMPFILE/lat.gz $TMPFILE/lat_rescore.gz

# Word alin
lattice-align-words-lexicon \
    ./data/lang_lp/phones/align_lexicon.int \
    $exp_dir/0.trans_mdl \
    "ark:gunzip -c $TMPFILE/lat_rescore.gz|" ark:-  \
    | lattice-1best ark:- ark:- |  nbest-to-ctm  ark:- $TMPFILE/align.ctm

python3 local/convert_ctm.py -i $TMPFILE/align.ctm \
    -w exp/chain/e2e_biphone_tree/graph_tgsmall/words.txt \
    -o $TMPFILE/out_ctm

# cat $TMPFILE/out_ctm


# Phone alin

zcat $TMPFILE/lat_rescore.gz  > $TMPFILE/lat
lattice-1best --acoustic-scale=1 ark:$TMPFILE/lat ark:$TMPFILE/1best.lats
nbest-to-linear ark:$TMPFILE/1best.lats ark,t:$TMPFILE/ali
ali-to-phones --ctm-output $exp_dir/0.trans_mdl \
    ark:$TMPFILE/ali \
    $TMPFILE/phone_alined.ctm

python3 local/convert_ctm.py -i $TMPFILE/phone_alined.ctm \
    -w ./data/lang_lp/phones.txt \
    -o $TMPFILE/out_phone_ctm

# cat $TMPFILE/out_phone_ctm

copy-int-vector ark,t:$TMPFILE/ali ark,t:$TMPFILE/transids.txt
show-transitions ./data/lang_lp/phones.txt $exp_dir/0.trans_mdl > $TMPFILE/transitions.txt
python3 ./local/map_kaldi_transitionids.py --input $TMPFILE/transids.txt --input_transitions $TMPFILE/transitions.txt --output $TMPFILE/out_state_seq

echo "cat $TMPFILE/out_state_seq"
cat $TMPFILE/out_state_seq
