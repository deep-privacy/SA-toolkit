#!/usr/bin/env bash

beam=15.0
max_active=7000
min_active=200
lattice_beam=8.0 # Beam we use in lattice generation.
minimize=false

echo "$0 $@"
. ./utils/parse_options.sh

if [ $# -ne 4 ]; then
    echo "Usage: $0 words trans_mdl graph output"
    echo ""
    echo "words: words.txt file in the lang folder"
    echo "trans_mdl: transition model"
    echo "graph: HCLG.fst file"
    echo "output: output file, usually filename of type lat.x.gz, where x is some number"
    exit 1
fi

words=$1
trans_mdl=$2
graph=$3
output=$4

thread_string=
thread_string="-parallel --num-threads=10"

cat /dev/stdin | \
    latgen-faster-mapped$thread_string --minimize=$minimize \
        --max-active=$max_active \
        --min-active=$min_active --beam=$beam --lattice-beam=$lattice_beam \
        --acoustic-scale=1.0 --allow-partial=true --word-symbol-table=$words \
        $trans_mdl \
        $graph ark:- "ark:|lattice-scale --acoustic-scale=10.0 ark:- ark:- | gzip -c >$output"
