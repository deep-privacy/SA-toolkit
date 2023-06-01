#!/usr/bin/env bash

min_length=10
max_length=99999999999999999999999999

. utils/parse_options.sh

data_dir=$1
out_data_dir=$2

utt2len_file=$data_dir/utt2len

mkdir -p $out_data_dir

# Filter the data based on utterance length
utils/filter_scp.pl <(awk -v min=$min_length -v max=$max_length '$2 >= min && $2 <= max {print}' $utt2len_file | cut -d" " -f1) <$data_dir/wav.scp >$out_data_dir/wav.scp
utils/filter_scp.pl <(awk -v min=$min_length -v max=$max_length '$2 >= min && $2 <= max {print}' $utt2len_file | cut -d" " -f1) <$data_dir/text >$out_data_dir/text
utils/filter_scp.pl <(awk -v min=$min_length -v max=$max_length '$2 >= min && $2 <= max {print}' $utt2len_file | cut -d" " -f1) <$data_dir/utt2spk >$out_data_dir/utt2spk
utils/filter_scp.pl <(awk -v min=$min_length -v max=$max_length '$2 >= min && $2 <= max {print}' $utt2len_file | cut -d" " -f1) <$data_dir/spk2utt >$out_data_dir/spk2utt
utils/filter_scp.pl <(awk -v min=$min_length -v max=$max_length '$2 >= min && $2 <= max {print}' $utt2len_file | cut -d" " -f1) <$data_dir/utt2dur >$out_data_dir/utt2dur
utils/filter_scp.pl <(awk -v min=$min_length -v max=$max_length '$2 >= min && $2 <= max {print}' $utt2len_file | cut -d" " -f1) <$data_dir/utt2len >$out_data_dir/utt2len
cp $data_dir/spk2gender $out_data_dir/spk2gender


utils/fix_data_dir.sh $out_data_dir || exit 1
utils/validate_data_dir.sh --no-feats $out_data_dir || exit 1
