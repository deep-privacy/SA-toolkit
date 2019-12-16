#!/bin/bash

dst=./data

mkdir -p $dst

spk_file=./download/LibriSpeech/SPEAKERS.TXT

spk2gender=$dst/spk2gender; [[ -f $spk2gender ]] && rm $spk2gender

grep "^^[0-9]*\s\|" -E  download/LibriSpeech/SPEAKERS.TXT | awk -F'|' '{gsub(/[ ]+/, ""); print $1, tolower($2)}' > $spk2gender

echo "$0 spk2gender file created"
