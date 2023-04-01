#!/usr/bin/env python3

"""
Copyright 2014-2021 Hubert Nourtel
"""

import os
import subprocess
import io
import csv

import torchaudio
import soundfile

from tqdm import tqdm
from argparse import ArgumentParser



def main():
    parser = ArgumentParser("Convert kaldi directory to sidekit csv")
    parser.add_argument("--kaldi-data", type=str, required=True)
    parser.add_argument("--out-csv", type=str, required=True)
    parser.add_argument("--sr", type=int, required=False, default=16000)
    args = parser.parse_args()
    kaldi_data_path = args.kaldi_data
    out_csv_path = args.out_csv

    # Read spk2gender file
    spk2gender_dict = {}
    if os.path.exists(os.path.join(kaldi_data_path, "spk2gender")):
        with open(os.path.join(kaldi_data_path, "spk2gender"), "r") as spk2gender_file:
            for line in spk2gender_file:
                split_line = line.split()
                spk2gender_dict[split_line[0]] = split_line[1]


    # Read utt2spk file
    utt2spk_dict = {}
    with open(os.path.join(kaldi_data_path, "utt2spk")) as utt2spk_file:
        for line in utt2spk_file:
            split_line = line.split()
            utt2spk_dict[split_line[0]] = split_line[1]

    # Read utt2dur file
    utt2dur_dict = {}
    with open(os.path.join(kaldi_data_path, "utt2dur"), "r") as utt2dur_file:
        for line in utt2dur_file:
            split_line = line.split()
            utt2dur_dict[split_line[0]] = split_line[1]

    spk_list = []
    out_csv_file = open(out_csv_path, "w", newline="")
    csv_writer = csv.writer(
        out_csv_file, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
    )

    # Write header
    csv_writer.writerow(
        [
            "speaker_idx",
            "database",
            "speaker_id",
            "start",
            "duration",
            "file_id",
            "gender",
        ]
    )

    wav_scp_path = os.path.join(kaldi_data_path, "wav.scp")
    dataset_name = os.path.basename(kaldi_data_path)

    # Count lines in scp file
    num_lines = sum(1 for line in open(wav_scp_path, "r"))

    # Iterate over wav.scp lines
    scp_file = open(wav_scp_path, "r")
    iter_scp = tqdm(scp_file, total=num_lines)
    for line in iter_scp:
        split_line = line.split()
        utt_id = split_line[0]
        file_path = split_line[1]
        spk_id = utt2spk_dict[utt_id]

        # Create list of speaker to have a unique idx for the user
        if spk_id not in spk_list:
            spk_list.append(spk_id)
        spk_idx = spk_list.index(spk_id)

        start = 0
        # Load duration from existing reco2dur file
        duration = int(float(utt2dur_dict[utt_id])*args.sr)

        gender = "-"
        if spk_id in spk2gender_dict:
            gender = spk2gender_dict[spk_id]

        csv_writer.writerow(
            [
                spk_idx,
                dataset_name,
                spk_id,
                start,
                duration,
                file_path,
                gender,
            ]
        )

    scp_file.close()
    out_csv_file.close()


if __name__ == "__main__":
    main()
