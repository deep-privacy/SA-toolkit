#!/usr/bin/env python3

"""
Copyright 2014-2021 Hubert Nourtel
"""

import os
import numpy as np
from satools.sidekit.bosaris import IdMap, Key
import argparse
from tqdm import tqdm

def main():

    # Retrieve arguments
    parser = argparse.ArgumentParser(description="Creating tests files for Sidekit training from Kaldi format dataset")
    parser.add_argument('--enrolls', required=True, type=str, help='Path to the enrollment file (Kaldi format)')
    parser.add_argument("--utt2spk", required=True, type=str, help='Path to the enrollment utt2spk file (Kaldi format)', dest="enroll_utt2spk")
    parser.add_argument('--trials', required=True, type=str, help='Path to the trials file (Kaldi format)')
    parser.add_argument('--out-dir', type=str, default='.', help='Path to the output directory')
    parser.add_argument('--out-file-prefix', type=str, help='Prefix for all output files')
    parser.add_argument('--out-format', type=str, default="h5", choices=["h5", "txt"], help='Format of output files')

    args = parser.parse_args()
    print("Creating test files..")

    # IDmap
    left_ids = []
    right_ids = []
    with open(args.enrolls) as enrolls_file:
        enrolls_lines = enrolls_file.read().splitlines()
    for line in tqdm(enrolls_lines, desc="Enrolls file processing"):
        cleared_line = line.split()[0]
        left_ids.append(cleared_line)
        right_ids.append(cleared_line)

    idmap = IdMap()
    idmap.set(np.array(left_ids), np.array(right_ids))

    # Key
    # Read utt2spk to have the matching model/speaker (model's name is the name of the utterance)
    enrolls_utt2spk = {}
    with open(args.enroll_utt2spk) as enrolls_utt2spk_file:
        for line in enrolls_utt2spk_file:
            split_line = line.split()
            enrolls_utt2spk[split_line[0]] = split_line[1]

    # Read trials file and store results in dict with all utterance for a given speaker
    trials_dict = {}
    segset = []
    with open(args.trials) as trials_file:
        trials_lines = trials_file.read().splitlines()
    for line in tqdm(trials_lines, desc="Trials file processing"):
        split_line = line.split()
        spk = split_line[0]
        utt = split_line[1]
        target = split_line[2]
        trials_dict.setdefault(spk, []).append((utt, target)) # key = spk, val = [(utt, target type)]
        if utt not in segset:
            segset.append(utt)

    # For each model in Idmap, retrieve trial utterances for the model's speaker and fill target/non-target lists
    modelset = []
    non = []
    tar = []
    for model in tqdm(left_ids, desc="Key file creation"):
        modelset.append(model)
        spk_model = enrolls_utt2spk[model]
        new_non = [False] * len(segset)
        new_tar = [False] * len(segset)
        for trial_utt, trial_tar in trials_dict[spk_model]:
            seg_idx = segset.index(trial_utt)
            if trial_tar == "target":
                new_tar[seg_idx] = True
            else:
                new_non[seg_idx] = True
        non.append(np.array(new_non))
        tar.append(np.array(new_tar))

    key = Key.create(np.array(modelset), np.array(segset), np.array(tar), np.array(non))

    # Ndx from Idmap
    ndxData = key.to_ndx()

    out_dir = args.out_dir
    out_prefix = args.out_file_prefix
    out_format = args.out_format

    # Writing output files
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if out_format == "txt":
        idmapWriteFunc = idmap.write_txt
        keyWriteFunc = key.write_txt
        ndxWriteFunc = ndxData.save_txt
    else:
        idmapWriteFunc = idmap.write
        keyWriteFunc = key.write
        ndxWriteFunc = ndxData.write
    idmapWriteFunc(os.path.join(out_dir, out_prefix + "_idmap." + out_format))
    keyWriteFunc(os.path.join(out_dir, out_prefix + "_key." + out_format))
    ndxWriteFunc(os.path.join(out_dir, out_prefix + "_ndx." + out_format))

    print("Bosaris test files successfully created")

if __name__ == "__main__":
    main()

