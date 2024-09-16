#!/usr/bin/env python3

# -*- coding: utf-8 -*-
# The script downloads the VoxCeleb datasets and converts all files to WAV.
# Requirement: ffmpeg and wget running on a Linux system.

# Taken from: https://raw.githubusercontent.com/clovaai/voxceleb_trainer/master/dataprep.py


"""
Usage:

./share/dataprep_aug.py --save-path ./data --download

./share/dataprep_aug.py --from ./data/RIRS_NOISES --make-csv-augment-reverb
./share/dataprep_aug.py --from ./data/musan_split --make-csv-augment-noise
"""

import argparse
import os
import sys
import subprocess
import hashlib
import time
import glob
import tarfile
from zipfile import ZipFile
from tqdm import tqdm
from scipy.io import wavfile
from pathlib import Path

import torchaudio
import csv

## ========== ===========
## Parse input arguments
## ========== ===========
parser = argparse.ArgumentParser(description="Data Augmentation dataset preparation")

parser.add_argument(
    "--save-path",
    type=str,
    default="data",
    help="Target directory to download the dataset",
)

parser.add_argument(
    "--download",
    dest="download",
    action="store_true",
    help="Download and extract the dataset",
)
parser.add_argument(
    "--make-csv-augment-reverb",
    dest="make_aug_csv_reverb",
    action="store_true",
    help="Create the dataset aug sidekit csv",
)
parser.add_argument(
    "--make-csv-augment-noise",
    dest="make_aug_csv_noise",
    action="store_true",
    help="Create the dataset aug sidekit csv",
)

## args for --make-*-csv
parser.add_argument(
    "--from",
    default="./data",
    dest="_from",
    type=str,
    help="Path to the root of the dataset",
)
parser.add_argument(
    "--out-csv", default="list/list.csv", type=str, help="File to the output csv"
)
parser.add_argument(
    "--fullpath",
    type=str,
    default="True",
    help='List training audio files with their full path, otherwise relative to "--from"',
)


## ========== ===========
## MD5SUM
## ========== ===========
def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


## ========== ===========
## Download with wget
## ========== ===========
def download(args, lines):
    for line in lines:
        url = line.split()[0]
        md5gt = line.split()[1]
        outfile = url.split("/")[-1]

        if os.path.exists("%s/%s" % (args.save_path, outfile)):
            print("File '%s' exist checking md5.." % outfile)
            md5ck = md5("%s/%s" % (args.save_path, outfile))
            if md5ck == md5gt:
                print("Skipping already downloaded '%s' md5 match." % outfile)
                continue

        ## Download files
        out = subprocess.call(
            "wget --no-check-certificate %s -O %s/%s" % (url, args.save_path, outfile),
            shell=True,
        )
        if out != 0:
            print(
                "Download %s failed please retry!!\n\nIf download fails too repeatedly, use alternate URL on the VoxCeleb website. (modify list/fileparts.txt)"
                % outfile
            )
            sys.exit(1)

        ## Check MD5
        md5ck = md5("%s/%s" % (args.save_path, outfile))
        if md5ck == md5gt:
            print("Checksum successful %s." % outfile)
        else:
            raise Warning("Checksum failed %s." % outfile)


## ========== ===========
## Concatenate file parts
## ========== ===========
def concatenate(args, lines):
    for line in lines:
        infile = line.split()[0]
        outfile = line.split()[1]
        md5gt = line.split()[2]

        ## Concatenate files
        out = subprocess.call(
            "cat %s/%s > %s/%s" % (args.save_path, infile, args.save_path, outfile),
            shell=True,
        )

        ## Check MD5
        md5ck = md5("%s/%s" % (args.save_path, outfile))
        if md5ck == md5gt:
            print("Checksum successful %s." % outfile)
        else:
            raise Warning("Checksum failed %s." % outfile)

        out = subprocess.call("rm %s/%s" % (args.save_path, infile), shell=True)


## ========== ===========
## Extract zip files
## ========== ===========
def full_extract(args, fname):
    print("Extracting %s" % fname)
    if fname.endswith(".tar.gz"):
        with tarfile.open(fname, "r:gz") as tar:
            tar.extractall(args.save_path)
    elif fname.endswith(".zip"):
        with ZipFile(fname, "r") as zf:
            zf.extractall(args.save_path)


## ========== ===========
## Partially extract zip files
## ========== ===========
def part_extract(args, fname, target):
    print("Extracting %s" % fname)
    with ZipFile(fname, "r") as zf:
        for infile in zf.namelist():
            if any([infile.startswith(x) for x in target]):
                zf.extract(infile, args.save_path)
            # zf.extractall(args.save_path)


## ========== ===========
## Split MUSAN for faster random access
## ========== ===========
def split_musan(args):
    files = glob.glob("%s/musan/*/*/*.wav" % args.save_path)

    audlen = 16000 * 5
    audstr = audlen

    for idx, file in enumerate(files):
        fs, aud = wavfile.read(file)
        writedir = os.path.splitext(file.replace("/musan/", "/musan_split/"))[0]
        os.makedirs(writedir)
        for st in range(0, len(aud) - audlen, audstr):
            wavfile.write(writedir + "/%05d.wav" % (st / fs), fs, aud[st : st + audlen])

        print(idx, file)


## ========== ===========
## Create sidekit csv file
##  FOR AUGMENTATAION
## ========== ===========
def make_aug_csv_reverb(root_data, out_filepath, fullpath):
    with open(out_filepath, "w", newline="") as out_csv_file:
        csv_writer = csv.writer(
            out_csv_file, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )

        # Write header
        csv_writer.writerow(
            [
                "channel",
                "database",
                "file_id",
                "type",
            ]
        )
        wav_count = 0
        pbar = tqdm(os.walk(root_data))
        for root, dirs, files in pbar:
            if Path(root).parent == Path(root_data):
                dataset = root.split("/")[-1]
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file_path)[1] == ".wav":
                    wav_count += 1
                    pbar.set_description(f"wav count : {wav_count}")
                    file_id = format_file_id(file_path, root_data, fullpath, False)

                    csv_writer.writerow([1.0, "REVERB", file_id, dataset])


def make_aug_csv_noise(root_data, out_filepath, fullpath):
    with open(out_filepath, "w", newline="") as out_csv_file:
        csv_writer = csv.writer(
            out_csv_file, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )

        # Write header
        csv_writer.writerow(
            [
                "database",
                "type",
                "file_id",
                "start",
                "duration",
            ]
        )
        wav_count = 0
        pbar = tqdm(os.walk(root_data))
        for root, dirs, files in pbar:
            if Path(root).parent == Path(root_data):
                dataset = root.split("/")[-1]
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file_path)[1] == ".wav":
                    wav_count += 1
                    pbar.set_description(f"wav count : {wav_count}")
                    duration = calculate_duration(file_path)
                    file_id = format_file_id(file_path, root_data, fullpath)

                    csv_writer.writerow(
                        [
                            "musan",
                            dataset,
                            file_id,
                            0.0,
                            duration,
                        ]
                    )


## ========== ===========
## Create CSV utils
## ========== ===========
def format_file_id(file_path, root_data, full_path, remove_extension=True):
    if remove_extension:
        file_id = os.path.splitext(os.path.realpath(file_path))[0]
    else:
        file_id = os.path.realpath(file_path)

    if full_path.lower() != "true":
        # Remove file extension and file root_data
        file_id = file_id.replace(root_data, "")
        # Remove first slash if present (it is not root_data)
        file_id = file_id[1:] if file_id[0] == "/" else file_id

    return file_id


def calculate_duration(file_path):
    try:
        audio_info = torchaudio.info(file_path)
        duration = audio_info.num_frames / audio_info.sample_rate
    except Exception:
        print("failed to load info of:", file_path)
        sys.exit(1)
        duration = 0

    return duration


## ========== ===========
## Main script
## ========== ===========
if __name__ == "__main__":
    args = parser.parse_args()

    if args.make_aug_csv_reverb:
        if not os.path.exists(args._from):
            os.makedirs(args._from, exist_ok=True)

        if args._from == parser.get_default("_from"):
            # change default to downloaded reverb
            args._from = "./data/RIRS_NOISES"

        if args.out_csv == parser.get_default("out_csv"):
            # change default to output reverb list
            args.out_csv = "data/reverb.csv"

        make_aug_csv_reverb(args._from, args.out_csv, args.fullpath)
        sys.exit(0)

    if args.make_aug_csv_noise:
        if not os.path.exists(args._from):
            raise ValueError(f"Dataset directory '{args._from}' does not exist.")

        if args._from == parser.get_default("_from"):
            # change default to downloaded reverb
            args._from = "./data/musan_split"

        if args.out_csv == parser.get_default("out_csv"):
            # change default to output reverb list
            args.out_csv = "data/musan.csv"

        make_aug_csv_noise(args._from, args.out_csv, args.fullpath)
        sys.exit(0)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    augfiles = """http://www.openslr.org/resources/28/rirs_noises.zip e6f48e257286e05de56413b4779d8ffb
http://www.openslr.org/resources/17/musan.tar.gz 0c472d4fc0c5141eca47ad1ffeb2a7df"""

    if args.download:
        download(args, augfiles.split("\n"))
    #  part_extract(args,os.path.join(args.save_path,'rirs_noises.zip'),['RIRS_NOISES/simulated_rirs/mediumroom','RIRS_NOISES/simulated_rirs/smallroom'])
    part_extract(args, os.path.join(args.save_path, "rirs_noises.zip"), ["RIRS_NOISES"])
    full_extract(args, os.path.join(args.save_path, "musan.tar.gz"))
    split_musan(args)
    if args.download:
        out = subprocess.call("rm -rf %s/%s" % (args.save_path, "musan"), shell=True)
