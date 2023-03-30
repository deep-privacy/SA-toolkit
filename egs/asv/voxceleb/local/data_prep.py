#!/usr/bin/env python3

# -*- coding: utf-8 -*-
# The script downloads the VoxCeleb datasets and converts all files to WAV.
# Requirement: ffmpeg and wget running on a Linux system.

# Taken from: https://raw.githubusercontent.com/clovaai/voxceleb_trainer/master/dataprep.py

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
from pathlib import Path

import torchaudio
import csv

## ========== ===========
## Parse input arguments
## ========== ===========
parser = argparse.ArgumentParser(description="VoxCeleb12 dataset preparation")

parser.add_argument(
    "--save-path",
    type=str,
    default="data",
    help="Target directory to download the dataset",
)
parser.add_argument("--user", type=str, default="user", help="Username")
parser.add_argument("--password", type=str, default="pass", help="Password")

parser.add_argument(
    "--download",
    dest="download",
    action="store_true",
    help="Download and extract the dataset",
)
parser.add_argument(
    "--convert", dest="convert", action="store_true", help="Convert voxceleb 2 from m4a to wav"
)
parser.add_argument(
    "--make-train-csv",
    dest="make_train_csv",
    action="store_true",
    help="Create the training sidekit csv",
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
    "--out-csv", default="list/voxceleb12.csv", type=str, help="File to the output csv"
)
parser.add_argument(
    "--fullpath",
    type=str,
    default="True",
    help='List training audio files with their full path, otherwise relative to "--from"',
)
parser.add_argument(
    "--filter-dir",
    dest="filter_dataset",
    type=str,
    default="voxceleb1/,voxceleb2/",
    help="List of dirs of process. Default: 'voxceleb1/,voxceleb2/' (delimited with ',')",
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
            "wget --no-check-certificate %s --user %s --password %s -O %s/%s"
            % (url, args.user, args.password, args.save_path, outfile),
            shell=True,
        )
        time.sleep(30)  # 30s sleep to attempt to avoid the Web Application Firewall
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
## Convert
## ========== ===========
def convert(args, files):
    print("Converting files from AAC to WAV")
    log_first = ""
    _log_first = ">/dev/null 2>/dev/null"
    for fname in tqdm(files):
        outfile = fname.replace(".m4a", ".wav")
        out = subprocess.call(
            "ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s %s"
            % (fname, outfile, log_first),
            shell=True,
        )
        log_first = _log_first
        if out != 0:
            raise ValueError("Conversion failed %s." % fname)


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
## Create sidekit csv file
##    FOR VOXCELEB12
## ========== ===========
def make_train_csv(root_data, out_filepath, fullpath, filter_dataset):
    # Browse directories to retrieve list of audio files
    spk_list = []
    with open(out_filepath, "w", newline="") as out_csv_file:
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

        pbar = tqdm(os.walk(root_data))
        for root, dirs, files in pbar:
            _continue = True
            for _filter in filter_dataset:
                if _filter in root:
                    _continue = False
                    break

            pbar.set_description(f"spk count: {len(spk_list)} scaning: {root}")
            if _continue:
                continue

            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file_path)[1] == ".wav":
                    dataset = root.split("/")
                    spk_id = dataset[-2]
                    if spk_id not in spk_list:
                        spk_list.append(spk_id)
                    spk_idx = spk_list.index(spk_id)
                    start = 0
                    duration = calculate_duration(file_path)
                    file_id = format_file_id(file_path, root_data, fullpath)

                    csv_writer.writerow(
                        [spk_idx, dataset[-3], spk_id, start, duration, file_id, "-"]
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
        duration = 0

    return duration


## ========== ===========
## Main script
## ========== ===========
if __name__ == "__main__":
    args = parser.parse_args()

    if args.make_train_csv:
        if not os.path.exists(args._from):
            raise ValueError(f"Dataset directory '{args._from}' does not exist.")

        make_train_csv(
            args._from, args.out_csv, args.fullpath, args.filter_dataset.split(",")
        )
        sys.exit(0)

    if not os.path.exists(args.save_path):
        raise ValueError(f"Target directory '{args.save_path}' does not exist.")

    f = open("list/fileparts.txt", "r")
    fileparts = f.readlines()
    f.close()

    f = open("list/files.txt", "r")
    files = f.readlines()
    f.close()

    if args.download:
        download(args, fileparts)
        concatenate(args, files)
        # DEV
        for file in files:
            full_extract(args, os.path.join(args.save_path, file.split()[1]))
        out = subprocess.call(
            "mv %s/wav %s/voxceleb1" % (args.save_path, args.save_path), shell=True
        )
        out = subprocess.call(
            "mv %s/dev/aac/ %s/aac/ && rm -r %s/dev"
            % (args.save_path, args.save_path, args.save_path),
            shell=True,
        )
        out = subprocess.call(
            "mv %s/aac %s/voxceleb2" % (args.save_path, args.save_path), shell=True
        )

        # TEST
        for file in fileparts:
            file = file.split()[0].split("/")[-1]
            if "test" not in file:
                continue
            full_extract(args, os.path.join(args.save_path, file))
        out = subprocess.call(
            "mv %s/wav %s/voxceleb1_test" % (args.save_path, args.save_path), shell=True
        )
        out = subprocess.call(
            "mv %s/aac %s/voxceleb2_test" % (args.save_path, args.save_path), shell=True
        )

    if args.convert:
        files = glob.glob("%s/voxceleb2_test/*/*/*.m4a" % args.save_path)
        files.sort()
        convert(args, files)
        files = glob.glob("%s/voxceleb2/*/*/*.m4a" % args.save_path)
        files.sort()
        convert(args, files)


