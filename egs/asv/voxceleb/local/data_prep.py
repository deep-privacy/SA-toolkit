#!/usr/bin/env python3

# -*- coding: utf-8 -*-
# The script downloads the VoxCeleb datasets and converts all files to WAV.
# Requirement: ffmpeg and wget running on a Linux system.

# Taken from: https://raw.githubusercontent.com/clovaai/voxceleb_trainer/master/dataprep.py
# If password are necessary, Please fill this [form](https://docs.google.com/forms/d/e/1FAIpQLSdQhpq2Be2CktaPhuadUMU7ZDJoQuRlFlzNO45xO-drWQ0AXA/viewform?fbzx=7440236747203254000) or this [form](https://mm.kaist.ac.kr/datasets/voxceleb/#downloads) to request a password.

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
    "--with-vox2", dest="vox2", action="store_true", help="Convert voxceleb 2 from m4a to wav", default=False,
)
parser.add_argument(
    "--make-train-data",
    dest="make_train_data",
    action="store_true",
    help="Create the training data",
)
parser.add_argument(
    "--make-test-data",
    dest="make_test",
    action="store_true",
    help="Create the 'Kaldi' test data",
)

## args for --make-*-data
parser.add_argument(
    "--from",
    default="./data",
    dest="_from",
    type=str,
    help="Path to the root of the dataset",
)
parser.add_argument(
    "--filter-dir",
    dest="filter_dataset",
    type=str,
    default="voxceleb1/,voxceleb2/",
    help="List of dirs of process. Default: 'voxceleb1/,voxceleb2/' (delimited with ',')",
)

parser.add_argument(
    "--tasks",
    type=str,
    default="voxceleb1-O voxceleb1-O-clean voxceleb1-E voxceleb1-E-clean voxceleb1-H voxceleb1-H-clean",
    help="The Dataset splits for identification",
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
    print("Concatenating")
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
        subprocess.call(["unzip", "-q", fname, "-d", args.save_path])


## ========== ===========
## Create data dir
##    FOR VOXCELEB12
## ========== ===========
def make_train_data(root_data, data_dir, filter_dataset):
    # Browse directories to retrieve list of audio files
    spk_list = []
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
                file_id, file_path = format_file_id(file_path)
                duration = calculate_duration(file_path)

                # write to wav.scp file
                with open(os.path.join(data_dir, "wav.scp"), "a") as wav_f:
                    wav_f.write(f"{file_id} {file_path}\n")

                # write to utt2spk file
                with open(os.path.join(data_dir, "utt2spk"), "a") as utt_f:
                    utt_f.write(f"{file_id} {spk_id}\n")

                # write to utt2dur file
                with open(os.path.join(data_dir, "utt2dur"), "a") as utt_d:
                    utt_d.write(f"{file_id} {duration}\n")

## ========== ===========
## Create data utils
## ========== ===========
def format_file_id(file_path):
    path = os.path.realpath(file_path)
    file_id = os.path.splitext(path)[0]
    file_id = "-".join(file_id.split("/")[-3:])
    return file_id, path


def calculate_duration(file_path):
    try:
        audio_info = torchaudio.info(file_path)
        duration = audio_info.num_frames / audio_info.sample_rate
    except Exception:
        print("failed to load info of:", file_path)
        duration = 0

    return duration


def make_test(args):
    if not args.vox2:
        args.tasks = "voxceleb1-O voxceleb1-O-clean"

    for task in args.tasks.split():
        name = ""
        if task == "voxceleb1-O":
            name = "veri_test.txt"
        if task == "voxceleb1-O-clean":
            name = "veri_test2.txt"
        if task == "voxceleb1-E-clean":
            name = "list_test_all2.txt"
        if task == "voxceleb1-H-clean":
            name = "list_test_hard2.txt"

        if name == "":
            raise Warning("Invalid task '%s'" % task)

        ###########################
        #  Download trials files  #
        ###########################

        if not os.path.exists("%s/%s" % (args.save_path, name)):
            out = subprocess.call(
                "wget --no-check-certificate %s -O %s/%s"
                % (
                    f"http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/{name}",
                    args.save_path,
                    name,
                ),
                shell=True,
            )
            if out != 0:
                print(
                    "Download %s failed please retry!!\n\nIf download fails too repeatedly, use alternate URL on the VoxCeleb website."
                    % name
                )
                sys.exit(1)

        #########################################
        #  convert trials file to kaldi format  #
        #########################################

        out = subprocess.call(
            """
            sed 's/\//-/g;s/\.wav//g' %s/%s | awk '{if($1=="1"){print $2,$3,"target"}else{print $2,$3,"nontarget"}}' > %s/%s.trials
            """
            % (
                args.save_path,
                name,
                args.save_path,
                task,
            ),
            shell=True,
        )
        if out != 0:
            print("Failed to convert files to kaldi trials" % name)
            sys.exit(1)

    ############################################################
    #  Index all wav file (from voxceleb 1 and voxceleb test)  #
    ############################################################

    indexed_dir = []
    pbar = tqdm(os.walk(args._from, followlinks=True))
    for root, dirs, files in pbar:
        _continue = True
        for _filter in args.filter_dataset.split(","):
            if _filter in root:
                _continue = False
                break

        if _continue:
            continue

        pbar.set_description(f"scaning: {root}")
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.splitext(file_path)[1] == ".wav":
                indexed_dir.append(file_path)


    ###################################################################
    #  Create wav.scp files from indexed wavs and kaldi trials files  #
    ###################################################################

    for task in args.tasks.split():
        os.makedirs(args.save_path + "/" + task, exist_ok=True)
        _trial = args.save_path + "/" + task + ".trials"
        task_trial = args.save_path + "/" + task + "/trials"
        out = subprocess.call(
            "mv %s %s" % (_trial, task_trial), shell=True
        )
        trial_file = open(task_trial)
        trial_file_val = set(trial_file.read().splitlines())
        trial_file.close()

        enroll_path = set()
        trials_path = set()

        for test in trial_file_val:
            test_path = test
            for position in [
                7,
                19,
                33,
                45,
            ]:  # Voxceleb only hardcoded filepath key position
                test_path = test_path[:position] + "/" + test_path[position + 1 :]

            enroll_path.add(test_path.split()[0])
            trials_path.add(test_path.split()[1])

        enroll_scp = open(args.save_path + "/" + task + "/enroll.wav.scp", "w")
        trials_scp = open(args.save_path + "/" + task + "/trials.wav.scp", "w")

        part_part_id = 0
        indexed_dir_dict = {}

        for file in tqdm(indexed_dir, desc=f"Creating scp files for {task}"):
            part_path = file.split("/")
            if len(part_path) < part_part_id or not part_path[part_part_id].startswith(
                "id"
            ):
                for i, pp in enumerate(part_path):
                    if part_path[i].startswith("id"):
                        part_part_id = i
            key = "/".join(part_path[part_part_id : part_part_id + 3]).replace(".wav", "")
            indexed_dir_dict[key] = file

        for e_p in enroll_path:
            if e_p in indexed_dir_dict:
                enroll_e_p = e_p
                for position in [7, 19]:  # Voxceleb only revert to original filepath key
                    enroll_e_p = enroll_e_p[:position] + "-" + enroll_e_p[position + 1 :]
                enroll_scp.write(f"{enroll_e_p} {indexed_dir_dict[e_p]}\n")
            else:
                raise Warning(f"{task} - Key '{e_p}' not found in scanned dir: {args._from}")

        for t_p in trials_path:
            if t_p in indexed_dir_dict:
                tirals_t_p = t_p
                for position in [7, 19]:  # Voxceleb only revert to original filepath key
                    tirals_t_p = tirals_t_p[:position] + "-" + tirals_t_p[position + 1 :]
                trials_scp.write(f"{tirals_t_p} {indexed_dir_dict[t_p]}\n")
            else:
                raise Warning(f"{task} - Key '{t_p}' not found in scanned dir: {args._from}")

        enroll_scp.close()
        trials_scp.close()

        utt2spk = open(args.save_path + "/" + task + "/enroll.utt2spk", "w")

        for e_p in enroll_path:
            # for Voxceleb the asv test is not done speaker <-> utter
            # but uttr <-> utter
            # so the utt2spk file is a utt2utt file.
            enroll_e_p = e_p
            for position in [7, 19]:  # Voxceleb only revert to original filepath key
                enroll_e_p = enroll_e_p[:position] + "-" + enroll_e_p[position + 1 :]
            utt2spk.write(f"{enroll_e_p} {enroll_e_p}\n")
        utt2spk.close()



## ========== ===========
## Main script
## ========== ===========
if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    if args.make_train_data:
        if not os.path.exists(args._from):
            raise ValueError(f"Dataset directory '{args._from}' does not exist.")

        data_dir = "%s/voxceleb1" % (args.save_path)
        if args.vox2:
            data_dir = "%s/voxceleb12" % (args.save_path)
            os.makedirs(data_dir, exist_ok=True)

        make_train_data(
            args._from, data_dir, args.filter_dataset.split(",")
        )
        sys.exit(0)

    ######
    # You might need to update the link to the fileparts: https://github.com/clovaai/voxceleb_trainer/blob/master/lists/fileparts.txt
    ####

    if args.vox2:
        fileparts = """https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partaa e395d020928bc15670b570a21695ed96
https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partab bbfaaccefab65d82b21903e81a8a8020
https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partac 017d579a2a96a077f40042ec33e51512
https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partad 7bb1e9f70fddc7a678fa998ea8b3ba19
https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partaa da070494c573e5c0564b1d11c3b20577
https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partab 17fe6dab2b32b48abaf1676429cdd06f
https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partac 1de58e086c5edf63625af1cb6d831528
https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partad 5a043eb03e15c5a918ee6a52aad477f9
https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partae cea401b624983e2d0b2a87fb5d59aa60
https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partaf fc886d9ba90ab88e7880ee98effd6ae9
https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partag d160ecc3f6ee3eed54d55349531cb42e
https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partah 6b84a81b9af72a9d9eecbb3b1f602e65
https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip 185fdc63c3c739954633d50379a3d102
https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_test_aac.zip 0d2b3ea430a821c33263b5ea37ede312"""
        files = """vox1_dev_wav_parta* vox1_dev_wav.zip ae63e55b951748cc486645f532ba230b
vox2_dev_aac_parta* vox2_dev_aac.zip bbc063c46078a602ca71605645c2a402"""

    else:
        fileparts = """https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partaa e395d020928bc15670b570a21695ed96
https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partab bbfaaccefab65d82b21903e81a8a8020
https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partac 017d579a2a96a077f40042ec33e51512
https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partad 7bb1e9f70fddc7a678fa998ea8b3ba19
https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip 185fdc63c3c739954633d50379a3d102"""
        files = """vox1_dev_wav_parta* vox1_dev_wav.zip ae63e55b951748cc486645f532ba230b"""


    if args.download:
        download(args, fileparts.split("\n"))
        concatenate(args, files.split("\n"))
        # DEV
        for file in files.split("\n"):
            full_extract(args, os.path.join(args.save_path, file.split()[1]))
        os.makedirs("%s/voxceleb1" % (args.save_path), exist_ok=True)
        out = subprocess.call(
            "mv %s/wav %s/voxceleb1" % (args.save_path, args.save_path), shell=True
        )
        if args.vox2:
            out = subprocess.call(
                "mv %s/dev/aac/ %s/aac/ && rm -r %s/dev"
                % (args.save_path, args.save_path, args.save_path),
                shell=True,
            )
            os.makedirs("%s/voxceleb2" % (args.save_path), exist_ok=True)
            out = subprocess.call(
                "mv %s/aac %s/voxceleb2" % (args.save_path, args.save_path), shell=True
            )

        # TEST
        for file in fileparts.split("\n"):
            file = file.split()[0].split("/")[-1]
            if "test" not in file:
                continue
            full_extract(args, os.path.join(args.save_path, file))
        os.makedirs("%s/voxceleb1_test" % (args.save_path), exist_ok=True)
        out = subprocess.call(
            "mv %s/wav %s/voxceleb1_test" % (args.save_path, args.save_path), shell=True
        )

        if args.vox2:
            os.makedirs("%s/voxceleb2_test" % (args.save_path), exist_ok=True)
            out = subprocess.call(
                "mv %s/aac %s/voxceleb2_test" % (args.save_path, args.save_path), shell=True
            )

    if args.convert:
        files = glob.glob("%s/voxceleb2_test/*/*/*/*.m4a" % args.save_path)
        files.sort()
        convert(args, files)
        files = glob.glob("%s/voxceleb2/*/*/*/*.m4a" % args.save_path)
        files.sort()
        convert(args, files)

    if args.make_test:
        args.save_path += "/asv_test"
        os.makedirs(args.save_path, exist_ok=True)
        if args.filter_dataset == "voxceleb1/,voxceleb2/":
            args.filter_dataset = "voxceleb1_test/"
        if args.vox2:
                args.filter_dataset = "voxceleb1/,voxceleb1_test/"
        make_test(args)
