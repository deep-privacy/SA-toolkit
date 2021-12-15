# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

"""
 wrap scripts in kaldi utils directory
"""

import subprocess
from .. import script_utils
import sys
import os
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import logging

run = script_utils.run


def split_data(dirname, num_jobs=0):
    """Call's Kaldi's utils/split_data.sh script

    Args:
        dirname: feature directory to be split
        num_jobs: number of splits
    """
    if num_jobs == 0:
        spk2utt_filename = os.path.join(dirname, "spk2utt")
        num_jobs = num_lines(spk2utt_filename)
    script_utils.run(["utils/split_data.sh", dirname, str(num_jobs)])
    return int(num_jobs)


def num_lines(filename):
    """Find out the number of lines in a file using wc

    This function uses subprocess to run the wc command.
    It quits if the return code from subprocess is non-zero.

    Args:
        filename: a string containing the path of the file

    Returns:
        An integer: the number of lines in the file.
    """
    try:
        p = subprocess.check_output(["wc", "-l", filename])
        return int(p.decode().strip().split()[0])
    except subprocess.CalledProcessError as cpe:
        sys.exit(cpe.returncode)


def make_soft_link(src, dst, relative=False, extra_opts=[]):
    """Create soft link using ln -r -s command

    The function calls sys.exit(1) if execution fails

    Args:
        src: source file
        dst: destination file
        relative: set to True if link is relative
        extra_opts: other options to be passed to ln
    """
    try:
        cmd = ["ln"]
        if relative:
            cmd.append("-r")
        cmd.append("-s")
        if extra_opts:
            cmd += extra_opts
        cmd += [src, dst]
        p = subprocess.run(cmd)
        if p.returncode != 0:
            sys.exit(p.returncode)
    except:
        sys.exit(1)


def touch_file(file_path):
    """Touch a file

    This function calls the touch command and quits if the call fails.
    """
    try:
        subprocess.run(["touch", file_path])
    except:
        sys.exit(1)


def cat(file_list, out_file):
    with open(out_file, "w") as opf:
        subprocess.run(["cat", *file_list], stdout=opf)


def creation_date_file(file):
    created = os.stat(file).st_ctime
    return datetime.fromtimestamp(created)


def scans_directory_for_ext(root_data, extension):
    logging.info(f"Locating {extension}(s)")
    wavs_path = []
    wav_count = 0
    pbar = tqdm(os.walk(root_data))
    for root, dirs, files in pbar:
        if Path(root).parent == Path(root_data):
            dataset = root.split("/")[-1]
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.splitext(file_path)[1] == f".{extension}":
                wav_count += 1
                pbar.set_description(f"file count : {wav_count}")
                wavs_path.append(file_path)

    return wavs_path
