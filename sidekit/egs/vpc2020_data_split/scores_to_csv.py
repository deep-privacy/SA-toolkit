#!/usr/bin/env python

import numpy as np
from collections import OrderedDict
import csv

filename = "results_allll.txt"

with open(filename) as f:
    content = f.read().splitlines()

all_res = {}

old_block = ""
block = ""
last_results = []
for line in content:
    if line.startswith("--p"):
        old_block = block
        block = line

    #  print(line)
    if old_block != block and old_block != "":
        if len(last_results) != 0:
            old_block = block

            arrs = np.array(last_results[0].split(" "))

            last_results = np.array(last_results)

            id = np.argwhere(arrs == "--pkwrap_vq_dim")[0][0]
            vq_dim = arrs[id + 1]

            id = np.argwhere(arrs == "--asv_model")[0][0]
            expe = arrs[id + 1].split("/")[-3]

            id = np.argwhere(
                last_results == "ASV: libri_test_enrolls - libri_test_trials_f"
            )[0][0]
            eer_f = last_results[id + 1].replace("EER_bootci: ", "").replace("interval: ", "")

            id = np.argwhere(
                last_results == "ASV: libri_test_enrolls - libri_test_trials_m"
            )[0][0]
            eer_m = last_results[id + 1].replace("EER_bootci: ", "").replace("interval: ", "")

            for x in last_results:
                if x.startswith("WER bootci dev_clean"):
                    dev_clean = " ".join(x.split("%")[1:]).replace("WER ", "").replace(" Conf Interval ", "").replace(" 95 "," ")
                if x.startswith("WER bootci test_clean"):
                    test_clean = " ".join(x.split("%")[1:]).replace("WER ", "").replace(" Conf Interval ", "").replace(" 95 "," ")
                if x.startswith("WER bootci test_other"):
                    test_other = " ".join(x.split("%")[1:]).replace("WER ", "").replace(" Conf Interval ", "").replace(" 95 "," ")

            if expe not in all_res:
                all_res[expe] = OrderedDict()

            if vq_dim not in all_res[expe]:
                all_res[expe][vq_dim] = []

            all_res[expe][vq_dim] += [
                eer_f,
                eer_m,
                dev_clean,
                test_clean,
                test_other,
            ]

            last_results = []

    last_results += [line]

with open(filename + ".csv", "w") as csvfile:
    spamwriter = csv.writer(
        csvfile, delimiter="|", quotechar="'", quoting=csv.QUOTE_MINIMAL
    )
    for k, row in all_res["libri460_fast2"].items():
        spamwriter.writerow([""] + row)
    for k, row in all_res["libri460_fast_vq"].items():
        spamwriter.writerow([k] + row)
    for k, row in all_res["libri460_fast_vq_spkdelta_l2norm"].items():
        spamwriter.writerow([k] + row)
    for k, row in all_res["libri460_fast_vq_spkdelta"].items():
        spamwriter.writerow([k] + row)
