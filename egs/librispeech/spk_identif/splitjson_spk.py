#!/usr/bin/env python
# encoding: utf-8

'''
splits json file based on speakers
'''

from __future__ import print_function
from __future__ import division

import argparse
import json
import logging
import os
import sys

from random import shuffle

import numpy as np
import itertools

def flatten(l):
    return sorted(list(itertools.chain.from_iterable(l)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json', type=str,
                        help='json file')
    parser.add_argument('--dev', '-d', type=int,
                        help='Number of utterances to be assigned for dev',
                        default=2)
    parser.add_argument('--test', '-t', type=int,
                        help='Number of utterances to be assigned for test',
                        default=2)
    parser.add_argument('--filter', '-f', type=str,
                        help="Filter(keep) the input speakers based on this spk2gender file",
                        nargs="?",
                        required=False,
                        default="")
    args = parser.parse_args()

    # logging info
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    # check directory
    filename = os.path.basename(args.json).split('.')[0]
    dirname = os.path.dirname(args.json)
    dirname = 'dump/split_utt_spk'.format(dirname)
    logging.info("Writing splits in '%s'" % dirname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # check for filter
    to_keep_spk = set()
    if args.filter != "":
        spk2gender_lines = [
            line.rstrip("\n").split("-")[0]
            for line in open(args.filter)
        ]
        spk_filter = set(spk2gender_lines)
        logging.info(f"filtering {len(spk_filter)} speakers from {args.json}")

    # load json and split keys
    j = json.load(open(args.json))
    utt_ids = j['utts'].keys()
    logging.info("number of utterances = %d" % len(utt_ids))
    spk2utt = {}
    for utt_id, utt_obj in j['utts'].items():
        spkid = utt_obj['utt2spk'].split("-")[0]
        if spkid not in spk2utt:
            spk2utt[spkid] = []
        spk2utt[spkid].append(utt_id)
    logging.info("number of speakers = %d", len(spk2utt))

    if args.filter != "":
        spk2utt = dict([x for x in spk2utt.items() if x[0] in spk_filter])
        logging.info("number of speakers after filter = %d", len(spk2utt))

    num_utt = [len(x) for x in spk2utt.values()]
    logging.info("minimum number of utt per spk = %d", min(num_utt)) # -> MGreenberg
    logging.info("maximum number of utt per spk = %d", max(num_utt))

    # Randomly shuffle the utterance ids for each speaker
    for k, v in spk2utt.items():
        shuffle(v)
        spk2utt[k] = v

    dev_uttids = flatten([x[-args.dev:] for x in spk2utt.values()])
    test_uttids = flatten([x[-args.dev-args.test:-args.dev] for x in spk2utt.values()])
    train_uttids = flatten([x[:-args.dev-args.test] for x in spk2utt.values()])
    logging.info("Train = %d, Test = %d, Dev = %d", len(train_uttids), len(test_uttids), len(dev_uttids))


    with open('{}/{}.{}.json'.format(dirname, filename, 'train'), "wb+") as f:
        new_dic = {}
        for utt_id in train_uttids:
            new_dic[utt_id] = j['utts'][utt_id]
        jsonstr = json.dumps({'utts': new_dic},
                              indent=4,
                              ensure_ascii=False,
                              sort_keys=True)
        f.write(jsonstr.encode('utf_8'))

    with open('{}/{}.{}.json'.format(dirname, filename, 'test'), "wb+") as f:
        new_dic = {}
        for utt_id in test_uttids:
            new_dic[utt_id] = j['utts'][utt_id]
        jsonstr = json.dumps({'utts': new_dic},
                              indent=4,
                              ensure_ascii=False,
                              sort_keys=True)
        f.write(jsonstr.encode('utf_8'))

    with open('{}/{}.{}.json'.format(dirname, filename, 'dev'), "wb+") as f:
        new_dic = {}
        for utt_id in dev_uttids:
            new_dic[utt_id] = j['utts'][utt_id]
        jsonstr = json.dumps({'utts': new_dic},
                              indent=4,
                              ensure_ascii=False,
                              sort_keys=True)
        f.write(jsonstr.encode('utf_8'))
