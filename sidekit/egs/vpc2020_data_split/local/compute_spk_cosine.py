import os
import argparse
import kaldiio
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve
import numpy as np

def read_utt2spk_file(utt2spk_file):
    """read utt2spk file, second column is the speaker id"""
    utt2spk = {}
    with open(utt2spk_file) as utt2spk_f:
        for line in utt2spk_f:
            lns = line.strip().split()
            utt2spk[lns[0]] = lns[1]
    return utt2spk


def cosine_scoring(embd1s, embd2s):
    scores = []
    for embd1, embd2 in zip(embd1s, embd2s):
        # Multiplying by -1 to ensure compatibility with affinity
        # Now lower value will indicate less affinity as compared
        # to original cosine distance
        score = 1 - cosine(embd1, embd2)
        scores.append(score)
    return scores


def main(args):
    trials = [x.split() for x in open(args.trials)]
    utt1s = [x[0] for x in trials]
    utt2s = [x[1] for x in trials]

    with kaldiio.ReadHelper(f'scp:{args.enroll_scp}') as reader:
        utt2embd_enroll = {utt:embd for utt, embd in reader}


    with kaldiio.ReadHelper(f'scp:{args.trial_scp}') as reader:
        utt2embd_trial = {utt:embd for utt, embd in reader}

    # Average the utterance-level xvectors to get speaker-level xvectors.
    ##
    utt2spk = read_utt2spk_file(args.enroll_utt2spk)

    # reverse utt2spk
    spk2utt = {}
    for k, v in utt2spk.items():
        spk2utt[v] = spk2utt.get(v, []) + [k]
    
    utt2embd_enroll_mean = {}
    for spk, uttrs in spk2utt.items():
        mean = np.mean([utt2embd_enroll[utt] for utt in uttrs], axis=0)
        norm = np.linalg.norm(mean, ord=2)
        mean /= norm
        utt2embd_enroll_mean[spk] = mean

    utt2embd_enroll_mean = [utt2embd_enroll_mean[utt] for utt in utt1s]
    utt2embd_trial = [utt2embd_trial[utt] for utt in utt2s]

    scores = cosine_scoring(utt2embd_enroll_mean, utt2embd_trial)
    score_file_kaldi = []
    for enroll, trial, score in zip(utt1s, utt2s, scores):
        score_file_kaldi.append([enroll, trial, str(score)])

    with open(args.output, "w") as txt_file:
        for line in score_file_kaldi:
            txt_file.write(" ".join(line) + "\n") # works with any number of elements in a line



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Speaker Verification Trials/Enroll Cosine Calculation.')

    # Kaldi trials files
    parser.add_argument('trials')

    #  utt2spk to compute the average speaker-level xvectors
    parser.add_argument('enroll_utt2spk')

    # Kaldi scp files
    parser.add_argument('trial_scp')
    parser.add_argument('enroll_scp')

    parser.add_argument('output')

    args = parser.parse_args()

    assert os.path.isfile(args.trials), "NO SUCH FILE: %s" % args.trials
    assert os.path.isfile(args.enroll_utt2spk), "NO SUCH FILE: %s" % args.enroll_utt2spk
    assert os.path.isfile(args.enroll_scp), "NO SUCH FILE: %s" % args.enroll_scp
    assert os.path.isfile(args.trial_scp), "NO SUCH FILE: %s" % args.trial_scp
    main(args)
