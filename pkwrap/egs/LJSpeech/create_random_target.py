import argparse
import pkwrap
import json
import random
import sys

if __name__ == "__main__":
    """
    python3 ~/lab/asr-based-privacy-preserving-separation/pkwrap/egs/LJSpeech/create_random_target.py --target-list ~/lab/asr-based-privacy-preserving-separation/pkwrap/egs/LJSpeech/data/LibriTTS/stats.json --in-wavscp ./data/train-clean-100/wav.scp --in-utt2spk ./data/train-clean-100/utt2spk > ./data/train-clean-100/target-mapping
    """

    random.seed(52)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-list",
        type=str,
        dest="target_list",
    )
    parser.add_argument("--in-wavscp", type=str, dest="_in_scp", default=None)
    parser.add_argument("--in-utt2spk", type=str, dest="_in_utt2spk", default=None)

    args = parser.parse_args()

    stats = open(args.target_list).readline()
    target_spkids = list(json.loads(stats).keys())
    target_spkids.sort()
    random.shuffle(target_spkids)

    _target_spkids = target_spkids.copy()

    utt2spk = pkwrap.utils.kaldi.read_wav_scp(args._in_utt2spk)
    source_spkid = list(set(utt2spk.values()))
    source_spkid.sort()

    #  print(source_spkid)
    #  print(target_spkids)
    wavs_scp = pkwrap.utils.kaldi.read_wav_scp(args._in_scp)

    source_spkid_2_target_spkid = {}
    for k, v in wavs_scp.items():
        if utt2spk[k] not in source_spkid_2_target_spkid:
            if len(target_spkids) == 0:
                target_spkids = _target_spkids.copy()
            source_spkid_2_target_spkid[utt2spk[k]] = random.choice(target_spkids)
            target_spkids.remove(source_spkid_2_target_spkid[utt2spk[k]])

        print(k, source_spkid_2_target_spkid[utt2spk[k]])

    print(source_spkid_2_target_spkid, file=sys.stderr)
