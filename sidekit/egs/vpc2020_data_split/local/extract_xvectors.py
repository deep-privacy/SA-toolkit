import torch
import torchaudio
from sidekit.nnet.xvector import Xtractor

import os
import io
import argparse
import subprocess

import kaldiio
import soundfile

def read_wav_scp(wav_scp):
    """Reads wav.scp file and returns a dictionary

    Args:
        wav_scp: a string, contains the path to wav.scp

    Returns:
        utt2wav: a dictionary, keys are the first column of wav.scp
            and values are the second column
    """
    utt2wav = {}
    with open(wav_scp) as ipf:
        for line in ipf:
            lns = line.strip().split()
            uttname = lns[0]
            utt2wav[uttname] = lns[1:]
    return utt2wav

def prepare(wav):
    """Reads a wav.scp entry like kaldi with embeded unix command
    and returns a pytorch tensor like it was open with torchaudio.load()
    (within some tolerance due to numerical precision)

    signal, _ = torchaudio.load("XX/1272-128104-0000.flac")
    signalv2 = prepare(['flac', '-c', '-d', '-s', 'XX/1272-128104-0000.flac', "|"])
    signalv3 = prepare(['XX/1272-128104-0000.flac'])

    print("all close:", torch.allclose(signal, signalv2, rtol=1e-1))
    print("all close:", torch.allclose(signal, signalv3, rtol=1e-1))

    Args:
        wav: a list containing the scp entry

    Returns:
        feats_torch torch.tensor dtype float32
    """
    wav = ' '.join(wav)
    if wav.strip().endswith("|"):
        devnull = open(os.devnull, 'w')
        try:
            wav_read_process = subprocess.Popen(
                wav.strip()[:-1],
                stdout=subprocess.PIPE,
                shell=True,
                stderr=devnull
            )
            sample, _ = soundfile.read(
                io.BytesIO(wav_read_process.communicate()[0]),
            )
        except Exception as e:
            raise IOError("Error processing wav file: {}\n{}".format(wav, e))
    else:
        sample, _ = soundfile.read(wav)
    feats_torch = torch.tensor(sample, dtype=torch.float32, requires_grad=False)
    return feats_torch

def load_model(model_path, device):
    device = torch.device(device)
    model_config = torch.load(model_path, map_location=device)

    model_opts = model_config["model_archi"]
    if "embedding_size" not in model_opts:
        model_opts["embedding_size"] = 256
    xtractor = Xtractor(model_config["speaker_number"],
                     model_archi=model_opts["model_type"],
                     loss=model_opts["loss"]["type"],
                     embedding_size=model_opts["embedding_size"])

    xtractor.load_state_dict(model_config["model_state_dict"], strict=True)
    xtractor = xtractor.to(device)
    xtractor.eval()
    return xtractor

@torch.no_grad()
def main(xtractor, kaldi_wav_scp, out_file, device):
    device = torch.device(device)

    utt2wav = read_wav_scp(kaldi_wav_scp)
    out_ark = os.path.realpath(os.path.join(os.path.dirname(out_file), os.path.splitext(os.path.basename(out_file))[0]))

    with kaldiio.WriteHelper(f'ark,scp:{out_ark}.ark,{os.path.realpath(out_file)}') as writer:
        for key, wav in utt2wav.items():
            signal = prepare(wav)
            signal = signal.to(device)
            signal = torch.unsqueeze(signal, 0)
            _, vec = xtractor(signal, is_eval=True)
            writer(key, vec.detach().cpu().numpy())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract the x-vectors given a sidekit model")
    parser.add_argument("--model", type=str, help="SideKit model", required=True)
    parser.add_argument("--wav-scp", type=str, required=True)
    parser.add_argument("--out-scp", type=str, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str, help="The device (cpu or cuda:0) to run the inference")
    args = parser.parse_args()

    assert os.path.isfile(args.model), "NO SUCH FILE: %s" % args.model
    assert os.path.isfile(args.wav_scp), "NO SUCH FILE: %s" % args.wav_scp
    assert os.path.isdir(os.path.dirname(args.out_scp)), "NO SUCH DIRECTORY: %s" % args.out_scp
    # If cuda device is requested, check if cuda is available
    args.device = args.device.strip().lower()
    if args.device == "cuda":
        assert torch.cuda.is_available(), "CUDA is not available, check configuration or run on cpu (--device cpu)"
    xtractor = load_model(args.model, args.device)
    main(xtractor, args.wav_scp, args.out_scp, args.device)
