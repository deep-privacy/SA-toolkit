import os
import sys
import tempfile
import random
import importlib.util
from types import SimpleNamespace


import torch
import torchaudio

from pkwrap.chain import ChainE2EModel
from pkwrap import TensorToKaldiMatrix
from pkwrap.script_utils import feat_writer

def load_wav(
    filepath: str,
    **kwargs,
):
    audio_frame, sampling_rate = torchaudio.load(filepath, **kwargs)
    if True: # normalize like kaldi
        audio_frame = audio_frame*2**15
    return audio_frame, sampling_rate

def get_extract_bn(device=torch.device("cuda")):
    pkwrap_path = "/lium/raid01_b/pchampi/lab/asr-based-privacy-preserving-separation/pkwrap/egs/librispeech/v1/"
    model = "local/chain/e2e/tuning/tdnnf.py"
    exp_path = "exp/chain/e2e_tdnnf/"
    model_weight = "final.pt"
    from_wav_cmvn = {
        "utt2spk": f"{pkwrap_path}/data/dev_clean_fbank_hires/utt2spk",
        "stats": f"{pkwrap_path}/data/dev_clean_fbank_hires/cmvn.scp",
        "filetype": "scp",
    }


    # load the pythom model
    config_path = pkwrap_path + model
    if not os.path.exists(config_path):
        raise FileNotFoundError("No file found at location {}".format(config_path))
    spec = importlib.util.spec_from_file_location("config", config_path)
    asr_model_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(asr_model_file)

    asr_net = asr_model_file.build(SimpleNamespace())
    pkwrapmodel = ChainE2EModel(asr_net, cmd_line=False,
                          **{
                              "dir": pkwrap_path + exp_path,
                              "base_model": pkwrap_path + exp_path + model_weight,
                              "from_wav_fbanks_conf": pkwrap_path + "configs/fbank_hires.conf", # kaldi Fbanks config file
                              #  "from_wav_cmvn": from_wav_cmvn,
                          })

    time_masking = torchaudio.transforms.TimeMasking(time_mask_param=5)
    freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)

    def data_aug(x, is_eval):
        out = x.permute(0, 2, 1)
        if not is_eval:
            out = freq_masking(out)
            out = time_masking(out)
        out = out.permute(0, 2, 1)
        return out

    net = pkwrapmodel.get_apply(device=device, data_aug=data_aug)
    def _apply(waveform, is_eval=False):
        #  waveform, _ = load_wav("/lium/raid01_b/pchampi/lab/asr-based-privacy-preserving-separation/pkwrap/egs/librispeech/v1/corpora/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac")
        #  waveform += 10e-6 * torch.randn(waveform.shape)
        #  waveform = waveform.to(device=device)
        #  key = "lbi-1272-128104-0000"

        key = "generic-spk"
        out, model = net(key, waveform, is_eval=is_eval)

        tmpname = next(tempfile._get_candidate_names())
        writer = feat_writer(f"ark,t:/tmp/asr_am_out_{tmpname}.ark")
        writer.Write("test_utts", TensorToKaldiMatrix(out[0].cpu()))
        writer.Close()
        torchaudio.save(f"/tmp/waveform_{tmpname}.wav", waveform[0].cpu().unsqueeze(0) / 2**15, 16000)
        print("== Example ASR decoding (No LM-rescoring): ==")
        print(f"cat /tmp/asr_am_out_{tmpname}.ark | {pkwrap_path}/shutil/decode/latgen-faster-mapped.sh {pkwrap_path}/exp/chain/e2e_biphone_tree/graph_tgsmall/words.txt {pkwrap_path}/exp/chain/e2e_tdnnf/0.trans_mdl exp/chain/e2e_biphone_tree/graph_tgsmall/HCLG.fst /tmp/decode_{tmpname}_lat.1.gz")
        sys.exit()

        out = model.bottleneck_out.permute(0, 2, 1).contiguous()
        return out

    return _apply

