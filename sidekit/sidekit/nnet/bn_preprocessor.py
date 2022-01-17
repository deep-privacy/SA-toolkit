import os
import sys
import tempfile
import random
import importlib.util
from types import SimpleNamespace
import random


import torch
import torchaudio

import pkwrap
import kaldiio

ECHO_TIMES = 0


def load_wav(
    filepath: str,
    **kwargs,
):
    audio_frame, sampling_rate = torchaudio.load(filepath, **kwargs)
    return audio_frame, sampling_rate


def get_extract_bn(device=torch.device("cuda")):
    pkwrap_path = pkwrap.__path__[0] + "/../egs/librispeech/v1/"
    model = os.getenv("pkwrap_model", "local/chain/e2e/tuning/tdnnf.py")
    exp_path = os.getenv("pkwrap_exp_dir", "exp/chain/e2e_tdnnf/")
    model_weight = "final.pt"

    # load the pythom model
    config_path = pkwrap_path + model
    if not os.path.exists(config_path):
        raise FileNotFoundError("No file found at location {}".format(config_path))
    spec = importlib.util.spec_from_file_location("config", config_path)
    asr_model_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(asr_model_file)

    args = SimpleNamespace()
    if int(os.getenv("pkwrap_vq_dim", "-1")) != -1:
        args = SimpleNamespace(
            freeze_encoder=True,
            codebook_size=int(os.getenv("pkwrap_vq_dim", "-1")),
        )

    asr_net = asr_model_file.build(args)
    pkwrap_chain = pkwrap.chain.ChainE2EModel(
        asr_net,
        cmd_line=False,
        **{
            "dir": pkwrap_path + exp_path,
            "base_model": pkwrap_path + exp_path + model_weight,
        },
    )

    time_masking = torchaudio.transforms.TimeMasking(time_mask_param=5)
    freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)

    def spec_augment(x):
        out = x.permute(0, 2, 1)
        out = freq_masking(out)
        out = time_masking(out)
        out = out.permute(0, 2, 1)
        return out

    net = pkwrap_chain.get_forward(device=device)

    def _apply(waveform, is_eval=False):

        data_aug = spec_augment
        if is_eval:
            data_aug = lambda x: x

        out, model = net(waveform, data_aug)

        global ECHO_TIMES
        if ECHO_TIMES < 3 and random.randint(1, 100) == 1:
            ECHO_TIMES += 1
            tmpname = next(tempfile._get_candidate_names())
            writer = kaldiio.WriteHelper(f"ark,t:/tmp/asr_am_out_{tmpname}.ark")
            writer("test_utts", out[0].cpu().numpy())
            writer.close()
            torchaudio.save(
                f"/tmp/waveform_{tmpname}.wav",
                waveform[0].cpu().unsqueeze(0) / 2 ** 15,
                16000,
            )
            print("== Example ASR decoding (No LM-rescoring): ==", file=sys.stderr)
            print(
                f"cat /tmp/asr_am_out_{tmpname}.ark | {pkwrap_path}/shutil/decode/latgen-faster-mapped.sh {pkwrap_path}/exp/chain/e2e_biphone_tree/graph_tgsmall/words.txt {pkwrap_path}/exp/chain/e2e_tdnnf/0.trans_mdl exp/chain/e2e_biphone_tree/graph_tgsmall/HCLG.fst /tmp/decode_{tmpname}_lat.1.gz",
                file=sys.stderr,
            )
        #  sys.exit()

        out = model.bottleneck_out.permute(0, 2, 1).contiguous()
        return out

    return _apply
