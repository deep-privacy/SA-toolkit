#!/usr/bin/env python3.0
# -*- coding: utf-8 -*-

import os
import shutil
import multiprocessing
from pathlib import Path
import torchaudio
import random
import glob
import logging

import torch

import satools.script_utils as script_utils
from satools.infer_helper import load_model
from satools.utils.kaldi import load_wav_from_scp

def copy_data_dir(dataset_path, output_path):
    # Copy utt2spk wav.scp and so on, but not the directories inside (may contains clear or anonymzied *.wav)
    os.makedirs(output_path, exist_ok=True)
    for p in glob.glob(str(Path(dataset_path) / '*'), recursive=False):
        if os.path.isfile(p):
            shutil.copy(p, output_path)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, id_wavs, get_f0_func):
        self.all_wavs = list(id_wavs.values())
        self.all_keys = list(id_wavs.keys())
        self.get_f0_func = get_f0_func

    def __len__(self):
        return len(self.all_wavs)

    def __getitem__(self, index):
        audio, freq = load_wav_from_scp(str(self.all_wavs[index]))
        f0 = self.get_f0_func(audio)
        return {"utid": self.all_keys[index],
                "audio": audio,
                "f0": f0,
                "freq": freq}

def collate_fn(item_list):
    batch_size = len(item_list)

    data_list_audio = [i['audio'] for i in item_list]
    lengths_tensor_audio = torch.tensor([i.shape[-1] for i in data_list_audio])
    max_len_audio = torch.max(lengths_tensor_audio).item()
    output_audio = torch.zeros([batch_size, max_len_audio])
    for i in range(batch_size):
        cur = data_list_audio[i]
        cur_len = data_list_audio[i].shape[-1]
        output_audio[i, :cur_len] = cur.squeeze()

    data_list_f0 = [i['f0'] for i in item_list]
    lengths_tensor_f0 = torch.tensor([i.shape[-1] for i in data_list_f0])
    max_len_f0 = torch.max(lengths_tensor_f0).item()
    output_f0 = torch.zeros([batch_size, max_len_f0])
    for i in range(batch_size):
        cur = data_list_f0[i]
        cur_len = data_list_f0[i].shape[-1]
        output_f0[i, :cur_len] = cur.squeeze()

    utids = [i['utid'] for i in item_list]
    freqs = [i['freq'] for i in item_list]
    return output_audio, output_f0, lengths_tensor_audio, utids, freqs

def process_data(dataset_path: str, target_selection_algorithm: str, wavscp: dict, settings: dict, progress):
    results_dir = settings.results_dir
    dataset_path = Path(str(dataset_path))
    output_path = Path(str(dataset_path) + settings.new_datadir_suffix)
    device = settings.device
    batch_size = settings.batch_size

    copy_data_dir(dataset_path, output_path)
    results_dir = output_path / results_dir
    os.makedirs(results_dir, exist_ok = True)

    wav_scp = dataset_path / 'wav.scp'
    utt2spk = dataset_path / 'utt2spk'
    wav_scp_out = output_path / 'wav.scp'

    option_args = {}
    if settings.f0_modification != "":
        option_args["f0_transformation"] = settings.f0_modification
    with progress.get_lock():
        model = load_model(settings.model, option_args=option_args)
    model.to(device)
    model.eval()
    possible_targets = None
    if hasattr(model, "spk"):
        possible_targets = model.spk.copy() # For spk and utt target_selection_algorithm random choice
    else:
        logging.info("Model without explicit target")

    source_utt2spk = script_utils.read_wav_scp(utt2spk)
    out_spk2target = {} # For spk target_selection_algorithm


    @torch.no_grad()
    def process_wav(utid, freq, audio, f0, original_len):

        freq = freq[0] # assume all freq = in same batch (and so dataset)
        audio = audio.to(device)

        # Anonymize function
        model.set_f0(f0.to(device)) # CPU extracted by Dataloader (num_workers)
        #  Batch select target spks from the available model list depending on target_selection_algorithm
        target_spks = []
        if target_selection_algorithm == "constant": # The best way/most secure to evaluate privacy when applied to all dataset (train included)
            target_constant_spkid = settings.target_constant_spkid # For constant target_selection_algorithm
            target_spks = [target_constant_spkid]*audio.shape[0]
        elif target_selection_algorithm == "none":
            pass
        elif target_selection_algorithm == "bad_for_evaluation":
            # This target selection algorithm is bad for evaluation as it does
            # not generate suitable training data for the ASV eval training
            # procedure. Use it with caution.
            for ut in utid:
                source_spk = source_utt2spk[ut]
                if source_spk not in out_spk2target:
                    out_spk2target[source_spk] = random.sample(possible_targets, 2)
                target_spks.append(random.choice(out_spk2target[source_spk]))
        elif target_selection_algorithm == "random_per_utt":
            target_spks = []
            for ut in utid:
                target_spks.append(random.choice(possible_targets))
        elif target_selection_algorithm == "random_per_spk_uniq":
            for ut in utid:
                source_spk = source_utt2spk[ut]
                if source_spk not in out_spk2target:
                    out_spk2target[source_spk] = random.choice(possible_targets)
                    # Remove target spk: size of possible source spk to anonymize == len(possible_targets) (==247) or you need to add spk target overlap)
                    possible_targets.remove(out_spk2target[source_spk])
                target_spks.append(out_spk2target[source_spk])
        elif target_selection_algorithm == "random_per_spk":
            for ut in utid:
                source_spk = source_utt2spk[ut]
                if source_spk not in out_spk2target:
                    out_spk2target[source_spk] = random.choice(possible_targets)
                target_spks.append(out_spk2target[source_spk])
        else:
            raise ValueError(f"{target_selection_algorithm} not implemented")
        targets_arg = {}
        if len(target_spks) != 0:
            targets_arg = {"target":target_spks}
        #  Batch conversion
        wav_conv = model.convert(audio, **targets_arg)
        wav_conv = wav_conv.cpu()

        def parallel_write():
            for i in range(wav_conv.shape[0]):
                wav = wav_conv[i]
                if len(wav.shape) == 1:
                    wav = wav.unsqueeze(0) # batch == 1 -> len(dst) % batch == 1
                wav = wav[:, :original_len[i]]
                # write to buffer
                u = utid[i]
                output_file = results_dir / f'{u}.wav'
                torchaudio.save(str(output_file), wav, freq, encoding='PCM_S', bits_per_sample=16)
        p = multiprocessing.Process(target=parallel_write, args=())
        p.start()
        return p

    nj = settings.data_loader_nj
    nj = min(nj, 18)
    p = None

    with open(wav_scp_out, 'wt', encoding='utf-8') as writer:
        filtered_wavs = {}
        for u, file in wavscp.items():
            output_file = results_dir / f'{u}.wav'
            filtered_wavs[u] = file

        data_loader = torch.utils.data.DataLoader(Dataset(filtered_wavs, model.get_f0), batch_size=batch_size, num_workers=nj, collate_fn=collate_fn)
        for audio, f0, original_len, utid, freq in data_loader:
            p = process_wav(utid, freq, audio, f0, original_len)
            for u in utid:
                output_file = results_dir / f'{u}.wav'
                writer.writelines(f"{u} {output_file}\n")
            with progress.get_lock():
                progress.value += batch_size
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    # wait for last p to write the anonymized audios
    if p:
        p.join()
