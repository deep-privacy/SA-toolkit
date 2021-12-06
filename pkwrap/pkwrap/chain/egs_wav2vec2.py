"""Module to prepare egs for e2e-lfmmi training with wav2vec2 models"""
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

import logging
import os
from collections import defaultdict
import subprocess
import io
import random
import torch
import soundfile

try:
    from _pkwrap import kaldi  # lazy import (kaldi-free decoding)
except ImportError as error:
    pass


class WavInfo(object):
    """WavInfo objects hole information about each example without the supervision fst graph"""

    def __init__(self, name, wav):
        self.name = name
        self.wav = wav

    def prepare(self):
        return prepare_e2e(self)


class EgsInfo(object):
    """EgsInfo objects hole information about each example"""

    def __init__(self, name, wav, fstscp, num_output_frames):
        self.name = name
        self.wav = wav
        self.num_output_frames = int(num_output_frames)
        self.supervision = None
        self.fstscp = fstscp

    def create_supervision(self, trans_mdl):
        """Creates supervision object given transition model

        Calls TrainingGraphToSupervisionE2e from Kaldi library

        Args:
            trans_mdl: Transition model object

        Returns:
            None
        """
        self.fst = kaldi.fst.StdVectorFst()
        kaldi.fst.ReadFstKaldi(self.fstscp, self.fst)

        self.supervision = kaldi.chain.Supervision()
        kaldi.chain.TrainingGraphToSupervisionE2e(
            self.fst, trans_mdl, self.num_output_frames, self.supervision
        )

    # TODO(srikanth): check if the supervision is already normalized to avoid
    # applying normalization more than once
    def normalize_supervision(self, normalize_fst):
        """Normalize supervision with fst"""
        kaldi.chain.AddWeightToSupervisionFst(normalize_fst, self.supervision)


def prepare_e2e(egs):
    """Reads a wav.scp entry like kaldi with embeded unix command
     and returns a pytorch tensor like it was open with torchaudio.load()
     (within some tolerance due to numerical precision)
     And a merged supervision for LF-MMI training

     signal, _ = torchaudio.load("XX/1272-128104-0000.flac")
     signalv2 = prepare(['flac', '-c', '-d', '-s', 'XX/1272-128104-0000.flac', "|"])
     signalv3 = prepare(['XX/1272-128104-0000.flac'])
     print("all close:", torch.allclose(signal, signalv2, rtol=1e-1))
     print("all close:", torch.allclose(signal, signalv3, rtol=1e-1))

     Args:
         wav: a list containing the scp entry
     Returns:
         feats: a Tensor of size  time x dimension
         sup: Supervision object to be used as target for LF-MMI

    Raises:
        IOError: when something wrong while read a file
    """
    # load the audio
    if not egs:
        return None, None
    devnull = open(os.devnull, "w")
    if len(egs.wav) == 1:
        samples, _ = soundfile.read(" ".join(egs.wav))
    else:
        try:
            wav_read_process = subprocess.Popen(
                " ".join(egs.wav), stdout=subprocess.PIPE, shell=True, stderr=devnull
            )
            samples, _ = soundfile.read(io.BytesIO(wav_read_process.communicate()[0]))
        except Exception:
            raise IOError("Error processing {}".format(egs.name))
    feats_torch = torch.tensor(samples, dtype=torch.float32, requires_grad=False)
    return (feats_torch, egs)


def GetSupervisionFromWav2Vec2Egs(
    transition_model, normalization_fst, egs_list, num_output_frames
):
    batch = []
    for item in egs_list:
        item.num_output_frames = num_output_frames
        item.create_supervision(transition_model)
        item.normalize_supervision(normalization_fst)
        batch.append(item.supervision)
    merged_sup = kaldi.chain.Supervision()
    kaldi.chain.MergeSupervisionE2e(batch, merged_sup)
    return merged_sup


def Wav2vec2EgsCollectFn(batch):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        lengths = torch.tensor([t.shape[0] for t in batch])
        if lengths.sum().item() != batch[0].shape[0] * len(batch):
            #  logging.warning("Padding tensor lengths={}".format(str(lengths)))
            return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    if isinstance(elem, EgsInfo):
        return list(batch)
    if isinstance(elem, WavInfo):
        return list(batch)
    if isinstance(elem, str):
        return list(batch)
    else:
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        res = [Wav2vec2EgsCollectFn(samples) for samples in transposed]
        return res
    raise TypeError(
        "default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found {}".format(
            elem_type
        )
    )


class Wav2vec2BatchSampler(
    torch.utils.data.BatchSampler
):  # pylint: disable=too-few-public-methods
    """An extension of BatchSampler to handle raw egs"""

    def __iter__(self):
        batch_by_length = defaultdict(list)
        # for i in torch.utils.data.RandomSampler(range(len(self.sampler))):
        for i in range(len(self.sampler)):
            idx = self.sampler[i]
            num_output_frames = idx.num_output_frames
            batch_by_length[num_output_frames].append(i)
            if len(batch_by_length[num_output_frames]) == self.batch_size:
                yield batch_by_length[num_output_frames]
                batch_by_length[num_output_frames] = []
        for num_output_frames in batch_by_length:
            if batch_by_length[num_output_frames] and not self.drop_last:
                yield batch_by_length[num_output_frames]


class Wav2vec2DecodeDataset(torch.utils.data.Dataset):
    """A Pytorch Dataset class to prepare wav for Wav2vec2 decoding"""

    def __init__(self, wav_spc_file):
        utt2wav = Wav2vec2EgsDataset.read_wav_scp(wav_spc_file)
        self.holder = []
        for key, wavfile in utt2wav.items():
            self.holder.append(WavInfo(key, wavfile))

    def __len__(self):
        return len(self.holder)

    def __getitem__(self, idx):
        feats, info = self.holder[idx].prepare()
        return feats, info.name

    def __item__(self, i):
        return self.holder[i]

    @classmethod
    def from_wav_scp(cls, wav_scp_file):
        return cls(wav_scp_file)


class Wav2vec2EgsDataset(torch.utils.data.Dataset):
    """A Pytorch Dataset class to prepare Egs for Wav2vec2 models training"""

    # TODO(srikanth): should reduce the number of parameters or reconfigure pylint
    def __init__(
        self,
        wav,
        fst_file,
        utt2len_file,
        transition_model_filename,
        normalization_fst_rxfilename,
        shuffle=False,
    ):
        """instantiates a Pytorch Dataset for E2E training

        Args:
            wav: the file with wav.scp
            fst_file: fst_file (e2e_biphone_tree/fst.{number}.scp)
            transition_model: transition model that maps transition ids to pdf ids
            normalization_fst: fst to normalize when supervision is created
        """
        self.transition_model = kaldi.hmm.TransitionModel()
        kaldi.hmm.ReadTransitionModel(
            self.transition_model,
            transition_model_filename,
        )
        self.normalization_fst = kaldi.fst.StdVectorFst()
        kaldi.fst.ReadFstKaldi(normalization_fst_rxfilename, self.normalization_fst)
        self.prepare_egs(wav, fst_file)
        if shuffle:
            random.shuffle(self.egs_holder)

    def __len__(self):
        return len(self.egs_holder)

    def __getitem__(self, idx):
        return prepare_e2e(self.egs_holder[idx])

    def __item__(self, i):
        return self.egs_holder[i]

    def prepare_egs(self, wav, fst_file):
        """Method to prepare egs_holder"""
        # egs file is wav.scp file
        utt2wav = Wav2vec2EgsDataset.read_wav_scp(wav)
        utt2len = Wav2vec2EgsDataset.read_utt2len_file(
            "{}/utt2len".format(os.path.dirname(wav))
        )
        self.egs_holder = self.get_egs_holder(fst_file, utt2wav, utt2len)

    @staticmethod
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
                lns = line.strip().rstrip("|").split()
                uttname = lns[0]
                utt2wav[uttname] = lns[1:]
        return utt2wav

    @staticmethod
    def read_utt2len_file(utt2len_file):
        """read utt2len file, second column is the number of output frames"""
        utt2len = {}
        with open(utt2len_file) as utt2len_f:
            for line in utt2len_f:
                lns = line.strip().split()
                utt2len[lns[0]] = float(lns[1])
        return utt2len

    def get_egs_holder(self, fst_file, utt2wav, utt2len):
        """Populate egs_holder"""
        egs_holder = []
        with open(fst_file) as ipf:
            total, done, skipped = 0, 0, 0
            for line in ipf:
                lns = line.strip().split()
                total += 1
                try:
                    uttname, fstscp = lns
                except Exception:
                    logging.error(
                        "Excepted fst file %s to have only 2 columns", fst_file
                    )
                if uttname not in utt2wav:
                    skipped += 1
                    continue
                if uttname not in utt2len:
                    logging.warning(
                        "Cannot find number of output frames for %s", uttname
                    )
                    skipped += 1
                    continue

                num_output_frames = utt2len[uttname]

                fst = kaldi.fst.StdVectorFst()
                kaldi.fst.ReadFstKaldi(fstscp, fst)
                min_path_length = kaldi.chain.FindMinimumLengthPathFromFst(fst)
                if min_path_length > num_output_frames:
                    logging.warning(
                        "get_egs_holder, %s has more labels than frames",
                        this_egs_info.name,
                    )
                    skipped += 1
                    continue

                this_egs_info = EgsInfo(
                    uttname, utt2wav[uttname], fstscp, num_output_frames
                )
                egs_holder.append(this_egs_info)
                done += 1
            logging.info(
                "In get_egs_holder: total=%d, done=%d, skipped=%d", total, done, skipped
            )
        return egs_holder
