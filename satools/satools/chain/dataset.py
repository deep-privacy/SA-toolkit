"""Module to prepare egs for e2e-lfmmi training models"""

import io
import logging
import os
import subprocess
import time
from collections import defaultdict

import torch
import torchaudio

from .. import augmentation
from .. import utils

try:
    from _satools import kaldi  # lazy import (kaldi-free import)
except ImportError as error:
    pass


class EgsInfo(utils.WavInfo):
    """EgsInfo objects hole information about each example"""

    def __init__(self, wav, name, filename, fstscp, num_output_frames):
        super().__init__(wav, name, filename)
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


def GetSupervisionFromEgs(
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


def EgsCollectFn(batch):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        lengths = [t.shape[0] for t in batch]
        if not all(element == lengths[0] for element in lengths):
            #  logging.warning("Padding tensor lengths={}".format(str(lengths)))
            return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem._typed_storage()._new_shared(numel, device=elem.device)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    if isinstance(elem, EgsInfo):
        return list(batch)
    if isinstance(elem, utils.WavInfo):
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
        res = [EgsCollectFn(samples) for samples in transposed]
        return res
    raise TypeError(
        "default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found {}".format(
            elem_type
        )
    )


class BatchSampler(torch.utils.data.BatchSampler):
    """Samples batch indices from sequence-length buckets efficiently
    with very little memory overhead."""

    def __init__(self, sampler, batch_size, drop_last, ran_iter=False, allow_some_padding=False):
        super().__init__(sampler, batch_size, drop_last)
        self.ran_iter = ran_iter
        self.allow_some_padding = allow_some_padding

    def __iter__(self):
        batch_by_length = defaultdict(list)
        if self.ran_iter == True:
            iterator = torch.utils.data.RandomSampler(range(len(self.sampler)))
        else:
            iterator = range(len(self.sampler))
        for i in iterator:
            idx = self.sampler[i]
            num_output_frames = idx.num_output_frames
            if self.allow_some_padding:
                num_output_frames = int(num_output_frames / 199)
            batch_by_length[num_output_frames].append(i)
            if len(batch_by_length[num_output_frames]) == self.batch_size:
                yield batch_by_length[num_output_frames]
                batch_by_length[num_output_frames] = []
        for num_output_frames in batch_by_length:
            if batch_by_length[num_output_frames] and not self.drop_last:
                yield batch_by_length[num_output_frames]


class EgsDataset(torch.utils.data.Dataset):
    """A Pytorch Dataset class to prepare EgsInfo training"""

    def __init__(
        self,
        wav_scp_file,
        fst_file,
        utt2len_file,
        transition_model_filename,
        normalization_fst_rxfilename,
        augmentation={},
    ):
        """instantiates a Pytorch Dataset for E2E training

        Args:
            wav_scp_file: the file to wav.scp
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
        self.prepare_egs(wav_scp_file, fst_file)

        self.augmentation = augmentation
        self.use_augmentation = False
        if self.augmentation and len(self.augmentation["pipeline"]) > 0 and self.augmentation["aug_number"] > 0:
            self.use_augmentation = True
            self.noise_df = None
            import pandas
            if "add_noise" in self.augmentation:
                noise_df = pandas.read_csv(self.augmentation["add_noise"]["noise_db_csv"])
                #  noise_df = noise_df.loc[noise_df.duration > self.duration]
                self.noise_df = noise_df.set_index(noise_df.type)

            self.rir_df = None
            if "add_reverb" in self.augmentation:
                tmp_rir_df = pandas.read_csv(self.augmentation["add_reverb"]["rir_db_csv"])
                tmp_rir_df = tmp_rir_df.loc[tmp_rir_df["type"] == "simulated_rirs"]
                # load the RIR database
                self.rir_df = tmp_rir_df.set_index(tmp_rir_df.type)

            logging.info("Using data augmentation")

    def __len__(self):
        return len(self.egs_holder)

    def __getitem__(self, idx):
        audio, sr = utils.kaldi.load_wav_from_scp(self.egs_holder[idx].filename)
        egs = self.egs_holder[idx]
        egs.wav = audio
        if self.use_augmentation:
            audio, selected_aug = augmentation.data_augmentation(audio,
                           self.augmentation,
                           noise_df=self.noise_df,
                           rir_df=self.rir_df)
            if "sanity_check_path" in self.augmentation and "sanity_check_samples" in self.augmentation and idx <= self.augmentation["sanity_check_samples"]:
                os.makedirs(self.augmentation["sanity_check_path"], exist_ok=True)
                torchaudio.save(self.augmentation["sanity_check_path"] + "/" + str(idx) + "-" + ','.join(selected_aug) + ".wav", audio, 16_000)


        audio = audio.squeeze(0)
        return audio, egs

    def __item__(self, i):
        return self.egs_holder[i]

    def prepare_egs(self, wav, fst_file):
        """Method to prepare egs_holder"""
        # egs file is wav.scp file
        utt2wav = utils.kaldi.read_wav_scp(wav)
        utt2len = utils.kaldi.read_utt2len_file(
            "{}/utt2len".format(os.path.dirname(wav))
        )
        self.egs_holder = self.get_egs_holder(fst_file, utt2wav, utt2len)

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
                this_egs_info = EgsInfo(
                    None, uttname, utt2wav[uttname], fstscp, num_output_frames
                )
                if min_path_length == -1:
                    logging.warning(
                        "get_egs_holder, Utterance %s rejected due to the failure of 'fst:RmEpsilon' to remove eps on input label",
                        this_egs_info.name,
                    )
                    skipped += 1
                    continue

                if min_path_length > num_output_frames:
                    logging.warning(
                        "get_egs_holder, Utterance %s rejected since it has more labels than frames",
                        this_egs_info.name,
                    )
                    skipped += 1
                    continue

                egs_holder.append(this_egs_info)
                done += 1
            logging.info(
                "In get_egs_holder: total=%d, done=%d, skipped=%d", total, done, skipped
            )
        return egs_holder
