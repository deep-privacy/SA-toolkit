"""
This file is part of SIDEKIT.
Copyright 2014-2023 Anthony Larcher
"""

import torch
import torchaudio
import logging
import pandas
import sys
import tqdm
import numpy
import random
import os
from itertools import chain

from .. import augmentation
from .. import utils


class SideSampler(torch.utils.data.Sampler):
    """
    Data Sampler used to generate uniformly distributed batches
    """

    def __init__(self,
                 data_source,
                 spk_count,
                 examples_per_speaker,
                 samples_per_speaker,
                 batch_size,
                 seed=0,
                 rank=0,
                 num_process=1,
                 num_replicas=1):
        """
        Samples data from the dataset uniformly for each speaker and generates batches
        with the specified size. This sampler is used for parallel computing with multiple GPUs.

        Args:
            data_source (torch.utils.data.Dataset): dataset to be sampled
            spk_count (int): number of unique speakers in the dataset
            examples_per_speaker (int): number of examples from each speaker in a batch
            samples_per_speaker (int):  number of samples for each speaker in epoch
            batch_size (int): batch size
            seed (int): random seed for shuffling the dataset
            rank (int): process rank for distributed computing
            num_process (int): number of parallel processes for distributed computing
        """
        self.train_sessions = data_source
        self.labels_to_indices = dict()
        self.spk_count = spk_count
        self.examples_per_speaker = examples_per_speaker
        self.samples_per_speaker = samples_per_speaker
        self.epoch = 0
        self.seed = seed
        self.rank = rank
        self.num_process = num_process
        self.num_replicas = num_replicas

        assert batch_size % examples_per_speaker == 0, f"minibatch_size:{batch_size} % examples_per_speaker_in_batch:{examples_per_speaker} = {batch_size % examples_per_speaker} But should equal 0"
        assert (self.samples_per_speaker * self.spk_count * self.examples_per_speaker) % self.num_process == 0, f"samples_per_speaker_in_epoch:{self.samples_per_speaker}  * spk_count:{self.spk_count} * examples_per_speaker_in_batch:{examples_per_speaker} % num_process:{self.num_process} = {(self.samples_per_speaker * self.spk_count * self.examples_per_speaker) % self.num_process} But should equal 0"

        self.batch_size = batch_size // (self.examples_per_speaker * self.num_replicas)

        # reference all segment indexes per speaker
        self.files = dict()
        for idx in range(self.spk_count):
            self.labels_to_indices[idx] = list()
        jdx = 0
        for idx, value in enumerate(self.train_sessions):
            if value not in self.files:
                self.files[value] = jdx
                jdx+=1
            self.labels_to_indices[self.files[value]].append(idx)
        # shuffle segments per speaker
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        for idx, ldlist in enumerate(self.labels_to_indices.values()):
            ldlist = numpy.array(ldlist)
            self.labels_to_indices[idx] = ldlist[torch.randperm(ldlist.shape[0], generator=g).numpy()]

        self.segment_cursors = numpy.zeros((len(self.labels_to_indices),), dtype=numpy.int32)

    def __iter__(self):
        """

        :return:
        """
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        numpy.random.seed(self.seed + self.epoch)

        # Generate batches per speaker
        straight = numpy.arange(self.spk_count)
        indices = numpy.ones((self.samples_per_speaker, self.spk_count), dtype=numpy.int32) * straight
        batch_cursor = 0
        # each line of "indices" represents all speaker indexes (shuffled in a different way)
        for idx in range(self.samples_per_speaker):
            if batch_cursor == 0:
                indices[idx, :] = numpy.random.permutation(straight)
            else:
                # if one batch is split between the end of previous line and the beginning of current line
                # we make sure no speaker is present twice in this batch
                probs = numpy.ones_like(straight)
                probs[indices[idx-1, -batch_cursor:]] = 0
                probs = probs/numpy.sum(probs)
                indices[idx, :self.batch_size - batch_cursor] = numpy.random.choice(self.spk_count, self.batch_size - batch_cursor, replace=False, p=probs)
                probs = numpy.ones_like(straight)
                probs[indices[idx, :self.batch_size - batch_cursor]] = 0
                to_pick = numpy.sum(probs).astype(numpy.int32)
                probs = probs/numpy.sum(probs)
                indices[idx, self.batch_size - batch_cursor:] = numpy.random.choice(self.spk_count, to_pick, replace=False, p=probs)

                assert numpy.sum(indices[idx, :]) == numpy.sum(straight)
            batch_cursor = (batch_cursor + indices.shape[1]) % self.batch_size

        # now we have the speaker indexes to sample in batches
        batch_matrix = numpy.repeat(indices, self.examples_per_speaker, axis=1).flatten()

        # we want to convert the speaker indexes into segment indexes
        self.index_iterator = numpy.zeros_like(batch_matrix)

        # keep track of next segment index to sample for each speaker
        for idx, value in enumerate(batch_matrix):
            if self.segment_cursors[value] > len(self.labels_to_indices[value]) - 1:
                if len(self.labels_to_indices[value]) > 1:
                    self.labels_to_indices[value] = self.labels_to_indices[value][torch.randperm(self.labels_to_indices[value].shape[0], generator=g)]
                self.segment_cursors[value] = 0
            self.index_iterator[idx] = self.labels_to_indices[value][self.segment_cursors[value]]
            self.segment_cursors[value] += 1
        #self.index_iterator = self.index_iterator.reshape(-1, self.num_process * self.examples_per_speaker)[:, self.rank * self.examples_per_speaker:(self.rank + 1) * self.examples_per_speaker].flatten()

        self.index_iterator = numpy.repeat(self.index_iterator, self.num_replicas)
        try:
            self.index_iterator = self.index_iterator.reshape(-1, self.num_process * self.examples_per_speaker * self.num_replicas)[:, self.rank * self.examples_per_speaker * self.num_replicas:(self.rank + 1) * self.examples_per_speaker * self.num_replicas].flatten()
        except Exception:
            logging.critical(f"Pease tweak samples_per_speaker_in_epoch paramater to another value {self.examples_per_speaker} does not allow iterator select examples, values like 100/200 work better")

        return iter(self.index_iterator[:(self.index_iterator.shape[0]//self.batch_size)*self.batch_size])

    def __len__(self) -> int:
        #return (self.samples_per_speaker * self.spk_count * self.examples_per_speaker) // self.num_process
        return (((self.samples_per_speaker * self.spk_count * self.examples_per_speaker * self.num_replicas)//self.batch_size)*self.batch_size)//self.num_process

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class SideSet(torch.utils.data.Dataset):
    """
    A dataset class for loading data for network training.
    """
    def __init__(self,
                 dataset_df,
                 segment_size=48000,
                 set_type="train",
                 tqdm_display=True,
                 overlap=-1,
                 augmentation={},
                 ):
        """
        Initializes the SideSet dataset.

        Args:
        - dataset_df: a pandas dataframe containing the dataset.
        - segment_size: an integer indicating the size of the segments to use for training. Default is 48000.
        - set_type: a string indicating whether the dataset is for "train" or "validation". Default is "train".
        - overlap: an integer indicating the amount of overlap between consecutive segments. Default is half of the segment_size.
        - augmentation: a dictionary containing the data augmentation configurations. Default is an empty dictionary.
        """
        self.change_params(
                 dataset_df=dataset_df,
                 segment_size=segment_size,
                 set_type=set_type,
                 tqdm_display=tqdm_display,
                 overlap=overlap,
                 augmentation=augmentation,
        )

    def change_params(self,
                 dataset_df=None,
                 segment_size=None,
                 set_type=None,
                 tqdm_display=None,
                 overlap=None,
                 augmentation=None,
                 ):

        if augmentation is None:
            augmentation = self.augmentation
        self.augmentation = augmentation
        if overlap is None:
            overlap = self.overlap
        self.overlap = overlap
        if tqdm_display is None:
            tqdm_display = self.tqdm_display
        self.tqdm_display = tqdm_display
        if set_type is None:
            set_type = self.set_type
        self.set_type = set_type
        if segment_size is None:
            segment_size = self.segment_size
        self.segment_size = segment_size
        if dataset_df is None:
            dataset_df = self.dataset_df
        self.dataset_df = dataset_df

        self.sample_number = segment_size
        self.overlap = segment_size//2 if overlap==-1 else overlap

        df = dataset_df

        tmp_sessions = df.loc[df['duration'] > self.sample_number]

        # Create lists for each column of the dataframe
        df_dict = dict(zip(df.columns, [[], [], [], [], [], [], []]))
        df_dict["file_start"] = list()
        df_dict["file_duration"] = list()

        # For each segment, get all possible segments with the current overlap
        session_range = tqdm.trange(len(tmp_sessions), desc='Indexing all ' + set_type + ' segments', disable=not tqdm_display)
        for idx in session_range:
            current_session = tmp_sessions.iloc[idx]

            # Compute possible starts
            possible_starts = numpy.arange(0,
                                           int(current_session.duration - self.sample_number),
                                           self.sample_number
                                           ) + int(current_session.duration % self.sample_number / 2)
            possible_starts = numpy.append(possible_starts, int(current_session.start))
            possible_starts = numpy.append(possible_starts, int(current_session.duration - self.sample_number))

            chunk_nb = len(possible_starts)

            df_dict["database"].append(([current_session.database] * chunk_nb))
            df_dict["speaker_id"].append(([current_session.speaker_id] * chunk_nb))
            df_dict["file_id"].append(([current_session.file_id] * chunk_nb))
            df_dict["start"].append((possible_starts))
            df_dict["duration"].append(([self.sample_number] * chunk_nb))
            df_dict["file_start"].append(([current_session.start] * chunk_nb))
            df_dict["file_duration"].append(([current_session.duration] * chunk_nb))
            df_dict["speaker_idx"].append(([current_session.speaker_idx] * chunk_nb))
            df_dict["gender"].append(([current_session.gender] * chunk_nb))

        df_dict["database"] = list(chain(*df_dict["database"]))
        df_dict["speaker_id"] = list(chain(*df_dict["speaker_id"]))
        df_dict["file_id"] = list(chain(*df_dict["file_id"]))
        df_dict["start"] = list(chain(*df_dict["start"]))
        df_dict["duration"] = list(chain(*df_dict["duration"]))
        df_dict["file_start"] = list(chain(*df_dict["file_start"]))
        df_dict["file_duration"] = list(chain(*df_dict["file_duration"]))
        df_dict["speaker_idx"] = list(chain(*df_dict["speaker_idx"]))
        df_dict["gender"] = list(chain(*df_dict["gender"]))

        self.sessions = pandas.DataFrame.from_dict(df_dict)
        self.len = len(self.sessions)
        self.initial_len = len(tmp_sessions)

        self.use_augmentation = False
        if self.augmentation and len(self.augmentation["pipeline"]) > 0 and self.augmentation["aug_number"] > 0:
            self.use_augmentation = True
            self.noise_df = None
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

            if tqdm_display:
                logging.info("Using data augmentation")


    def __getitem__(self, index):
        """

        :return:
        """
        current_session = self.sessions.iloc[index % self.len]

        original_start = int(current_session['start'])
        if self.overlap > 0:
            lowest_shift = self.overlap/2
            highest_shift = self.overlap/2
            if original_start < (current_session['file_start'] + self.sample_number/2):
                lowest_shift = int(original_start - current_session['file_start'])
            if original_start + self.sample_number > (current_session['file_start'] + current_session['file_duration']) - self.sample_number/2:
                highest_shift = int((current_session['file_start'] + current_session['file_duration']) - (original_start + self.sample_number))
            start_frame = original_start + int(random.uniform(-lowest_shift, highest_shift))
        else:
            start_frame = original_start

        speech, sr = utils.kaldi.load_wav_from_scp(current_session['file_id'],
                                            frame_offset=start_frame,
                                            num_frames=self.sample_number)

        speech += 10e-6 * torch.randn(speech.shape)

        speaker_idx = current_session["speaker_idx"]

        speaker_idx = torch.tensor(speaker_idx, dtype=torch.int64)

        if self.use_augmentation:
            speech, selected_aug = augmentation.data_augmentation(speech,
                           self.augmentation,
                           noise_df=self.noise_df,
                           rir_df=self.rir_df)
            if "sanity_check_path" in self.augmentation and "sanity_check_samples" in self.augmentation and index <= self.augmentation["sanity_check_samples"]:
                os.makedirs(self.augmentation["sanity_check_path"], exist_ok=True)
                torchaudio.save(self.augmentation["sanity_check_path"] + "/" + str(index) + "-" + ','.join(selected_aug) + ".wav", speech, 16_000)

            speech = speech.squeeze(0)

        return speech.contiguous(), speaker_idx.contiguous()

    def __len__(self):
        """
        Return the length of the dataset

        :param self:
        :return:
        """
        return self.len

