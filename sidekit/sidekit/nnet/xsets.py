# -*- coding: utf-8 -*-
#
# This file is part of SIDEKIT.
#
# SIDEKIT is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#
# SIDEKIT is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#
# SIDEKIT is free software: you can redistribute it and/or modify
# it under the terms of the GNU LLesser General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# SIDEKIT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with SIDEKIT.  If not, see <http://www.gnu.org/licenses/>.

"""
Copyright 2014-2021 Anthony Larcher

"""

import math
import numpy
import pandas
import random
import torch
import torchaudio
import tqdm
import soundfile
import yaml

from torch.utils.data import Dataset
from .augmentation import data_augmentation
from ..bosaris.idmap import IdMap

__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2015-2021 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


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
        """[summary]

        Args:
            data_source ([type]): [description]
            spk_count ([type]): [description]
            examples_per_speaker ([type]): [description]
            samples_per_speaker ([type]): [description]
            batch_size ([type]): [description]
            num_replicas: number of GPUs for parallel computing
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

        assert batch_size % examples_per_speaker == 0
        assert (self.samples_per_speaker * self.spk_count * self.examples_per_speaker) % self.num_process == 0

        self.batch_size = batch_size // (self.examples_per_speaker * self.num_replicas)
        #self.batch_size = batch_size // self.examples_per_speaker

        # reference all segment indexes per speaker
        for idx in range(self.spk_count):
            self.labels_to_indices[idx] = list()
        for idx, value in enumerate(self.train_sessions):
            self.labels_to_indices[value].append(idx)
        # shuffle segments per speaker
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        for idx, ldlist in enumerate(self.labels_to_indices.values()):
            ldlist = numpy.array(ldlist)
            self.labels_to_indices[idx] = ldlist[torch.randperm(ldlist.shape[0], generator=g).numpy()]

        self.segment_cursors = numpy.zeros((len(self.labels_to_indices),), dtype=numpy.int)


    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        numpy.random.seed(self.seed + self.epoch)

        # Generate batches per speaker
        straight = numpy.arange(self.spk_count)
        indices = numpy.ones((self.samples_per_speaker, self.spk_count), dtype=numpy.int) * straight
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
                to_pick = numpy.sum(probs).astype(numpy.int)
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
                self.labels_to_indices[value] = self.labels_to_indices[value][torch.randperm(self.labels_to_indices[value].shape[0], generator=g)]
                self.segment_cursors[value] = 0
            self.index_iterator[idx] = self.labels_to_indices[value][self.segment_cursors[value]]
            self.segment_cursors[value] += 1
        #self.index_iterator = self.index_iterator.reshape(-1, self.num_process * self.examples_per_speaker)[:, self.rank * self.examples_per_speaker:(self.rank + 1) * self.examples_per_speaker].flatten()

        self.index_iterator = numpy.repeat(self.index_iterator, self.num_replicas)
        self.index_iterator = self.index_iterator.reshape(-1, self.num_process * self.examples_per_speaker * self.num_replicas)[:, self.rank * self.examples_per_speaker * self.num_replicas:(self.rank + 1) * self.examples_per_speaker * self.num_replicas].flatten()

        return iter(self.index_iterator)

    def __len__(self) -> int:
        #return (self.samples_per_speaker * self.spk_count * self.examples_per_speaker) // self.num_process
        return (self.samples_per_speaker * self.spk_count * self.examples_per_speaker * self.num_replicas) // self.num_process



    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class SideSet(Dataset):

    def __init__(self,
                 dataset,
                 set_type="train",
                 chunk_per_segment=1,
                 transform_number=1,
                 overlap=0.,
                 dataset_df=None,
                 min_duration=0.165,
                 output_format="pytorch",
                 ):
        """

        :param dataset_yaml: name of the YAML file describing the dataset
        :param set_type: string, can be "train" or "validation"
        :param chunk_per_segment: number of chunks to select for each segment
        default is 1 and -1 means select all possible chunks
        """
        self.data_path = dataset["data_path"]
        self.sample_rate = int(dataset["sample_rate"])
        self.data_file_extension = dataset["data_file_extension"]
        self.transformation = ''
        self.min_duration = min_duration
        self.output_format = output_format
        self.transform_number = transform_number

        if set_type == "train":
            self.duration = dataset["train"]["duration"]
            self.transformation = dataset["train"]["transformation"]
        else:
            self.duration = dataset["valid"]["duration"]
            self.transformation = dataset["valid"]["transformation"]

        self.sample_number = int(self.duration * self.sample_rate)
        self.overlap = int(overlap * self.sample_rate)

        # Load the dataset description as pandas.dataframe
        if dataset_df is None:
            df = pandas.read_csv(dataset["dataset_description"])
        else:
            assert isinstance(dataset_df, pandas.DataFrame)
            df = dataset_df

        # From each segment which duration is longer than the chosen one
        # select the requested segments
        if set_type == "train":
            tmp_sessions = df.loc[df['duration'] > self.duration]
        else:
            if not "duration" == '':
                tmp_sessions = df.loc[df['duration'] > self.duration]
            else:
                self.sessions = df

        # Create lists for each column of the dataframe
        df_dict = dict(zip(df.columns, [[], [], [], [], [], [], []]))
        df_dict["file_start"] = list()
        df_dict["file_duration"] = list()

        # For each segment, get all possible segments with the current overlap
        for idx in tqdm.trange(len(tmp_sessions), desc='indexing all ' + set_type + ' segments', mininterval=1, disable=None):
            current_session = tmp_sessions.iloc[idx]

            # Compute possible starts
            possible_starts = numpy.arange(0,
                                           int(self.sample_rate * (current_session.duration - self.duration)),
                                           self.sample_number
                                           ) + int(self.sample_rate * (current_session.duration % self.duration / 2))
            possible_starts += int(self.sample_rate * current_session.start)

            # Select max(seg_nb, possible_segments) segments
            if chunk_per_segment == -1:
                starts = possible_starts
                chunk_nb = len(possible_starts)
            else:
                chunk_nb = min(len(possible_starts), chunk_per_segment)
                starts = numpy.random.permutation(possible_starts)[:chunk_nb]

            # Once we know how many segments are selected, create the other fields to fill the DataFrame
            for ii in range(chunk_nb):
                df_dict["database"].append(current_session.database)
                df_dict["speaker_id"].append(current_session.speaker_id)
                df_dict["file_id"].append(current_session.file_id)
                df_dict["start"].append(starts[ii])
                df_dict["duration"].append(self.duration)
                df_dict["file_start"].append(current_session.start)
                df_dict["file_duration"].append(current_session.duration)
                df_dict["speaker_idx"].append(current_session.speaker_idx)
                df_dict["gender"].append(current_session.gender)

        self.sessions = pandas.DataFrame.from_dict(df_dict)
        self.len = len(self.sessions)

        self.transform = dict()
        if (self.transformation["pipeline"] != '') and (self.transformation["pipeline"] is not None):
            transforms = self.transformation["pipeline"].split(',')
            if "add_noise" in transforms:
                self.transform["add_noise"] = self.transformation["add_noise"]
            if "add_reverb" in transforms:
                self.transform["add_reverb"] = self.transformation["add_reverb"]
            if "codec" in transforms:
                self.transform["codec"] = []
            if "phone_filtering" in transforms:
                self.transform["phone_filtering"] = []

        self.noise_df = None
        if "add_noise" in self.transform:
            noise_df = pandas.read_csv(self.transformation["add_noise"]["noise_db_csv"])
            noise_df = noise_df.loc[noise_df.duration > self.duration]
            self.noise_df = noise_df.set_index(noise_df.type)

        self.rir_df = None
        if "add_reverb" in self.transform:
            tmp_rir_df = pandas.read_csv(self.transformation["add_reverb"]["rir_db_csv"])
            tmp_rir_df = tmp_rir_df.loc[tmp_rir_df["type"] == "simulated_rirs"]
            # load the RIR database
            self.rir_df = tmp_rir_df.set_index(tmp_rir_df.type)


    def __getitem__(self, index):
        """

        :return:
        """
        # Check the size of the file
        current_session = self.sessions.iloc[index]

        # TODO is this required ?
        nfo = torchaudio.info(f"{self.data_path}/{current_session['file_id']}{self.data_file_extension}")
        original_start = int(current_session['start'])
        if self.overlap > 0:
            lowest_shift = self.overlap/2
            highest_shift = self.overlap/2
            if original_start < (current_session['file_start']*self.sample_rate + self.sample_number/2):
                lowest_shift = int(original_start - current_session['file_start']*self.sample_rate)
            if original_start + self.sample_number > (current_session['file_start'] + current_session['file_duration'])*self.sample_rate - self.sample_number/2:
                highest_shift = int((current_session['file_start'] + current_session['file_duration'])*self.sample_rate - (original_start + self.sample_number))
            start_frame = original_start + int(random.uniform(-lowest_shift, highest_shift))
        else:
            start_frame = original_start

        conversion_rate = nfo.sample_rate // self.sample_rate

        if start_frame + conversion_rate * self.sample_number >= nfo.num_frames:
            start_frame = numpy.min(nfo.num_frames - conversion_rate * self.sample_number - 1)

        speech, speech_fs = torchaudio.load(f"{self.data_path}/{current_session['file_id']}{self.data_file_extension}",
                                            frame_offset=conversion_rate*start_frame,
                                            num_frames=conversion_rate*self.sample_number)

        if nfo.sample_rate != self.sample_rate:
            speech = torchaudio.transforms.Resample(nfo.sample_rate, self.sample_rate).forward(speech)

        speech += 10e-6 * torch.randn(speech.shape)

        if len(self.transform) > 0:
            speech = data_augmentation(speech,
                                       self.sample_rate,
                                       self.transform,
                                       self.transform_number,
                                       noise_df=self.noise_df,
                                       rir_df=self.rir_df)

        speaker_idx = current_session["speaker_idx"]

        if self.output_format == "pytorch":
            return speech, torch.tensor(speaker_idx)
        else:
            return speech, speaker_idx

    def __len__(self):
        """

        :param self:
        :return:
        """
        return self.len

def get_sample(path, resample=None):
  effects = [
    ["remix", "1"]
  ]
  if resample:
    effects.append(["rate", f'{resample}'])
  return torchaudio.sox_effects.apply_effects_file(path, effects=effects)


class IdMapSet(Dataset):
    """
    DataSet that provide data according to a sidekit.IdMap object
    """

    def __init__(self,
                 idmap_name,
                 data_path,
                 file_extension,
                 transform_pipeline={},
                 transform_number=1,
                 sliding_window=False,
                 window_len=3.,
                 window_shift=1.5,
                 sample_rate=16000,
                 min_duration=0.165
                 ):
        """

        :param data_root_name:
        :param idmap_name:
        """
        if isinstance(idmap_name, IdMap):
            self.idmap = idmap_name
        else:
            self.idmap = IdMap(idmap_name)

        self.data_path = data_path
        self.file_extension = file_extension
        self.len = self.idmap.leftids.shape[0]
        self.transformation = transform_pipeline
        self.min_duration = min_duration
        self.sample_rate = sample_rate
        self.sliding_window = sliding_window
        self.window_len = int(window_len * self.sample_rate)
        self.window_shift = int(window_shift * self.sample_rate)
        self.transform_number = transform_number

        self.noise_df = None
        if "add_noise" in self.transformation:
            # Load the noise dataset, filter according to the duration
            noise_df = pandas.read_csv(self.transformation["add_noise"]["noise_db_csv"])
            tmp_df = noise_df.loc[noise_df['duration'] > self.duration]
            self.noise_df = tmp_df['file_id'].tolist()

        self.rir_df = None
        if "add_reverb" in self.transformation:
            # load the RIR database
            tmp_rir_df = pandas.read_csv(self.transformation["add_reverb"]["rir_db_csv"])
            self.rir_df = zip(tmp_rir_df['file_id'].tolist(), tmp_rir_df['channel'].tolist())

    def __getitem__(self, index):
        """

        :param index:
        :return:
        """
        # Read start and stop and convert to time in seconds
        if self.idmap.start[index] is None:
            start = 0
        else:
            start = int(self.idmap.start[index] * 0.01 * self.sample_rate)

        if self.idmap.stop[index] is None:
            speech, speech_fs = torchaudio.load(f"{self.data_path}/{self.idmap.rightids[index]}.{self.file_extension}")
            duration = int(speech.shape[1] - start)
        else:
            duration = int(self.idmap.stop[index] * 0.01 * self.sample_rate) - start
            # add this in case the segment is too short
            if duration <= self.min_duration * self.sample_rate:
                middle = start + duration // 2
                start = int(max(0, int(middle - (self.min_duration * self.sample_rate / 2))))
                duration = int(self.min_duration * self.sample_rate)
            speech, speech_fs = torchaudio.load(f"{self.data_path}/{self.idmap.rightids[index]}.{self.file_extension}",
                                                frame_offset=start,
                                                num_frames=duration)

        speech += 10e-6 * torch.randn(speech.shape)

        if self.sliding_window:
            speech = speech.squeeze().unfold(0, self.window_len, self.window_shift)
            #middle_points = numpy.arange(start + self.window_len / 2,
            #                             start + duration - self.window_len / 2,
            #                             self.window_shift)
            #starts = middle_points - self.window_shift / 2
            #stops = middle_points + self.window_shift / 2
            #starts[0] = start
            #stops[-1] = start + duration
            #stop = stops
            #start = starts
            stop = start + duration
        else:
            stop = start + duration

        if len(self.transformation.keys()) > 0:
            speech = data_augmentation(speech,
                                       speech_fs,
                                       self.transformation,
                                       self.transform_number,
                                       noise_df=self.noise_df,
                                       rir_df=self.rir_df)


        speech = speech.squeeze()
        
        return speech, self.idmap.leftids[index], self.idmap.rightids[index], start, stop

    def __len__(self):
        """

        :param self:
        :return:
        """
        return self.len


class IdMapSetPerSpeaker(Dataset):
    """
    DataSet that provide data according to a sidekit.IdMap object
    """

    def __init__(self,
                 idmap_name,
                 data_path,
                 file_extension,
                 transform_pipeline={},
                 transform_number=1,
                 sample_rate=16000,
                 min_duration=0.165
                 ):
        """

        :param idmap_name:
        :param data_root_path:
        :param file_extension:
        :param transform_pipeline:
        :param transform_number:
        :param sample_rate:
        :param min_duration:
        """
        if isinstance(idmap_name, IdMap):
            self.idmap = idmap_name
        else:
            self.idmap = IdMap(idmap_name)

        self.data_path = data_path
        self.file_extension = file_extension
        self.len = len(set(self.idmap.leftids))
        self.transformation = transform_pipeline
        self.transform_number = transform_number
        self.min_duration = min_duration
        self.sample_rate = sample_rate
        self.speaker_list = list(set(self.idmap.leftids))
        self.output_im = IdMap()
        self.output_im.leftids = numpy.unique(self.idmap.leftids)
        self.output_im.rightids = self.output_im.leftids
        self.output_im.start = numpy.empty(self.output_im.rightids.shape[0], "|O")
        self.output_im.stop = numpy.empty(self.output_im.rightids.shape[0], "|O")

        self.noise_df = None
        if "add_noise" in self.transformation:
            # Load the noise dataset, filter according to the duration
            noise_df = pandas.read_csv(self.transformation["add_noise"]["noise_db_csv"])
            tmp_df = noise_df.loc[noise_df['duration'] > self.duration]
            self.noise_df = tmp_df['file_id'].tolist()

        self.rir_df = None
        if "add_reverb" in self.transformation:
            # load the RIR database
            tmp_rir_df = pandas.read_csv(self.transformation["add_reverb"]["rir_db_csv"])
            self.rir_df = zip(tmp_rir_df['file_id'].tolist(), tmp_rir_df['channel'].tolist())

    def __getitem__(self, index):
        """

        :param index:
        :return:
        """

        # Loop on all segments from the given speaker to load data
        spk_id = self.output_im.leftids[index]
        tmp_data = []
        #nfo = soundfile.info(f"{self.data_path}/{self.idmap.rightids[index]}.{self.file_extension}")
        for sid, seg_id, seg_start, seg_stop in zip(self.idmap.leftids, self.idmap.rightids,
                                                    self.idmap.start, self.idmap.stop):
            if sid == spk_id:

                # Read start and stop and convert to time in seconds
                if seg_start is None:
                    start = 0
                else:
                    start = int(seg_start * 0.01 * self.sample_rate)

                if seg_stop is None:
                    speech, speech_fs = torchaudio.load(f"{self.data_path}/{self.idmap.rightids[index]}.{self.file_extension}")
                    duration = int(speech.shape[1] - start)
                else:
                    duration = int(seg_stop * 0.01 * self.sample_rate) - start
                    # add this in case the segment is too short
                    if duration <= self.min_duration * self.sample_rate:
                        middle = start + duration // 2
                        start = int(max(0, int(middle - (self.min_duration * self.sample_rate / 2))))
                        duration = int(self.min_duration * self.sample_rate)

                    speech, speech_fs = torchaudio.load(f"{self.data_path}/{seg_id}.{self.file_extension}",
                                                        frame_offset=start,
                                                        num_frames=duration)

                speech += 10e-6 * torch.randn(speech.shape)
                tmp_data.append(speech)

        speech = torch.cat(tmp_data, dim=1)
        speech += 10e-6 * torch.randn(speech.shape)

        if len(self.transformation.keys()) > 0:
            speech = data_augmentation(speech,
                                       speech_fs,
                                       self.transformation,
                                       self.transform_number,
                                       noise_df=self.noise_df,
                                       rir_df=self.rir_df)

        stop = start + duration
        speech = speech.squeeze()

        return speech, self.idmap.leftids[index], self.idmap.rightids[index], start, stop

    def __len__(self):
        """

        :param self:
        :return:
        """
        return self.len

