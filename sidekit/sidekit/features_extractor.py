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
Copyright 2014-2021 Sylvain Meignier and Anthony Larcher

    :mod:`features_server` provides methods to manage features

"""
import copy
import h5py
import logging
import numpy
import os


from . import PARAM_TYPE
from .frontend.features import mfcc, plp
from .frontend.io import read_audio, read_label, write_hdf5, _add_reverb, _add_noise
from .frontend.vad import vad_snr, vad_percentil
from .mixture import vad_energy
from .sidekit_wrappers import process_parallel_lists
from .bosaris import IdMap


__license__ = "LGPL"
__author__ = "Anthony Larcher & Sylvain Meignier"
__copyright__ = "Copyright 2014-2021 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


class FeaturesExtractor(object):
    """
    A FeaturesExtractor process an audio file in SPHERE, WAVE or RAW PCM format and extract filter-banks,
    cepstral coefficients, bottle-neck features (in the future), log-energy and perform a speech activity detection.
    """

    def __init__(self,
                 audio_filename_structure=None,
                 feature_filename_structure=None,
                 sampling_frequency=None,
                 lower_frequency=None,
                 higher_frequency=None,
                 filter_bank=None,
                 filter_bank_size=None,
                 window_size=None,
                 shift=None,
                 ceps_number=None,
                 vad=None,
                 snr=None,
                 pre_emphasis=None,
                 save_param=None,
                 keep_all_features=None,
                 feature_type=None,
                 rasta_plp=None,
                 compressed='percentile'):
        """
        :param audio_filename_structure: a string that gives the structure of the input file to process
        :param feature_filename_structure: a string that gives the structure of the output file to write
        :param sampling_frequency: optional, only required if processing RAW PCM. For other formats, this information
        is read from the file
        :param lower_frequency: lower frequency (in Herz) of the filter bank
        :param higher_frequency: higher frequency of the filter bank
        :param filter_bank: type of fiter scale to use, can be "lin" or "log" (for linear of log-scale)
        :param filter_bank_size: number of filters banks
        :param window_size: size of the sliding window to process (in seconds)
        :param shift: time shift of the sliding window (in seconds)
        :param ceps_number: number of cepstral coefficients to extract
        :param vad: type of voice actovoty detection algorithm to use. Can be "energy", "snr", "percentil" or "lbl"
        to read from a file
        :param snr: signal to noise ratio used for "snr" vad algorithm
        :param pre_emphasis: value given for the pre-emphasis filter (default is 0.97)
        :param save_param: list of strings that indicate which parameters to save. The strings can be:
        "cep" for cepstral coefficients, "fb" for filter-banks, "energy" for the log-energy, "bnf"
        for bottle-neck features and "vad" for the frame selection labels. In the resulting files, parameters are
         always concatenated in the following order: (energy,fb, cep, bnf, vad_label). Default keeps all.
        :param keep_all_features: boolean, if True, all frames are writen; if False, keep only frames according to
        the vad label
        """

        # Set the default values
        self.audio_filename_structure = None
        self.feature_filename_structure = '{}'
        self.sampling_frequency = 8000
        self.lower_frequency = None
        self.higher_frequency = None
        self.filter_bank = None
        self.filter_bank_size = None
        self.window_size = None
        self.shift = None
        self.ceps_number = None
        self.vad = None
        self.snr = None
        self.pre_emphasis = 0.97
        self.save_param = ["energy", "cep", "fb", "bnf", "vad"]
        self.keep_all_features = None
        self.feature_type = 'mfcc'
        self.rasta_plp = True

        if audio_filename_structure is not None:
            self.audio_filename_structure = audio_filename_structure
        if feature_filename_structure is not None:
            self.feature_filename_structure = feature_filename_structure
        if sampling_frequency is not None:
            self.sampling_frequency = sampling_frequency
        if lower_frequency is not None:
            self.lower_frequency = lower_frequency
        if higher_frequency is not None:
            self.higher_frequency = higher_frequency
        if filter_bank is not None:
            self.filter_bank = filter_bank
        if filter_bank_size is not None:
            self.filter_bank_size = filter_bank_size
        if window_size is not None:
            self.window_size = window_size
        if shift is not None:
            self.shift = shift
        if ceps_number is not None:
            self.ceps_number = ceps_number
        if vad is not None:
            self.vad = vad
        if snr is not None:
            self.snr = snr
        if pre_emphasis is not None:
            self.pre_emphasis = pre_emphasis
        if save_param is not None:
            self.save_param = save_param
        if keep_all_features is not None:
            self.keep_all_features = keep_all_features
        if feature_type is not None:
            self.feature_type = feature_type
        if rasta_plp is not None:
            self.rasta_plp = rasta_plp
        if compressed is not None:
            self.compressed = compressed

        self.window_sample = None
        if not (self.window_size is None or self.sampling_frequency is None):
            self.window_sample = int(self.window_size * self.sampling_frequency)

        self.shift_sample = None
        if not (self.shift is None or self.sampling_frequency is None):
            self.shift_sample = int(self.shift * self.sampling_frequency)

        self.show = 'empty'

    def __repr__(self):
        ch = '\t show: {} keep_all_features: {}\n'.format(
            self.show, self.keep_all_features)
        ch += '\t audio_filename_structure: {}  \n'.format(self.audio_filename_structure)
        ch += '\t feature_filename_structure: {}  \n'.format(self.feature_filename_structure)
        ch += '\t pre-emphasis: {} \n'.format(self.pre_emphasis)
        ch += '\t lower_frequency: {}  higher_frequency: {} \n'.format(
            self.lower_frequency, self.higher_frequency)
        ch += '\t sampling_frequency: {} \n'.format(self.sampling_frequency)
        ch += '\t filter bank: {} filters of type {}\n'.format(
            self.filter_bank_size, self.filter_bank)
        ch += '\t ceps_number: {} \n\t window_size: {} shift: {} \n'.format(
            self.ceps_number, self.window_size, self.shift)
        ch += '\t vad: {}  snr: {} \n'.format(self.vad, self.snr)
        return ch

    def extract(self, show, channel,
                input_audio_filename=None,
                output_feature_filename=None,
                backing_store=False,
                noise_file_name=None,
                snr=10,
                reverb_file_name=None,
                reverb_level=-26.):
        """
        Compute the acoustic parameters (filter banks, cepstral coefficients, log-energy and bottleneck features
        for a single channel from a given audio file.

        :param show: ID if the show
        :param channel: channel number (0 if mono file)
        :param input_audio_filename: name of the input audio file to consider if the name of the audio file is independent from the ID of the show
        :param output_feature_filename: name of the output feature file to consider if the name of the feature file is independent from the ID of the show
        :param backing_store: boolean, if False, nothing is writen to disk, if True, the file is writen to disk when closed
        :param rasta: boolean, only for PLP parameters, if True, perform RASTA filtering

        :return: an hdf5 file handler
        """
        # Create the filename to load

        # If the input audio file name does not include the ID of the show
        # (i.e., if the audio_filename_structure does not include {})
        # the audio_filename_structure is updated to use the input_audio_filename
        if input_audio_filename is not None:
            self.audio_filename_structure = input_audio_filename
        audio_filename = self.audio_filename_structure.format(show)

        # If the output file name does not include the ID of the show,
        # (i.e., if the feature_filename_structure does not include {})
        # the feature_filename_structure is updated to use the output_feature_filename
        if output_feature_filename is not None:
            self.feature_filename_structure = output_feature_filename
        feature_filename = self.feature_filename_structure.format(show)

        # Open audio file, get the signal and possibly the sampling frequency
        signal, sample_rate = read_audio(audio_filename, self.sampling_frequency)
        if signal.ndim == 1:
            signal = signal[:, numpy.newaxis]
        # AJOUTER  LE BRUITAGE ET REVERB DU SIGNAL SI NECESSAIRE
        if noise_file_name is not None:
            signal[:, channel] = _add_noise(signal[:, channel], noise_file_name, snr, sample_rate)

        if reverb_file_name is not None:
            signal[:, channel] = _add_reverb(signal[:, channel], reverb_file_name, sample_rate, reverb_level)

        # Process the target channel to return Filter-Banks, Cepstral coefficients and BNF if required
        length, chan = signal.shape

        # If the size of the signal is not enough for one frame, return zero features
        if length < self.window_sample:
            cep = numpy.empty((0, self.ceps_number), dtype=PARAM_TYPE)
            energy = numpy.empty((0, 1), dtype=PARAM_TYPE)
            fb = numpy.empty((0, self.filter_bank_size), dtype=PARAM_TYPE)
            label = numpy.empty((0, 1), dtype='int8')

        else:
            # Random noise is added to the input signal to avoid zero frames.
            numpy.random.seed(0)
            signal[:, channel] += 0.0001 * numpy.random.randn(signal.shape[0])

            dec = self.shift_sample * 250 * 25000 + self.window_sample
            dec2 = self.window_sample - self.shift_sample
            start = 0
            end = min(dec, length)

            # Process the signal by batch to avoid problems for very long signals
            while start < (length - dec2):
                logging.info('process part : %f %f %f',
                             start / self.sampling_frequency,
                             end / self.sampling_frequency,
                             length / self.sampling_frequency)

                if self.feature_type == 'mfcc':
                    # Extract cepstral coefficients, energy and filter banks
                    cep, energy, _, fb = mfcc(signal[start:end, channel],
                                              fs=self.sampling_frequency,
                                              lowfreq=self.lower_frequency,
                                              maxfreq=self.higher_frequency,
                                              nlinfilt=self.filter_bank_size if self.filter_bank == "lin" else 0,
                                              nlogfilt=self.filter_bank_size if self.filter_bank == "log" else 0,
                                              nwin=self.window_size,
                                              shift=self.shift,
                                              nceps=self.ceps_number,
                                              get_spec=False,
                                              get_mspec=True,
                                              prefac=self.pre_emphasis)
                elif self.feature_type == 'plp':
                    cep, energy, _, fb = plp(signal[start:end, channel],
                                             nwin=self.window_size,
                                             fs=self.sampling_frequency,
                                             plp_order=self.ceps_number,
                                             shift=self.shift,
                                             get_spec=False,
                                             get_mspec=True,
                                             prefac=self.pre_emphasis,
                                             rasta=self.rasta_plp)
                
                # Perform feature selection
                label, threshold = self._vad(cep, energy, fb, signal[start:end, channel])

                if len(label) < len(energy):
                    label = numpy.hstack((label, numpy.zeros(len(energy)-len(label), dtype='bool')))

                start = end - dec2
                end = min(end + dec, length)
                if cep.shape[0] > 0:
                    logging.info('!! size of signal cep: %f len %d type size %d', cep[-1].nbytes/1024/1024,
                                 len(cep[-1]),
                                 cep[-1].nbytes/len(cep[-1]))

        # Compute the mean and std of fb and cepstral coefficient computed for all selected frames
        energy_mean = energy[label].mean(axis=0)
        energy_std = energy[label].std(axis=0)
        fb_mean = fb[label, :].mean(axis=0)
        fb_std = fb[label, :].std(axis=0)
        cep_mean = cep[label, :].mean(axis=0)
        cep_std = cep[label, :].std(axis=0)
        # bnf_mean = bnf[label, :].mean(axis=0)
        # bnf_std = bnf[label, :].std(axis=0)

        # Create the HDF5 file
        # Create the directory if it dosn't exist
        dir_name = os.path.dirname(feature_filename)  # get the path
        if not os.path.exists(dir_name) and not (dir_name == ''):
            os.makedirs(dir_name) 
        h5f = h5py.File(feature_filename, 'w', backing_store=backing_store, driver='core')
        if "cep" not in self.save_param:
            cep = None
            cep_mean = None
            cep_std = None
        if "energy" not in self.save_param:
            energy = None
            energy_mean = None
            energy_std = None
        if "fb" not in self.save_param:
            fb = None
            fb_mean = None
            fb_std = None
        if "bnf" not in self.save_param:
            bnf = None
            bnf_mean = None
            bnf_std = None
        if "vad" not in self.save_param:
            label = None
        logging.info(label)
        write_hdf5(show, h5f,
                   cep, cep_mean, cep_std,
                   energy, energy_mean, energy_std,
                   fb, fb_mean, fb_std,
                   bnf, bnf_mean, bnf_std,
                   label,
                   self.compressed)

        return h5f

    def extract_from_signal(self, signal,
                            sample_rate,
                            noise_file_name=None,
                            snr=10,
                            reverb_file_name=None,
                            reverb_level=-26.):
        """
        Compute the acoustic parameters (filter banks, cepstral coefficients, log-energy and bottleneck features
        for a single channel from a given audio file.

        :param show: ID if the show
        :param channel: channel number (0 if mono file)
        :param input_audio_filename: name of the input audio file to consider if the name of the audio file is independent from the ID of the show
        :param output_feature_filename: name of the output feature file to consider if the name of the feature file is independent from the ID of the show
        :param backing_store: boolean, if False, nothing is writen to disk, if True, the file is writen to disk when closed
        :param rasta: boolean, only for PLP parameters, if True, perform RASTA filtering

        :return: an hdf5 file handler
        """
        if signal.ndim == 1:
            signal = signal[:, numpy.newaxis]

        # AJOUTER  LE BRUITAGE ET REVERB DU SIGNAL SI NECESSAIRE
        if noise_file_name is not None:
            signal[:, 0] = _add_noise(signal[:, 0], noise_file_name, snr, sample_rate)

        if reverb_file_name is not None:
            signal[:, 0] = _add_reverb(signal[:, 0], reverb_file_name, sample_rate, reverb_level)

        # Process the target channel to return Filter-Banks, Cepstral coefficients and BNF if required
        length, chan = signal.shape

        # If the size of the signal is not enough for one frame, return zero features
        if length < self.window_sample:
            cep = numpy.empty((0, self.ceps_number), dtype=PARAM_TYPE)
            energy = numpy.empty((0, 1), dtype=PARAM_TYPE)
            fb = numpy.empty((0, self.filter_bank_size), dtype=PARAM_TYPE)
            label = numpy.empty((0, 1), dtype='int8')

        else:
            # Random noise is added to the input signal to avoid zero frames.
            numpy.random.seed(0)
            signal[:, 0] += 0.0001 * numpy.random.randn(signal.shape[0])

            dec = self.shift_sample * 250 * 25000 + self.window_sample
            dec2 = self.window_sample - self.shift_sample
            start = 0
            end = min(dec, length)

            # Process the signal by batch to avoid problems for very long signals
            while start < (length - dec2):
                logging.info('process part : %f %f %f',
                             start / self.sampling_frequency,
                             end / self.sampling_frequency,
                             length / self.sampling_frequency)

                if self.feature_type == 'mfcc':
                    # Extract cepstral coefficients, energy and filter banks
                    cep, energy, _, fb = mfcc(signal[start:end, 0],
                                              fs=self.sampling_frequency,
                                              lowfreq=self.lower_frequency,
                                              maxfreq=self.higher_frequency,
                                              nlinfilt=self.filter_bank_size if self.filter_bank == "lin" else 0,
                                              nlogfilt=self.filter_bank_size if self.filter_bank == "log" else 0,
                                              nwin=self.window_size,
                                              shift=self.shift,
                                              nceps=self.ceps_number,
                                              get_spec=False,
                                              get_mspec=True,
                                              prefac=self.pre_emphasis)
                elif self.feature_type == 'plp':
                    cep, energy, _, fb = plp(signal[start:end, 0],
                                             nwin=self.window_size,
                                             fs=self.sampling_frequency,
                                             plp_order=self.ceps_number,
                                             shift=self.shift,
                                             get_spec=False,
                                             get_mspec=True,
                                             prefac=self.pre_emphasis,
                                             rasta=self.rasta_plp)

                # Perform feature selection
                label, threshold = self._vad(cep, energy, fb, signal[start:end, 0])

                if len(label) < len(energy):
                    label = numpy.hstack((label, numpy.zeros(len(energy) - len(label), dtype='bool')))

                start = end - dec2
                end = min(end + dec, length)
                if cep.shape[0] > 0:
                    logging.info('!! size of signal cep: %f len %d type size %d', cep[-1].nbytes / 1024 / 1024,
                                 len(cep[-1]),
                                 cep[-1].nbytes / len(cep[-1]))

        return label, energy, cep, fb

    def save(self,
             show,
             channel=0,
             input_audio_filename=None,
             output_feature_filename=None,
             noise_file_name=None,
             snr=10,
             reverb_file_name=None,
             reverb_level=-26.):
        """
        Compute the acoustic parameters (filter banks, cepstral coefficients, log-energy and bottleneck features
        for a single channel from a given audio file and save them to disk in a HDF5 format

        :param show:
        :param channel:
        :param input_audio_filename:
        :param output_feature_filename:
        :return:
        """
        # Load the cepstral coefficients, energy, filter-banks, bnf and vad labels
        h5f = self.extract(show,
                           channel,
                           input_audio_filename,
                           output_feature_filename,
                           backing_store=True,
                           noise_file_name=noise_file_name,
                           snr=snr,
                           reverb_file_name=reverb_file_name,
                           reverb_level=reverb_level)
        logging.info(h5f.filename)

        # Write the hdf5 file to disk
        h5f.close()

    @staticmethod
    def _save(show,
              feature_filename_structure,
              save_param,
              cep,
              energy,
              fb,
              bnf,
              label,
              compressed='percentile'):
        """

        :param show:
        :param feature_filename_structure:
        :param save_param:
        :param cep:
        :param energy:
        :param fb:
        :param bnf:
        :param label:
        :return:
        """
        feature_filename = feature_filename_structure.format(show)
        logging.info('output finename: '+feature_filename)
        dir_name = os.path.dirname(feature_filename)  # get the path
        if not os.path.exists(dir_name) and not (dir_name == ''):
            os.makedirs(dir_name)

        h5f = h5py.File(feature_filename, 'a', backing_store=True, driver='core')

        if "cep" not in save_param:
            cep = None
            cep_mean = None
            cep_std = None
        else:
            cep_mean = cep[label, :].mean(axis=0)
            cep_std = cep[label, :].std(axis=0)
        if "energy" not in save_param:
            energy = None
            energy_mean = None
            energy_std = None
        else:
            energy_mean = energy[label].mean(axis=0)
            energy_std = energy[label].std(axis=0)
        if "fb" not in save_param:
            fb = None
            fb_mean = None
            fb_std = None
        else:
            fb_mean = fb[label, :].mean(axis=0)
            fb_std = fb[label, :].std(axis=0)
        if "bnf" not in save_param:
            bnf = None
            bnf_mean = None
            bnf_std = None
        if "vad" not in save_param:
            label = None
        logging.info(label)

        write_hdf5(show, h5f,
                   cep, cep_mean, cep_std,
                   energy, energy_mean, energy_std,
                   fb, fb_mean, fb_std,
                   bnf, bnf_mean, bnf_std,
                   label, compressed)
        h5f.close()

    def save_multispeakers(self,
                           idmap,
                           channel=0,
                           input_audio_filename=None,
                           output_feature_filename=None,
                           keep_all=True,
                           skip_existing_file=False,
                           compressed='percentile'):
        """
        :param idmap:
        :param channel:
        :param input_audio_filename:
        :param output_feature_filename:
        :param keep_all:
        :param skip_existing_file:
        :return:
        """
        param_vad = self.vad
        save_param = copy.deepcopy(self.save_param)
        self.save_param = ["energy", "cep", "fb", "vad"]

        self.vad = None
        if output_feature_filename is None:
            output_feature_filename = self.feature_filename_structure

        tmp_dict = dict()
        nb = 0
        for show, _id, start, stop in zip(idmap.rightids, idmap.leftids, idmap.start, idmap.stop):

            if skip_existing_file:
                if keep_all:
                    file_name = output_feature_filename.format(show)
                else:
                    file_name = output_feature_filename.format(show+'/' + _id)
                if os.path.isfile(file_name):
                    logging.info('existing file: SKIP '+file_name)
                    continue

            if show not in tmp_dict:
                tmp_dict[show] = dict()
            if _id not in tmp_dict[show]:
                tmp_dict[show][_id] = numpy.arange(start, stop-1)
                nb += 1
            else:
                tmp_dict[show][_id] = numpy.concatenate((tmp_dict[show][_id], numpy.arange(start, stop-1)), axis=0)


        output_show = list()
        output_id = list()
        output_start = list()
        output_stop = list()
        global_compression = 'none'
        if self.compressed == 'percentile':
            global_compression = 'percentile'
        self.compressed = 'none'

        for show in tmp_dict:
            # temp_file_name = tempfile.NamedTemporaryFile().name
            # logging.info('tmp file name: '+temp_file_name)
            self.vad = None
            h5f = self.extract(show, channel, input_audio_filename, backing_store=False)
            energy = h5f.get(show + '/energy')[()]
            label = h5f.get(show + '/vad')[()]
            fb = h5f.get(show + '/fb')[()]
            cep = h5f.get(show + '/cep')[()]
            h5f.close()
            self.vad = param_vad
            l = energy.shape[0]
            for _id in tmp_dict[show]:
                idx = tmp_dict[show][_id]
                idx = idx[idx < l]
                _, threshold_id = self._vad(None, energy[idx], None, None)
                logging.info('show: ' + show + ' cluster: ' + _id + ' thr:' + str(threshold_id))
                label_id = energy > threshold_id
                label[idx] = label_id[idx].flatten()

                if not keep_all:
                    output_show.append(show + '/' + _id)
                    output_id.append(_id)
                    output_start.append(0)
                    output_stop.append(idx.shape[0])
                    logging.info('keep_all id: ' + show + ' show: ' + show + '/' + _id + ' start: 0 stop: ' +
                                 str(idx.shape[0]))
                    self._save(show+'/' + _id,
                               output_feature_filename,
                               save_param, cep[idx],
                               energy[idx],
                               fb[idx],
                               None,
                               label[idx],
                               global_compression)

                if keep_all:
                    self._save(show, output_feature_filename, save_param, cep, energy, fb, None, label, global_compression)

        self.vad = param_vad
        self.save_param = save_param

        self.compressed = global_compression

        if keep_all:
            return copy.deepcopy(idmap)
        out_idmap = IdMap()
        out_idmap.set(numpy.array(output_id),
                      numpy.array(output_show),
                      start=numpy.array(output_start, dtype='int32'),
                      stop=numpy.array(output_stop, dtype='int32'))
        return out_idmap

    def _vad(self, cep, log_energy, fb, x, label_file_name=None):
        """
        Apply Voice Activity Detection.

        :param cep: cepstral coefficient (for future VAD)
        :param log_energy: logarithm of the energy
        :param fb: filter bank coefficients (for future VAD)
        :param x: signal
        :return:
        """
        threshold = -numpy.inf
        label = None
        if self.vad is None:
            logging.info('no vad')
            label = numpy.array([True] * log_energy.shape[0])
        elif self.vad == 'snr':
            logging.info('vad : snr')
            window_sample = int(self.window_size * self.sampling_frequency)
            label = vad_snr(x, self.snr, fs=self.sampling_frequency,
                            shift=self.shift, nwin=window_sample)
        elif self.vad == 'energy':
            logging.info('vad : energy')
            label, threshold = vad_energy(log_energy, distrib_nb=3,
                               nb_train_it=8, flooring=0.0001,
                               ceiling=1.5, alpha=0.2)
        elif self.vad == 'percentil':
            label, threshold = vad_percentil(log_energy, 10)
            logging.info('percentil '+str(threshold))
        elif self.vad == 'dnn':
            pass  # TO DO
        elif self.vad == 'lbl':  # load existing labels as reference
            logging.info('vad : lbl')
            label = read_label(label_file_name)
        else:
            logging.warning('Wrong VAD type')
        return label, threshold

    @process_parallel_lists
    def save_list(self,
                  show_list,
                  channel_list,
                  audio_file_list=None,
                  feature_file_list=None,
                  noise_file_list=None,
                  snr_list=None,
                  reverb_file_list=None,
                  reverb_levels=None,
                  num_thread=1):
        """
        Compute the acoustic parameters (filter banks, cepstral coefficients, log-energy and bottleneck features
        for a list of audio files and save them to disk in a HDF5 format
        The process is parallelized if num_thread is higher than 1


        :param show_list: list of IDs of the show to process
        :param channel_list: list of channel indices corresponding to each show
        :param audio_file_list: list of input audio files if the name is independent from the ID of the show
        :param feature_file_list: list of output audio files if the name is independent from the ID of the show
        :param num_thread: number of parallel process to run
        :return:
        """

        """
        TO DO: manage multi-processing when writting in a single file 
        (use a queue but can we still use @process_parallel_lists ???)
        
        """

        logging.info(self)

        # get the length of the longest list
        max_length = max([len(l) for l in [show_list, channel_list, audio_file_list, feature_file_list]
                          if l is not None])

        if show_list is None:
            show_list = numpy.empty(int(max_length), dtype='|O')
        if audio_file_list is None:
            audio_file_list = numpy.empty(int(max_length), dtype='|O')
        if feature_file_list is None:
            feature_file_list = numpy.empty(int(max_length), dtype='|O')
        if noise_file_list is None:
            noise_file_list =  numpy.empty(int(max_length), dtype='|O')
            snr_list = numpy.empty(int(max_length), dtype='|O')
        elif snr_list is None:
            snr_list = numpy.full(int(max_length), 5.)
        if reverb_file_list is None:
            reverb_file_list = numpy.empty(int(max_length), dtype='|O')
            reverb_levels = numpy.empty(int(max_length), dtype='|O')
        elif reverb_levels is None:
            reverb_levels = numpy.full(int(max_length), -26.)


        for show, channel, audio_file, feature_file, noise_file, snr, reverb_file, reverb_level in zip(show_list,
                                                                                                       channel_list,
                                                                                                       audio_file_list,
                                                                                                       feature_file_list,
                                                                                                       noise_file_list,
                                                                                                       snr_list,
                                                                                                       reverb_file_list,
                                                                                                       reverb_levels):

            self.save(show, channel, audio_file, feature_file, noise_file, snr, reverb_file, reverb_level)
