#coding: utf-8 -*-
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
Copyright 2014-2021 Anthony Larcher, Yevhenii Prokopalo
"""


import logging
import math
import os
import numpy
import random
import pandas
import shutil
import torch
import tqdm
import yaml

from collections import OrderedDict
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .pooling import MeanStdPooling
from .pooling import AttentivePooling
from .pooling import GruPooling
from .preprocessor import MfccFrontEnd
from .preprocessor import MelSpecFrontEnd
from .preprocessor import RawPreprocessor
from .xsets import SideSet
from .xsets import IdMapSet
from .xsets import IdMapSetPerSpeaker
from .xsets import SideSampler
from .res_net import ResBlockWFMS
from .res_net import ResBlock
from .res_net import PreFastResNet34
from .res_net import PreHalfResNet34
from .res_net import PreResNet34
from ..bosaris import IdMap
from ..bosaris import Key
from ..bosaris import Ndx
from ..statserver import StatServer
from ..iv_scoring import cosine_scoring
from .sincnet import SincNet
from ..bosaris.detplot import rocch, rocch2eer
from .loss import SoftmaxAngularProto
from .loss import l2_norm
from .loss import ArcMarginProduct
from .loss import ArcLinear
from .loss import AngularProximityMagnet


os.environ['MKL_THREADING_LAYER'] = 'GNU'

#torch.backends.cudnn.benchmark = True

__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2015-2021 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reS'


def seed_worker(seed_val):
    """

    :param worker_id:
    :return:
    """
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def eer(negatives, positives):
    """
    Logarithmic complexity EER computation

    :param negatives: negative_scores (numpy array): impostor scores
    :param positives: positive_scores (numpy array): genuine scores
    :return: float: Equal Error Rate (EER)
    """
    positives = numpy.sort(positives)
    negatives = numpy.sort(negatives)[::-1]

    pos_count = positives.shape[0]
    neg_count = negatives.shape[0]

    p_score = positives[0]
    n_score = negatives[0]

    p_index = 0
    n_index = 0

    next_p_jump = pos_count//2
    next_n_jump = neg_count//2

    kdx = 0
    while True:
        kdx += 1
        if p_index < 0 or n_index < 0:
            return 0
        if p_index > pos_count or n_index > neg_count:
            return 100
        if p_score < n_score:
            p_index = p_index + next_p_jump
            n_index = n_index + next_n_jump
            if next_p_jump == 0 and next_n_jump == 0:
                break
        elif p_score >= n_score:
            p_index = p_index - next_p_jump
            n_index = n_index - next_n_jump
            if next_p_jump == 0 and next_n_jump == 0:
                break

        p_score = positives[p_index]
        n_score = negatives[n_index]
        next_p_jump = next_p_jump//2
        next_n_jump = next_n_jump//2

    eer_predicate = 100

    tfr = (abs(p_index))/pos_count
    tfa = (1+abs(n_index))/neg_count
    if (p_score == n_score and tfr == tfa):
        return tfr

    while positives[p_index] < negatives[n_index]:
        if p_index < pos_count - 1:
            p_index += 1
        elif n_index < neg_count - 1:
            n_index += 1
        else:
            break

    while positives[p_index] > negatives[n_index] and n_index >= 1:
        n_index -= 1

    tfr = (1+p_index)/pos_count
    tfa = (1+n_index)/neg_count

    while tfa > tfr:
        p_index += 1
        while positives[p_index] > negatives[n_index] and n_index >= 1:
            n_index -= 1
        tfr = (1+p_index)/pos_count
        tfa = (1+n_index)/neg_count

    if abs(tfr - tfa) <= eer_predicate:
        eer_predicate = abs(tfr - tfa)
        eer = (tfr + tfa) / 2
    else:
        return eer

    tfr = p_index/pos_count
    tfa = (1+n_index)/neg_count
    if abs(tfr - tfa) <= eer_predicate:
        eer_predicate = abs(tfr - tfa)
        eer = (tfr + tfa) / 2
    else:
        return eer

    while True:
        while negatives[n_index + 1] <= positives[p_index - 1]:
            p_index -= 1
            tfr = p_index/pos_count
            tfa = (1+n_index)/neg_count
            if abs(tfr - tfa) <= eer_predicate:
                eer_predicate = abs(tfr - tfa)
                eer = (tfr + tfa) / 2
            else:
                return eer
        while negatives[n_index + 1] > positives[p_index - 1]:
            n_index += 1
            tfr = p_index/pos_count
            tfa = (1+n_index)/neg_count
            if abs(tfr - tfa) <= eer_predicate:
                eer_predicate = abs(tfr - tfa)
                eer = (tfr + tfa) / 2
            else:
                return eer

    return eer


def test_metrics(model,
                 device,
                 model_opts,
                 data_opts,
                 train_opts):
    """Compute model metrics

    Args:
        model ([type]): [description]
        validation_loader ([type]): [description]
        device ([type]): [description]
        speaker_number ([type]): [description]
        model_archi ([type]): [description]

    Raises:
        NotImplementedError: [description]
        NotImplementedError: [description]

    Returns:
        [type]: [description]
    """
    transform_pipeline = dict()

    xv_stat = extract_embeddings(idmap_name=data_opts["test"]["idmap"],
                                 model_filename=model,
                                 data_root_name=data_opts["test"]["data_path"],
                                 device=device,
                                 transform_pipeline=transform_pipeline,
                                 num_thread=train_opts["num_cpu"],
                                 mixed_precision=train_opts["mixed_precision"])

    tar, non = cosine_scoring(xv_stat,
                              xv_stat,
                              Ndx(data_opts["test"]["ndx"]),
                              wccn=None,
                              check_missing=True,
                              device=device
                              ).get_tar_non(Key(data_opts["test"]["key"]))

    pmiss, pfa = rocch(tar, non)

    return rocch2eer(pmiss, pfa)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_filename='model_best.pth.tar'):
    """

    :param state:
    :param is_best:
    :param filename:
    :param best_filename:
    :return:
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)


class TrainingMonitor():
    """

    """
    def __init__(self,
                 output_file,
                 log_interval=10,
                 patience=numpy.inf,
                 best_accuracy=0.0,
                 best_eer_epoch=1,
                 best_eer=100,
                 compute_test_eer=False
                 ):

        self.current_epoch = 0
        self.log_interval = log_interval
        self.init_patience = patience
        self.current_patience = patience
        self.best_accuracy = best_accuracy
        self.best_eer_epoch = best_eer_epoch
        self.best_eer = best_eer
        self.compute_test_eer = compute_test_eer
        self.test_eer = []

        self.training_loss = []
        self.training_acc = []

        self.val_loss = []
        self.val_acc = []
        self.val_eer = []

        self.is_best = True

        # Initialize the logger
        logging_format = '%(asctime)-15s %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=logging_format, datefmt='%m-%d %H:%M')
        self.logger = logging.getLogger('Monitoring')
        self.logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(output_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)

    def display(self):
        """

        :return:
        """
        # TODO
        self.logger.critical(f"***Validation metrics - Cross validation accuracy = {self.val_acc[-1]} %, EER = {self.val_eer[-1] * 100} %")
        self.logger.critical(f"***Test metrics - Test EER = {self.test_eer[-1] * 100} %")

    def display_final(self):
        """

        :return:
        """
        self.logger.critical(f"Best accuracy {self.best_accuracy * 100.} obtained at epoch {self.best_accuracy_epoch}")

    def update(self,
               epoch=None,
               training_acc=None,
               training_loss=None,
               test_eer=None,
               val_eer=None,
               val_loss=None,
               val_acc=None):

        if epoch is not None:
            self.current_epoch = epoch
        if training_acc is not None:
            self.training_acc.append(training_acc)
        if training_loss is not None:
            self.training_loss.append(training_loss)
        if val_eer is not None:
            self.val_eer.append(val_eer)
        if val_loss is not None:
            self.val_loss.append(val_loss)
        if val_acc is not None:
            self.val_acc.append(val_acc)

        # remember best accuracy and save checkpoint
        if self.compute_test_eer and test_eer is not None:
            self.test_eer.append(test_eer)
            self.is_best = test_eer < self.best_eer
            self.best_eer = min(test_eer, self.best_eer)
            if self.is_best:
                self.best_eer_epoch = epoch
                self.current_patience = self.init_patience
            else:
                self.current_patience -= 1
        elif val_eer is not None:
            self.is_best = val_eer < self.best_eer
            self.best_eer = min(val_eer, self.best_eer)
            if self.is_best:
                self.best_eer_epoch = epoch
                self.current_patience = self.init_patience
            else:
                self.current_patience -= 1


class Xtractor(torch.nn.Module):
    """
    Class that defines an x-vector extractor based on 5 convolutional layers and a mean standard deviation pooling
    """

    def __init__(self,
                 speaker_number,
                 model_archi="xvector",
                 loss=None,
                 norm_embedding=False,
                 aam_margin=0.2,
                 aam_s=30,
                 embedding_size=256):
        """
        If config is None, default architecture is created
        :param model_archi:
        """
        super(Xtractor, self).__init__()
        self.speaker_number = speaker_number
        self.feature_size = None
        self.norm_embedding = norm_embedding

        if model_archi == "xvector":

            self.input_nbdim = 2

            if loss not in ["cce", 'aam']:
                raise NotImplementedError(f"The valid loss are for now cce and aam ")
            else:
                self.loss = loss

            self.activation = torch.nn.LeakyReLU(0.2)

            self.preprocessor = MfccFrontEnd()
            self.feature_size = self.preprocessor.n_mfcc

            self.sequence_network = torch.nn.Sequential(OrderedDict([
                ("conv1", torch.nn.Conv1d(self.feature_size, 512, 5, dilation=1)),
                ("activation1", torch.nn.LeakyReLU(0.2)),
                ("batch_norm1", torch.nn.BatchNorm1d(512)),
                ("conv2", torch.nn.Conv1d(512, 512, 3, dilation=2)),
                ("activation2", torch.nn.LeakyReLU(0.2)),
                ("batch_norm2", torch.nn.BatchNorm1d(512)),
                ("conv3", torch.nn.Conv1d(512, 512, 3, dilation=3)),
                ("activation3", torch.nn.LeakyReLU(0.2)),
                ("batch_norm3", torch.nn.BatchNorm1d(512)),
                ("conv4", torch.nn.Conv1d(512, 512, 1)),
                ("activation4", torch.nn.LeakyReLU(0.2)),
                ("batch_norm4", torch.nn.BatchNorm1d(512)),
                ("conv5", torch.nn.Conv1d(512, 1536, 1)),
                ("activation5", torch.nn.LeakyReLU(0.2)),
                ("batch_norm5", torch.nn.BatchNorm1d(1536))
            ]))

            self.embedding_size = embedding_size

            self.stat_pooling = MeanStdPooling()
            self.stat_pooling_weight_decay = 0
            self.before_speaker_embedding = torch.nn.Sequential(OrderedDict([
                ("linear6", torch.nn.Linear(3072, self.embedding_size))
            ]))

            if self.loss == "aam":
                self.after_speaker_embedding = ArcMarginProduct(self.embedding_size,
                                                                int(self.speaker_number),
                                                                s=64,
                                                                m=0.2,
                                                                easy_margin=False)
            elif self.loss == "cce":
                self.after_speaker_embedding = torch.nn.Sequential(OrderedDict([
                    ("activation6", torch.nn.LeakyReLU(0.2)),
                    ("batch_norm6", torch.nn.BatchNorm1d(512)),
                    ("dropout6", torch.nn.Dropout(p=0.05)),
                    ("linear7", torch.nn.Linear(512, 512)),
                    ("activation7", torch.nn.LeakyReLU(0.2)),
                    ("batch_norm7", torch.nn.BatchNorm1d(512)),
                    ("linear8", torch.nn.Linear(512, int(self.speaker_number)))
                ]))

            self.preprocessor_weight_decay = 0.0002
            self.sequence_network_weight_decay = 0.0002
            self.before_speaker_embedding_weight_decay = 0.002
            self.after_speaker_embedding_weight_decay = 0.002

        elif model_archi == "resnet34":

            self.preprocessor = MelSpecFrontEnd(n_mels=80)
            self.sequence_network = PreResNet34()
            self.embedding_size = embedding_size

            self.before_speaker_embedding = torch.nn.Linear(in_features=5120,
                                                            out_features=self.embedding_size)

            self.stat_pooling = AttentivePooling(256, 80, global_context=True)

            self.loss = "aam"
            self.after_speaker_embedding = ArcMarginProduct(self.embedding_size,
                                                            int(self.speaker_number),
                                                            s = 30.0,
                                                            m = 0.20,
                                                            easy_margin = False)

            self.preprocessor_weight_decay = 0.00002
            self.sequence_network_weight_decay = 0.00002
            self.stat_pooling_weight_decay = 0.00002
            self.before_speaker_embedding_weight_decay = 0.00002
            self.after_speaker_embedding_weight_decay = 0.0002

        elif model_archi == "fastresnet34":
            self.preprocessor = MelSpecFrontEnd()
            self.sequence_network = PreFastResNet34()
            self.embedding_size = embedding_size

            self.before_speaker_embedding = torch.nn.Linear(in_features = 2560,
                                                            out_features = self.embedding_size)

            self.stat_pooling = AttentivePooling(128, 80, global_context=False)
            self.stat_pooling_weight_decay = 0

            self.loss = loss
            if self.loss == "aam":
                self.after_speaker_embedding = ArcMarginProduct(self.embedding_size,
                                                                int(self.speaker_number),
                                                                s = 30,
                                                                m = 0.2,
                                                                easy_margin = False)

            elif self.loss == 'aps':
                self.after_speaker_embedding = SoftmaxAngularProto(int(self.speaker_number))
            elif self.loss == 'smn':
                self.after_speaker_embedding = AngularProximityMagnet(int(self.speaker_number))

            self.preprocessor_weight_decay = 0.00002
            self.sequence_network_weight_decay = 0.00002
            self.stat_pooling_weight_decay = 0.00002
            self.before_speaker_embedding_weight_decay = 0.00002
            self.after_speaker_embedding_weight_decay = 0.0002

        elif model_archi == "halfresnet34":
            self.preprocessor = MelSpecFrontEnd(n_fft=1024,
                                                win_length=400,
                                                hop_length=160,
                                                n_mels=80)
            self.sequence_network = PreHalfResNet34()

            self.embedding_size = embedding_size
            #self.embedding_size = 256
            #self.before_speaker_embedding = torch.nn.Linear(in_features = 5120,
            #                                                out_features = self.embedding_size)

            self.before_speaker_embedding = torch.nn.Sequential(OrderedDict([
                ("lin_be", torch.nn.Linear(in_features = 5120, out_features = self.embedding_size, bias=False)),
                ("bn_be", torch.nn.BatchNorm1d(self.embedding_size))
                ]))

            self.stat_pooling = AttentivePooling(256, 80, global_context=True)

            self.loss = loss
            if self.loss == "aam":
                self.after_speaker_embedding = ArcMarginProduct(self.embedding_size,
                                                                int(self.speaker_number),
                                                                s = 30,
                                                                m = 0.2,
                                                                easy_margin = False)
            elif self.loss == 'aps':
                self.after_speaker_embedding = SoftmaxAngularProto(int(self.speaker_number))
            self.preprocessor_weight_decay = 0.00002
            self.sequence_network_weight_decay = 0.00002
            self.stat_pooling_weight_decay = 0.00002
            self.before_speaker_embedding_weight_decay = 0.00002
            self.after_speaker_embedding_weight_decay = 0.000

        elif model_archi == "rawnet2":

            if loss not in ["cce", 'aam']:
                raise NotImplementedError(f"The valid loss are for now cce and aam ")
            else:
                self.loss = loss

            self.input_nbdim = 2

            filts = [128, [128, 128], [128, 256], [256, 256]]
            self.norm_embedding = True

            self.preprocessor = RawPreprocessor(nb_samp=48000,
                                                in_channels=1,
                                                out_channels=filts[0],
                                                kernel_size=3)

            self.sequence_network = torch.nn.Sequential(OrderedDict([
                ("block0", ResBlockWFMS(nb_filts=filts[1], first=True)),
                ("block1", ResBlockWFMS(nb_filts=filts[1])),
                ("block2", ResBlockWFMS(nb_filts=filts[2])),
                ("block3", ResBlockWFMS(nb_filts=[filts[2][1], filts[2][1]])),
                ("block4", ResBlockWFMS(nb_filts=[filts[2][1], filts[2][1]])),
                ("block5", ResBlockWFMS(nb_filts=[filts[2][1], filts[2][1]]))
            ]))

            self.stat_pooling = GruPooling(input_size=filts[2][-1],
                                           gru_node=1024,
                                           nb_gru_layer=1)

            self.before_speaker_embedding = torch.nn.Linear(in_features = 1024,
                                                            out_features = 1024)

            if self.loss == "aam":
                if loss == 'aam':
                    self.after_speaker_embedding = ArcLinear(1024,
                                                             int(self.speaker_number),
                                                             margin=aam_margin, s=aam_s)
            elif self.loss == "cce":
                self.after_speaker_embedding = torch.nn.Linear(in_features = 1024,
                                                               out_features = int(self.speaker_number),
                                                               bias = True)

            self.preprocessor_weight_decay = 0.000
            self.sequence_network_weight_decay = 0.000
            self.stat_pooling_weight_decay = 0.000
            self.before_speaker_embedding_weight_decay = 0.00
            self.after_speaker_embedding_weight_decay = 0.00

        else:
            is_first_resblock = True

            if isinstance(model_archi, dict):
                cfg = model_archi
            else:
                # Load Yaml configuration
                with open(model_archi, 'r') as fh:
                    cfg = yaml.load(fh, Loader=yaml.FullLoader)

            self.loss = cfg["loss"]["type"]
            if self.loss == "aam":
                self.aam_margin = cfg["loss"]["aam_margin"]
                self.aam_s = cfg["loss"]["aam_s"]

            """
            Prepare Preprocessor
            """
            self.preprocessor = None
            if "preprocessor" in cfg:
                if cfg['preprocessor']["type"] == "sincnet":
                    self.preprocessor = SincNet(
                        waveform_normalize=cfg['preprocessor']["waveform_normalize"],
                        sample_rate=cfg['preprocessor']["sample_rate"],
                        min_low_hz=cfg['preprocessor']["min_low_hz"],
                        min_band_hz=cfg['preprocessor']["min_band_hz"],
                        out_channels=cfg['preprocessor']["out_channels"],
                        kernel_size=cfg['preprocessor']["kernel_size"],
                        stride=cfg['preprocessor']["stride"],
                        max_pool=cfg['preprocessor']["max_pool"],
                        instance_normalize=cfg['preprocessor']["instance_normalize"],
                        activation=cfg['preprocessor']["activation"],
                        dropout=cfg['preprocessor']["dropout"]
                    )
                    self.feature_size = self.preprocessor.dimension
                elif cfg['preprocessor']["type"] == "rawnet2":
                    self.preprocessor = RawPreprocessor(nb_samp=int(cfg['preprocessor']["sampling_frequency"] * cfg['preprocessor']["duration"]),
                                                        in_channels=1,
                                                        out_channels=cfg["feature_size"],
                                                        kernel_size=cfg['preprocessor']["kernel_size"],
                                                        stride=cfg['preprocessor']["stride"],
                                                        padding=cfg['preprocessor']["padding"],
                                                        dilation=cfg['preprocessor']["dilation"])
                    self.feature_size = cfg["feature_size"]
                self.preprocessor_weight_decay = 0.000

            """
            Prepare sequence network
            """
            # Get Feature size
            if self.feature_size is None:
                self.feature_size = cfg["preprocessor"]["feature_size"]

            input_size = self.feature_size

            # Get activation function
            if cfg["activation"] == 'LeakyReLU':
                self.activation = torch.nn.LeakyReLU(0.2)
            elif cfg["activation"] == 'PReLU':
                self.activation = torch.nn.PReLU()
            elif cfg["activation"] == 'ReLU6':
                self.activation = torch.nn.ReLU6()
            else:
                self.activation = torch.nn.ReLU()

            # Create sequential object for the first part of the network
            segmental_layers = []
            for k in cfg["segmental"].keys():
                if k.startswith("lin"):
                    segmental_layers.append((k, torch.nn.Linear(input_size,
                                                                cfg["segmental"][k]["output"])))
                    input_size = cfg["segmental"][k]["output"]

                elif k.startswith("conv2D"):
                    segmental_layers.append((k, torch.nn.Conv2d(in_channels=1,
                                                                out_channels=entry_conv_out_channels,
                                                                kernel_size=entry_conv_kernel_size,
                                                                padding=3,
                                                                stride=1)))

                elif k.startswith("conv"):
                    segmental_layers.append((k, torch.nn.Conv1d(input_size,
                                                                cfg["segmental"][k]["output_channels"],
                                                                kernel_size=cfg["segmental"][k]["kernel_size"],
                                                                dilation=cfg["segmental"][k]["dilation"])))
                    input_size = cfg["segmental"][k]["output_channels"]

                elif k.startswith("ctrans"):
                    segmental_layers.append((k, torch.nn.ConvTranspose1d(input_size,
                                                                         cfg["segmental"][k]["output_channels"],
                                                                         kernel_size=cfg["segmental"][k]["kernel_size"],
                                                                         dilation=cfg["segmental"][k]["dilation"])))
                elif k.startswith("activation"):
                    segmental_layers.append((k, self.activation))

                elif k.startswith('batch_norm'):
                    segmental_layers.append((k, torch.nn.BatchNorm1d(input_size)))

                elif k.startswith('resblock'):
                    segmental_layers.append((ResBlock(cfg["segmental"][k]["input_channel"],
                                                      cfg["segmental"][k]["output_channel"],
                                                      is_first_resblock)))
                    is_first_resblock = False

            self.sequence_network = torch.nn.Sequential(OrderedDict(segmental_layers))
            self.sequence_network_weight_decay = cfg["segmental"]["weight_decay"]

            """
            Pooling
            """
            self.stat_pooling = MeanStdPooling()
            tmp_input_size = input_size * 2
            if cfg["stat_pooling"]["type"] == "GRU":
                self.stat_pooling = GruPooling(input_size=cfg["stat_pooling"]["input_size"],
                                               gru_node=cfg["stat_pooling"]["gru_node"],
                                               nb_gru_layer=cfg["stat_pooling"]["nb_gru_layer"])
                tmp_input_size = cfg["stat_pooling"]["gru_node"]

            self.stat_pooling_weight_decay = cfg["stat_pooling"]["weight_decay"]

            """
            Prepare last part of the network (after pooling)
            """
            # Create sequential object for the second part of the network
            input_size = tmp_input_size
            before_embedding_layers = []
            for k in cfg["before_embedding"].keys():
                if k.startswith("lin"):
                    if cfg["before_embedding"][k]["output"] == "speaker_number":
                        before_embedding_layers.append((k, torch.nn.Linear(input_size, self.speaker_number)))
                    else:
                        before_embedding_layers.append((k, torch.nn.Linear(input_size,
                                                                           cfg["before_embedding"][k]["output"])))
                        input_size = cfg["before_embedding"][k]["output"]

                elif k.startswith("activation"):
                    before_embedding_layers.append((k, self.activation))

                elif k.startswith('batch_norm'):
                    before_embedding_layers.append((k, torch.nn.BatchNorm1d(input_size)))

                elif k.startswith('dropout'):
                    before_embedding_layers.append((k, torch.nn.Dropout(p=cfg["before_embedding"][k])))

            self.embedding_size = input_size
            self.before_speaker_embedding = torch.nn.Sequential(OrderedDict(before_embedding_layers))
            self.before_speaker_embedding_weight_decay = cfg["before_embedding"]["weight_decay"]

            # if loss_criteria is "cce"
            # Create sequential object for the second part of the network
            if self.loss == "cce":
                after_embedding_layers = []
                for k in cfg["after_embedding"].keys():
                    if k.startswith("lin"):
                        if cfg["after_embedding"][k]["output"] == "speaker_number":
                            after_embedding_layers.append((k, torch.nn.Linear(input_size, self.speaker_number)))
                        else:
                            after_embedding_layers.append((k, torch.nn.Linear(input_size,
                                                                          cfg["after_embedding"][k]["output"])))
                            input_size = cfg["after_embedding"][k]["output"]

                    elif k.startswith('arc'):
                        after_embedding_layers.append((k, ArcLinear(input_size,
                                                                    self.speaker_number,
                                                                    margin=self.aam_margin,
                                                                    s=self.aam_s)))

                    elif k.startswith("activation"):
                        after_embedding_layers.append((k, self.activation))

                    elif k.startswith('batch_norm'):
                        after_embedding_layers.append((k, torch.nn.BatchNorm1d(input_size)))

                    elif k.startswith('dropout'):
                        after_embedding_layers.append((k, torch.nn.Dropout(p=cfg["after_embedding"][k])))

                self.after_speaker_embedding = torch.nn.Sequential(OrderedDict(after_embedding_layers))

            elif self.loss == "aam":
                self.norm_embedding = True
                self.after_speaker_embedding = ArcMarginProduct(input_size,
                                                                int(self.speaker_number),
                                                                s=64,
                                                                m=0.2,
                                                                easy_margin=True)

            self.after_speaker_embedding_weight_decay = cfg["after_embedding"]["weight_decay"]

    def forward(self, x, is_eval=False, target=None, norm_embedding=True):
        """

        :param x:
        :param is_eval: False for training
        :return:
        """
        if self.preprocessor is not None:
            x = self.preprocessor(x, is_eval)

        x = self.sequence_network(x)

        # Mean and Standard deviation pooling
        x = self.stat_pooling(x)

        x = self.before_speaker_embedding(x)

        if norm_embedding:
            x = l2_norm(x)

        if self.loss == "cce":
            if is_eval:
                return x
            else:
                return self.after_speaker_embedding(x), x

        elif self.loss in ['aam', 'aps']:
            x = self.after_speaker_embedding(x, target=target), torch.nn.functional.normalize(x, dim=1)
        elif self.loss == 'smn':
            if not is_eval:
                x = self.after_speaker_embedding(x, target=target), x


        return x

    def context_size(self):
        context = 1
        if isinstance(self, Xtractor):
            for name, module in self.sequence_network.named_modules():
                if name.startswith("conv"):
                    context += module.dilation[0] * (module.kernel_size[0] - 1)
        else:
            for name, module in self.module.sequence_network.named_modules():
                if name.startswith("conv"):
                    context += module.dilation[0] * (module.kernel_size[0] - 1)
        return context


def fill_dict(target_dict, source_dict, prefix = ""):
    """
    Recursively Fill a dictionary target_dict by taking values from source_dict

    :param target_dict: output dictionary that is initialized with default values
    :param source_dict: input dictionary
    :return:
    """
    for k1, v1 in target_dict.items():

        if isinstance(v1, dict):
            if k1 in source_dict and isinstance(source_dict[k1], dict):
                fill_dict(v1, source_dict[k1], prefix + "\t")
            else:
                pass
        else:
            if k1 in source_dict and source_dict[k1] is not None:
                target_dict[k1] = source_dict[k1]
            else:
                pass


def update_training_dictionary(dataset_description,
                               model_description,
                               training_description,
                               kwargs=None):
    """

    :param dataset_description:
    :param model_description:
    :param training_description:
    :param kwargs:
    :return:
    """
    dataset_opts=dict()
    model_opts=dict()
    training_opts=dict()

    if isinstance(dataset_description, str) and os.path.isfile(dataset_description):
        with open(dataset_description, 'r') as fh:
            tmp_data_dict = yaml.load(fh, Loader=yaml.FullLoader)
    else:
        tmp_data_dict = dataset_description

    if isinstance(model_description, str) and os.path.isfile(model_description):
        with open(model_description, 'r') as fh:
            tmp_model_dict = yaml.load(fh, Loader=yaml.FullLoader)
    else:
        tmp_model_dict = model_description

    if isinstance(training_description, str) and os.path.isfile(training_description):
        with open(training_description, 'r') as fh:
            tmp_train_dict = yaml.load(fh, Loader=yaml.FullLoader)
    else:
        tmp_train_dict = training_description

    # Initialize default dictionaries
    dataset_opts["data_path"] = None
    dataset_opts["dataset_csv"] = None
    dataset_opts["stratify"] = False
    dataset_opts["data_file_extension"] = ".wav"
    dataset_opts["sample_rate"] = 16000

    dataset_opts["validation_ratio"] = 0.1
    dataset_opts["batch_size"] = 64

    dataset_opts["train"] = dict()
    dataset_opts["train"]["duration"] = 4.
    dataset_opts["train"]["chunk_per_segment"] = -1
    dataset_opts["train"]["overlap"] = 3.9
    dataset_opts["train"]["sampler"] = dict()
    dataset_opts["train"]["sampler"]["examples_per_speaker"] = 1
    dataset_opts["train"]["sampler"]["samples_per_speaker"] = 100
    dataset_opts["train"]["sampler"]["augmentation_replica"] = 1
    dataset_opts["train"]["transform_number"] = 2
    dataset_opts["train"]["transformation"] = dict()
    dataset_opts["train"]["transformation"]["pipeline"] = ""
    dataset_opts["train"]["transformation"]["add_noise"] = dict()
    dataset_opts["train"]["transformation"]["add_noise"]["noise_db_csv"] = ""
    dataset_opts["train"]["transformation"]["add_noise"]["data_path"] = ""
    dataset_opts["train"]["transformation"]["add_reverb"] = dict()
    dataset_opts["train"]["transformation"]["add_reverb"]["rir_db_csv"] = ""
    dataset_opts["train"]["transformation"]["add_reverb"]["data_path"] = ""

    dataset_opts["valid"] = dict()
    dataset_opts["valid"]["duration"] = 2.
    dataset_opts["valid"]["transformation"] = dict()
    dataset_opts["valid"]["transformation"]["pipeline"] = ""
    dataset_opts["valid"]["transformation"]["add_noise"] = dict()
    dataset_opts["valid"]["transformation"]["add_noise"]["noise_db_csv"] = ""
    dataset_opts["valid"]["transformation"]["add_noise"]["data_path"] = ""
    dataset_opts["valid"]["transformation"]["add_reverb"] = dict()
    dataset_opts["valid"]["transformation"]["add_reverb"]["noise_db_csv"] = ""
    dataset_opts["valid"]["transformation"]["add_reverb"]["data_path"] = ""

    dataset_opts["test"] = dict()
    dataset_opts["test"]["idmap"] = ""
    dataset_opts["test"]["ndx"] = ""
    dataset_opts["test"]["key"] = ""
    dataset_opts["test"]["data_path"] =""

    # Initialize model options
    model_opts["speaker_number"] = None
    model_opts["embedding_size"] = 256
    model_opts["loss"] = dict()
    model_opts["loss"]["type"] ="aam"
    model_opts["loss"]["aam_margin"] = 0.2
    model_opts["loss"]["aam_s"] = 30

    model_opts["initial_model_name"] = None
    model_opts["reset_parts"] = []
    model_opts["freeze_parts"] = []

    model_opts["model_type"] = "fastresnet"

    model_opts["preprocessor"] = dict()
    model_opts["preprocessor"]["type"] =  "mel_spec"
    model_opts["preprocessor"]["feature_size"] = 80

    # Initialize training options
    training_opts["log_file"] = "sidekit.log"
    training_opts["numpy_seed"] = 0
    training_opts["torch_seed"] = 0
    training_opts["random_seed"] = 0
    training_opts["deterministic"] = False
    training_opts["epochs"] = 100
    training_opts["lr"] = 1e-3
    training_opts["patience"] = 50
    training_opts["multi_gpu"] = False
    training_opts["num_cpu"] = 5
    training_opts["mixed_precision"] = False
    training_opts["clipping"] = False

    training_opts["optimizer"] = dict()
    training_opts["optimizer"]["type"] = "sgd"
    training_opts["optimizer"]["options"] = None

    training_opts["scheduler"] = dict()
    training_opts["scheduler"]["type"] = "ReduceLROnPlateau"
    training_opts["scheduler"]["step_size_up"] = 10
    training_opts["scheduler"]["base_lr"] = 1e-8
    training_opts["scheduler"]["mode"] = "triangular2"

    training_opts["compute_test_eer"] = False
    training_opts["log_interval"] = 10
    training_opts["validation_frequency"] = 1

    training_opts["tmp_model_name"] = "tmp_model.pt"
    training_opts["best_model_name"] = "best_model.pt"
    training_opts["checkpoint_frequency"] = "10"

    # Use options from the YAML config files
    fill_dict(dataset_opts, tmp_data_dict)
    fill_dict(model_opts, tmp_model_dict)
    fill_dict(training_opts, tmp_train_dict)

    # Overwrite with manually given parameters
    if "lr" in kwargs:
        training_opts["lr"] = kwargs['lr']
    if "batch_size" in kwargs:
        dataset_opts["batch_size"] = kwargs["batch_size"]
    if "optimizer" in kwargs:
        training_opts["optimizer"]["type"] = kwargs["optimizer"]
    if "scheduler" in kwargs:
        training_opts["scheduler"]["type"] = kwargs["scheduler"]
    if "margin" in kwargs:
        model_opts["loss"]["aam_margin"] = kwargs["margin"]
    if "aam_s" in kwargs:
        model_opts["loss"]["aam_s"] = kwargs["aam_s"]

    return dataset_opts, model_opts, training_opts


def get_network(model_opts, local_rank):
    """

    :param model_opts:
    :param local_rank:
    :return:
    """

    if model_opts["model_type"] in ["xvector", "rawnet2", "resnet34", "fastresnet34", "halfresnet34"]:
        model = Xtractor(model_opts["speaker_number"], model_opts["model_type"], loss=model_opts["loss"]["type"], embedding_size=model_opts["embedding_size"])
    else:
        # Custom type of model
        model = Xtractor(model_opts["speaker_number"], model_opts, loss=model_opts["loss"]["type"], embedding_size=model_opts["embedding_size"])

    # Load the model if it exists
    if model_opts["initial_model_name"] is not None and os.path.isfile(model_opts["initial_model_name"]):
        logging.critical(f"*** Load model from = {model_opts['initial_model_name']}")
        checkpoint = torch.load(model_opts["initial_model_name"])

        """
        Here we remove all layers that we don't want to reload

        """
        pretrained_dict = checkpoint["model_state_dict"]
        for part in model_opts["reset_parts"]:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith(part)}

        new_model_dict = model.state_dict()
        new_model_dict.update(pretrained_dict)
        model.load_state_dict(new_model_dict)

        # Freeze required layers
        for name, param in model.named_parameters():
            if name.split(".")[0] in model_opts["reset_parts"]:
                param.requires_grad = False

    if model_opts["loss"]["type"] == "aam" and not (model_opts["loss"]["aam_margin"] == 0.2 and model_opts["loss"]["aam_s"] == 30):
        model.after_speaker_embedding.change_params(model_opts["loss"]["aam_s"], model_opts["loss"]["aam_margin"])
        print(f"Modified AAM: margin = {model.after_speaker_embedding.m} and s = {model.after_speaker_embedding.s}")

    if local_rank < 1:
        

        logging.info(model)
        logging.info("Model_parameters_count: {:d}".format(
            sum(p.numel()
                for p in model.sequence_network.parameters()
                if p.requires_grad) + \
            sum(p.numel()
                for p in model.before_speaker_embedding.parameters()
                if p.requires_grad) + \
            sum(p.numel()
                for p in model.stat_pooling.parameters()
                if p.requires_grad)))

    return model


def get_loaders(dataset_opts, training_opts, model_opts, local_rank=0):
    """

    :param dataset_opts:
    :param training_opts:
    :param model_opts:
    :return:
    """

    """
    Set the dataloaders according to the dataset_yaml
    
    First we load the dataframe from CSV file in order to split it for training and validation purpose
    Then we provide those two
    """
    df = pandas.read_csv(dataset_opts["dataset_csv"])

    stratify = None
    if dataset_opts["stratify"]:
        stratify = df["speaker_idx"]
    training_df, validation_df = train_test_split(df,
                                                  test_size=dataset_opts["validation_ratio"],
                                                  stratify=stratify)

    torch.manual_seed(training_opts['torch_seed'] + local_rank)
    torch.cuda.manual_seed(training_opts['torch_seed'] + local_rank)

    training_set = SideSet(dataset_opts,
                           set_type="train",
                           chunk_per_segment=-1,
                           transform_number=dataset_opts['train']['transform_number'],
                           overlap=dataset_opts['train']['overlap'],
                           dataset_df=training_df,
                           output_format="pytorch",
                           )

    validation_set = SideSet(dataset_opts,
                             set_type="validation",
                             dataset_df=validation_df,
                             output_format="pytorch")

    if model_opts["loss"]["type"] == 'aps':
        samples_per_speaker = 2
    else:
        samples_per_speaker = 1

    if training_opts["multi_gpu"]:
        assert dataset_opts["batch_size"] % torch.cuda.device_count() == 0
        assert dataset_opts["batch_size"] % samples_per_speaker == 0
        batch_size = dataset_opts["batch_size"]//(torch.cuda.device_count() * dataset_opts["train"]["sampler"]["examples_per_speaker"])

        side_sampler = SideSampler(data_source=training_set.sessions['speaker_idx'],
                                   spk_count=model_opts["speaker_number"],
                                   examples_per_speaker=dataset_opts["train"]["sampler"]["examples_per_speaker"],
                                   samples_per_speaker=dataset_opts["train"]["sampler"]["samples_per_speaker"],
                                   batch_size=batch_size,
                                   seed=training_opts['torch_seed'],
                                   rank=local_rank,
                                   num_process=torch.cuda.device_count(),
                                   num_replicas=dataset_opts["train"]["sampler"]["augmentation_replica"]
                                   )
    else:
        batch_size = dataset_opts["batch_size"] // dataset_opts["train"]["sampler"]["examples_per_speaker"]
        side_sampler = SideSampler(data_source=training_set.sessions['speaker_idx'],
                                   spk_count=model_opts["speaker_number"],
                                   examples_per_speaker=dataset_opts["train"]["sampler"]["examples_per_speaker"],
                                   samples_per_speaker=dataset_opts["train"]["sampler"]["samples_per_speaker"],
                                   batch_size=batch_size,
                                   seed=training_opts['torch_seed'],
                                   rank=0,
                                   num_process=torch.cuda.device_count(),
                                   num_replicas=dataset_opts["train"]["sampler"]["augmentation_replica"]
                                   )

    training_loader = DataLoader(training_set,
                                 batch_size=batch_size * dataset_opts["train"]["sampler"]["augmentation_replica"],
                                 shuffle=False,
                                 drop_last=True,
                                 pin_memory=True,
                                 sampler=side_sampler,
                                 num_workers=training_opts["num_cpu"],
                                 persistent_workers=False,
                                 worker_init_fn=seed_worker)

    validation_loader = DataLoader(validation_set,
                                   batch_size=batch_size,
                                   drop_last=False,
                                   pin_memory=True,
                                   num_workers=training_opts["num_cpu"],
                                   persistent_workers=False,
                                   worker_init_fn=seed_worker)

    # Compute indices for target and non-target trials once only to avoid recomputing for each epoch
    classes = torch.ShortTensor(validation_set.sessions['speaker_idx'].to_numpy())
    mask = classes.unsqueeze(1) == classes.unsqueeze(1).T
    tar_indices = torch.tril(mask, -1).numpy()
    non_indices = torch.tril(~mask, -1).numpy()

    # Select a subset of non-target trials to reduce the number of tests
    tar_non_ratio = numpy.sum(tar_indices)/numpy.sum(non_indices)
    non_indices *= (numpy.random.rand(*non_indices.shape) < tar_non_ratio)

    return training_loader, validation_loader, side_sampler, tar_indices, non_indices


def get_optimizer(model, model_opts, train_opts, training_loader):
    """

    :param model:
    :param model_opts:
    :param train_opts:
    :param training_loader:
    :return:
    """
    if train_opts["optimizer"]["type"] == 'adam':
        _optimizer = torch.optim.Adam
        _options = {'lr': train_opts["lr"]}
    elif train_opts["optimizer"]["type"] == 'rmsprop':
        _optimizer = torch.optim.RMSprop
        _options = {'lr': train_opts["lr"]}
    else:  # train_opts["optimizer"]["type"] == 'sgd'
        _optimizer = torch.optim.SGD
        _options = {'lr': train_opts["lr"], 'momentum': 0.9}

    param_list = []
    if type(model) is Xtractor:
        if model.preprocessor is not None:
            param_list.append({'params': model.preprocessor.parameters(),
                               'weight_decay': model.preprocessor_weight_decay})
        param_list.append({'params': model.sequence_network.parameters(),
                           'weight_decay': model.sequence_network_weight_decay})
        param_list.append({'params': model.stat_pooling.parameters(),
                           'weight_decay': model.stat_pooling_weight_decay})
        param_list.append({'params': model.before_speaker_embedding.parameters(),
                           'weight_decay': model.before_speaker_embedding_weight_decay})
        param_list.append({'params': model.after_speaker_embedding.parameters(),
                           'weight_decay': model.after_speaker_embedding_weight_decay})

    else:
        if model.module.preprocessor is not None:
            param_list.append({'params': model.module.preprocessor.parameters(),
                               'weight_decay': model.module.preprocessor_weight_decay})
        param_list.append({'params': model.module.sequence_network.parameters(),
                           'weight_decay': model.module.sequence_network_weight_decay})
        param_list.append({'params': model.module.stat_pooling.parameters(),
                           'weight_decay': model.module.stat_pooling_weight_decay})
        param_list.append({'params': model.module.before_speaker_embedding.parameters(),
                           'weight_decay': model.module.before_speaker_embedding_weight_decay})
        param_list.append({'params': model.module.after_speaker_embedding.parameters(),
                           'weight_decay': model.module.after_speaker_embedding_weight_decay})

    optimizer = _optimizer(param_list, **_options)

    if train_opts["scheduler"]["type"] == 'CyclicLR':
        cycle_momentum = True
        if train_opts["optimizer"]["type"] == "adam":
            cycle_momentum = False
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer,
                                                      base_lr=train_opts["scheduler"]["base_lr"],
                                                      max_lr=train_opts["lr"],
                                                      step_size_up=train_opts["scheduler"]["step_size_up"],
                                                      step_size_down=None,
                                                      cycle_momentum=cycle_momentum,
                                                      mode=train_opts["scheduler"]["mode"])

    elif train_opts["scheduler"]["type"] == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                         milestones=[10000,50000,100000],
                                                         gamma=0.5)

    elif train_opts["scheduler"]["type"] == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                           step_size=1 * training_loader.__len__(),
                                                           gamma=0.95)

    elif train_opts["scheduler"]["type"] == "StepLR2":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                           step_size=1 * training_loader.__len__(),
                                                           gamma=0.5)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               mode='min',
                                                               factor=0.5,
                                                               patience=3000,
                                                               verbose=True)

    return optimizer, scheduler


def save_model(model, training_monitor, model_opts, training_opts, optimizer, scheduler, epoch):
    """

    :param model:
    :param training_monitor:
    :param model_opts:
    :param training_opts:
    :param optimizer:
    :param scheduler:
    :return:
    """

    best_name = training_opts["best_model_name"]
    tmp_name = training_opts["tmp_model_name"]

    if epoch is not None:
        best_name = best_name + f"_epoch{epoch}"

    # TODO à reprendre
    if type(model) is Xtractor:
        save_checkpoint({
            'epoch': training_monitor.current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': training_monitor.best_accuracy,
            'scheduler': scheduler,
            'speaker_number' : model.speaker_number,
            'model_archi': model_opts,
            'loss': model_opts["loss"]["type"]
        }, training_monitor.is_best, filename=tmp_name, best_filename=best_name)
    else:
        save_checkpoint({
            'epoch': training_monitor.current_epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': training_monitor.best_accuracy,
            'scheduler': scheduler,
            'speaker_number': model.module.speaker_number,
            'model_archi': model_opts,
            'loss': model_opts["loss"]["type"]
        }, training_monitor.is_best, filename=tmp_name, best_filename=best_name)


class AAMScheduler():
    """
    For now we only update margin
    """
    def __init__(self, original_margin, final_margin, final_steps_nb, update_frequency, mode='lin', Tau=1, verbose=True):
        """

        :param final_margin:
        :param num_epochs:
        :param mode: can be linear or exp
        :param verbose:
        """
        self.current_margin = original_margin
        self.original_margin = original_margin
        self.final_margin = final_margin
        self.final_steps_nb = final_steps_nb
        self.update_frequency = update_frequency
        self.mode = mode
        self.Tau = Tau
        self.verbose = verbose
        self._counter = 0

    def __step__(self):
        self._counter += 1

        if self._counter % self.update_frequency == 0:
            # update the parameters
            if self.mode == "lin":
                self.current_margin = self.original_margin + \
                                      (self.final_margin - self.original_margin) * \
                                      (self._counter / self.final_steps_nb)
            else:
                self.current_margin = self.original_margin + \
                                      (self.final_margin - self.original_margin) * \
                                      (1 - numpy.exp(-self._counter / (self.final_steps_nb/7)))

        return self.current_margin


def xtrain(dataset_description,
           model_description,
           training_description,
           **kwargs):
    """
    REFACTORING
    - en cas de redemarrage à partir d'un modele existant, recharger l'optimize et le scheduler
    """

    local_rank = -1
    if "RANK" in os.environ:
        local_rank = int(os.environ['RANK'])

    # Test to optimize
    torch.backends.cudnn.benchmark = True
    torch.autograd.profiler.emit_nvtx(enabled=False)

    dataset_opts, model_opts, training_opts = update_training_dictionary(dataset_description,
                                                                         model_description,
                                                                         training_description,
                                                                         kwargs)

    # Initialize the training monitor
    monitor = TrainingMonitor(output_file=training_opts["log_file"],
                              patience=training_opts["patience"],
                              best_accuracy=0.0,
                              best_eer_epoch=1,
                              best_eer=100,
                              compute_test_eer=training_opts["compute_test_eer"])

    # Make PyTorch Deterministic
    torch.backends.cudnn.deterministic = False
    if training_opts["deterministic"]:
        torch.backends.cudnn.deterministic = True

    # Set all the seeds
    random.seed(training_opts["random_seed"])
    numpy.random.seed(training_opts["numpy_seed"]) # Set the random seed of numpy for the data split.
    torch.manual_seed(training_opts["torch_seed"])
    torch.cuda.manual_seed(training_opts["torch_seed"])

    # Display the entire configurations as YAML dictionaries
    if local_rank < 1:
        monitor.logger.info("\n*********************************\nDataset options\n*********************************\n")
        monitor.logger.info(yaml.dump(dataset_opts, default_flow_style=False))
        monitor.logger.info("\n*********************************\nModel options\n*********************************\n")
        monitor.logger.info(yaml.dump(model_opts, default_flow_style=False))
        monitor.logger.info("\n*********************************\nTraining options\n*********************************\n")
        monitor.logger.info(yaml.dump(training_opts, default_flow_style=False))

    # Initialize the model
    model = get_network(model_opts, local_rank)
    if local_rank < 1:
        monitor.logger.info(model)

    embedding_size = model.embedding_size
    aam_scheduler = None
    #if model.loss == "aam":
    #    aam_scheduler = AAMScheduler(model_opts["loss"]["aam_margin"],
    #                                 final_margin=0.5,
    #                                 final_steps_nb=120000,
    #                                 update_frequency=25000,
    #                                 mode='exp',
    #                                 Tau=1,
    #                                 verbose=True)

    # Set the device and manage parallel processing
    torch.cuda.set_device(local_rank)
    if local_rank >= 0:
        device = torch.device(local_rank)
    else:
        device = torch.device("cuda")

    if training_opts["multi_gpu"]:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.to(device)


    """ [HOW TO] from https://gist.github.com/sgraaf/5b0caa3a320f28c27c12b5efeb35aa4c
        - Add the following line right after "if __name__ == '__main__':" in your main script :
        parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.')
        - Then, in your shell :
        export NUM_NODES=1
        export NUM_GPUS_PER_NODE=2
        export NODE_RANK=0
        export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))
        python -m torch.distributed.launch \
            --nproc_per_node=$NUM_GPUS_PER_NODE \
            --nnodes=$NUM_NODES \
            --node_rank $NODE_RANK \
            train_xvector.py ...
    """
    if training_opts["multi_gpu"]:
        if local_rank < 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank)
    else:
        print("Train on a single GPU")

    # Initialise data loaders
    training_loader, validation_loader,\
    sampler, validation_tar_indices, validation_non_indices = get_loaders(dataset_opts,
                                                                          training_opts,
                                                                          model_opts,
                                                                          local_rank)

    if local_rank < 1:
        monitor.logger.info(f"Start training process")
        monitor.logger.info(f"Use \t{torch.cuda.device_count()} \tgpus")
        monitor.logger.info(f"Use \t{training_opts['num_cpu']} \tcpus")

        monitor.logger.info(f"Validation EER will be measured using")
        monitor.logger.info(f"\t {numpy.sum(validation_tar_indices)} target trials and")
        monitor.logger.info(f"\t {numpy.sum(validation_non_indices)} non-target trials")

    # Create optimizer and scheduler
    optimizer, scheduler = get_optimizer(model, model_opts, training_opts, training_loader)

    scaler = None
    if training_opts["mixed_precision"]:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, training_opts["epochs"] + 1):

        monitor.update(epoch=epoch)

        # Process one epoch and return the current model
        if monitor.current_patience == 0:
            print(f"Stopping at epoch {epoch} for cause of patience")
            break

        sampler.set_epoch(epoch)
        if training_opts["multi_gpu"]:
            torch.distributed.barrier()

        model = train_epoch(model,
                            training_opts,
                            monitor,
                            training_loader,
                            optimizer,
                            scheduler,
                            device,
                            scaler=scaler)
        #                    aam_scheduler=aam_scheduler)

        # Cross validation
        if math.fmod(epoch, training_opts["validation_frequency"]) == 0:
            val_acc, val_loss, val_eer = cross_validation(model,
                                                          validation_loader,
                                                          device,
                                                          [validation_loader.dataset.__len__(), embedding_size],
                                                          validation_tar_indices,
                                                          validation_non_indices,
                                                          training_opts["mixed_precision"])

            test_eer = None
            if training_opts["compute_test_eer"] and local_rank < 1:
                test_eer = test_metrics(model, device, model_opts, dataset_opts, training_opts)

            monitor.update(test_eer=test_eer,
                           val_eer=val_eer,
                           val_loss=val_loss,
                           val_acc=val_acc)

            if local_rank < 1:
                monitor.display()

            # Save the current model and if needed update the best one
            # TODO ajouter une option qui garde les modèles à certaines époques (par exemple avant le changement de LR
            if local_rank < 1:
                save_model(model, monitor, model_opts, training_opts, optimizer, scheduler, epoch)


    for ii in range(torch.cuda.device_count()):
        monitor.logger.info(torch.cuda.memory_summary(ii))

    # TODO gérer l'affichage en utilisant le training_monitor
    if local_rank < 1:
        monitor.display_final()

    return monitor.best_eer


def train_epoch(model,
                training_opts,
                training_monitor,
                training_loader,
                optimizer,
                scheduler,
                device,
                scaler=None,
                clipping=False,
                aam_scheduler=None):
    """

    :param model:
    :param training_opts:
    :param training_monitor:
    :param training_loader:
    :param optimizer:
    :param scheduler:
    :param device:
    :param scaler:
    :param clipping:
    :param aam_scheduler:
    :return:
    """
    model.train()
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    if isinstance(model, Xtractor):
        loss_criteria = model.loss
    else:
        loss_criteria = model.module.loss

    accuracy = 0.0
    running_loss = 0.0
    batch_count = 0
    for batch_idx, (data, target) in enumerate(training_loader):
        data = data.squeeze().to(device)

        target = target.squeeze()
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                if loss_criteria == 'aam':
                    output_tuple, _ = model(data, target=target)
                    output, no_margin_output = output_tuple
                    loss = criterion(output, target)
                elif loss_criteria == 'smn':
                    output_tuple, _ = model(data, target=target)
                    loss, output = output_tuple
                    loss += criterion(output, target)
                elif loss_criteria == 'aps':
                    output_tuple, _ = model(data, target=target)
                    loss, output = output_tuple
                else:
                    output, _ = model(data, target=None)
                    loss = criterion(output, target)

            scaler.scale(loss).backward()
            if clipping:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()

        else:
            if loss_criteria == 'aam':
                output, _ = model(data, target=target)
                loss = criterion(output, target)
            elif loss_criteria == 'aps':
                output_tuple, _ = model(data, target=target)
                cos_sim_matx, output = output_tuple
                loss = criterion(cos_sim_matx, torch.arange(0, int(data.shape[0]/2), device=device)) + criterion(output, target)
            else:
                output, _ = model(data, target=None)
                loss = criterion(output, target)

            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        accuracy += (torch.argmax(no_margin_output.data, 1) == target).sum().cpu()
        batch_count += 1

        if math.fmod(batch_idx, training_opts["log_interval"]) == 0:
            batch_size = target.shape[0]
            training_monitor.update(training_loss=loss.item(),
                                    training_acc=100.0 * accuracy.item() / ((batch_idx + 1) * batch_size))

            training_monitor.logger.info('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.3f}'.format(
                training_monitor.current_epoch,
                batch_idx + 1,
                training_loader.__len__(),
                100. * batch_idx / training_loader.__len__(),
                running_loss / batch_count,
                100.0 * accuracy / (batch_count*target.shape[0])))
            running_loss = 0.0
            accuracy = 0.0
            batch_count = 0
            running_loss = 0.0
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(training_monitor.best_eer)
        else:
            scheduler.step()
        if aam_scheduler is not None:
            model.after_speaker_embedding.margin = aam_scheduler.__step__()
    return model

def cross_validation(model, validation_loader, device, validation_shape, tar_indices, non_indices, mixed_precision=False):
    """

    :param model:
    :param validation_loader:
    :param device:
    :param validation_shape:
    :return:
    """
    model.eval()
    if isinstance(model, Xtractor):
        loss_criteria = model.loss
    else:
        loss_criteria = model.module.loss

    accuracy = 0.0
    loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    embeddings = torch.zeros(validation_shape)
    cursor = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm.tqdm(validation_loader, desc='validation compute', mininterval=1, disable=None)):
            if target.dim() != 1:
                target = target.squeeze()
            target = target.to(device)
            batch_size = target.shape[0]
            data = data.squeeze().to(device)
            with torch.cuda.amp.autocast(enabled=mixed_precision):
                output, batch_embeddings = model(data, target=None, is_eval=True)
                if loss_criteria == 'cce':
                    batch_embeddings = l2_norm(batch_embeddings)
                if loss_criteria == 'smn':
                    batch_embeddings, batch_predictions = output
                else:
                    batch_predictions = output
                accuracy += (torch.argmax(batch_predictions.data, 1) == target).sum()
                loss += criterion(batch_predictions, target)
            embeddings[cursor:cursor + batch_size,:] = batch_embeddings.detach().cpu()
            cursor += batch_size

    local_device = "cpu" if embeddings.shape[0] > 3e4 else device
    embeddings = embeddings.to(local_device)
    scores = torch.einsum('ij,kj', embeddings, embeddings).cpu().numpy()
    negatives = scores[non_indices]
    positives = scores[tar_indices]

    pmiss, pfa = rocch(positives, negatives)
    equal_error_rate = rocch2eer(pmiss, pfa)

    return (100. * accuracy.cpu().numpy() / validation_shape[0],
            loss.cpu().numpy() / ((batch_idx + 1) * batch_size),
            equal_error_rate)


def extract_embeddings(idmap_name,
                       model_filename,
                       data_root_name,
                       device,
                       batch_size=1,
                       file_extension="wav",
                       transform_pipeline={},
                       sliding_window=False,
                       win_duration=3.,
                       win_shift=1.5,
                       num_thread=1,
                       sample_rate=16000,
                       mixed_precision=False,
                       norm_embeddings=True):
    """

    :param idmap_name:
    :param model_filename:
    :param data_root_name:
    :param device:
    :param file_extension:
    :param transform_pipeline:
    :param sliding_window:
    :param win_duration:
    :param win_shift:
    :param num_thread:
    :param sample_rate:
    :param mixed_precision:
    :return:
    """

    if sliding_window:
        batch_size = 1

    # Load the model
    if isinstance(model_filename, str):
        checkpoint = torch.load(model_filename, map_location=device)
        speaker_number = checkpoint["speaker_number"]
        model_opts = checkpoint["model_archi"]
        model_opts["embedding_size"] = 256
        model = Xtractor(speaker_number,
                         model_archi=model_opts["model_type"],
                         loss=model_opts["loss"]["type"],
                         embedding_size=model_opts["embedding_size"])
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = model_filename

    if isinstance(idmap_name, IdMap):
        idmap = idmap_name
    else:
        idmap = IdMap(idmap_name)

    # Create dataset to load the data
    dataset = IdMapSet(idmap_name=idmap,
                       data_path=data_root_name,
                       file_extension=file_extension,
                       transform_pipeline=transform_pipeline,
                       transform_number=0,
                       sliding_window=sliding_window,
                       window_len=win_duration,
                       window_shift=win_shift,
                       sample_rate=sample_rate,
                       min_duration=win_duration
                       )

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=num_thread)

    with torch.no_grad():
        model.eval()
        model.to(device)

        embed = []
        modelset= []
        segset = []
        starts = []

        for idx, (data, mod, seg, start, stop) in enumerate(tqdm.tqdm(dataloader,
                                                                      desc='xvector extraction',
                                                                      mininterval=1,
                                                                      disable=None)):

            if data.dim() > 2:
                data = data.squeeze()

            with torch.cuda.amp.autocast(enabled=mixed_precision):
                tmp_data = torch.split(data,data.shape[0]//(max(1, data.shape[0]//100)))
                for td in tmp_data:
                    _, vec = model(x=td.to(device), is_eval=True, norm_embedding=norm_embeddings)
                    embed.append(vec.detach().cpu())
                modelset.extend(mod * data.shape[0])
                segset.extend(seg * data.shape[0])
                if sliding_window:
                    tmp_start = numpy.arange(0, data.shape[0] * win_shift, win_shift)
                    starts.extend(tmp_start * sample_rate + start.detach().cpu().numpy())
                else:
                    starts.append(start.numpy())

        embeddings = StatServer()
        embeddings.stat1 = numpy.concatenate(embed)
        embeddings.modelset = numpy.array(modelset).astype('>U')
        embeddings.segset = numpy.array(segset).astype('>U')
        embeddings.start = numpy.array(starts).squeeze()
        embeddings.stop = embeddings.start + win_duration
        embeddings.stat0 = numpy.ones((embeddings.modelset.shape[0], 1))

    return embeddings


def extract_embeddings_per_speaker(idmap_name,
                                   model_filename,
                                   data_root_name,
                                   device,
                                   file_extension="wav",
                                   transform_pipeline={},
                                   sample_rate=16000,
                                   mixed_precision=False,
                                   num_thread=1):
    """

    :param idmap_name:
    :param model_filename:
    :param data_root_name:
    :param device:
    :param file_extension:
    :param transform_pipeline:
    :param sample_rate:
    :param mixed_precision:
    :param num_thread:
    :return:
    """
    # Load the model
    if isinstance(model_filename, str):
        checkpoint = torch.load(model_filename, map_location=device)
        speaker_number = checkpoint["speaker_number"]
        model_opts = checkpoint["model_archi"]
        model = Xtractor(speaker_number, model_archi=model_opts["model_type"], loss=model_opts["loss"]["type"])
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = model_filename

    if isinstance(idmap_name, IdMap):
        idmap = idmap_name
    else:
        idmap = IdMap(idmap_name)

    # Create dataset to load the data
    dataset = IdMapSetPerSpeaker(idmap_name=idmap,
                                 data_path=data_root_name,
                                 file_extension=file_extension,
                                 transform_pipeline=transform_pipeline,
                                 sample_rate=sample_rate,
                                 min_duration=1.)

    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=num_thread)

    with torch.no_grad():
        model.eval()
        model.to(device)

        # Get the size of embeddings to extract
        emb_size = model.embedding_size

        # Create the StatServer
        embeddings = StatServer()
        embeddings.modelset = dataset.output_im.leftids
        embeddings.segset = dataset.output_im.rightids
        embeddings.start = dataset.output_im.start
        embeddings.stop = dataset.output_im.stop
        embeddings.stat0 = numpy.ones((embeddings.modelset.shape[0], 1))
        embeddings.stat1 = numpy.ones((embeddings.modelset.shape[0], emb_size))

        # Process the data
        with torch.cuda.amp.autocast(enabled=mixed_precision):
            for idx, (data, mod, seg, start, stop) in enumerate(tqdm.tqdm(dataloader,
                                                                          desc='xvector extraction',
                                                                          mininterval=1)):
                if data.shape[1] > 20000000:
                    data = data[..., :20000000]
                _, vec = model(x=data.to(device), is_eval=True, norm_embedding=True)
                embeddings.stat1[idx, :] = vec.detach().cpu()

    return embeddings

