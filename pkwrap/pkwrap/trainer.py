"""This module contains training related classes and functions"""
from dataclasses import dataclass
from . import script_utils
from _pkwrap import kaldi
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

@dataclass
class TrainerOpts:
    num_jobs_initial:int = 1
    num_jobs_final: int = 6
    lr_initial: float = 0.01
    lr_final: float = 0.001
    iter_no: int = 0
    num_epochs: int = 6
    train_stage: int = 0
    frames_per_iter: int = 120000
    chunk_width: str = "140"
    cmd: str = 'queue.pl -l q_gpu -V'
    diagnostics_interval: int = 10
    checkpoint_interval: int = 100
    srand: int = 1
    
    def load_from_config(self, cfg):
        for key, value in cfg.items():
            if hasattr(self, key):
                type_of_value = type(getattr(self, key))
                setattr(self, key, type_of_value(value))
        return self

    def load_from_config_file(self, cfg, name="trainer"):
        raise NotImplementedError

@dataclass
class ModelOpts:
    model_file: str = ''
    dirname: str = './'
    left_context: int = 0
    right_context: int = 0
    egs_dir: str = './egs'
    den_graph: str = './den.fst'
    frame_subsampling_factor: int = 3

    def set_dirname(self, dirname, reset_paths=True):
        self.dirname = dirname
        if reset_paths:
            self.egs_dir = os.path.join(self.dirname, 'egs')
            self.den_graph = os.path.join(self.dirname, 'den.fst')

    def load_from_config(self, cfg):
        for key, value in cfg.items():
            if hasattr(self, key):
                type_of_value = type(getattr(self, key))
                setattr(self, key, type_of_value(value))
        return self
