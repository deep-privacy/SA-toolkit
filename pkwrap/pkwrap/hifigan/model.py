from dataclasses import dataclass
import logging

from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group
import torch
import os
import glob
from . import dataset

#  torch.set_num_threads(1)
#  torch.set_num_interop_threads(1)


@dataclass
class F0QuantModelOpts:
    mode: str = ""
    checkpoint_path: str = ""
    rank: int = 1
    lr: float = 0.0002
    minibatch_size: int = 64
    init_weight_model: str = ""
    resblock_type: str = "ResBlock1"
    adam_b1: float = 0.8
    adam_b2: float = 0.99
    lr_decay: float = 0.999
    segment_size: float = 16640
    multispkr: str = "single"
    f0_stats: str = "exp/f0_stats.pth"
    num_workers: int = 4
    train_utterances: list = "utt1.wav,utt2.wav"
    test_utterances: list = "utt1.wav,utt2.wav"

    def load_from_config(self, cfg):
        for key, value in cfg.items():
            if hasattr(self, key):
                type_of_value = type(getattr(self, key))
                setattr(self, key, type_of_value(value))
        return self


class F0QuantModel:
    def __init__(self, model_cls, **kwargs):
        self.opts = F0QuantModelOpts()
        self.opts.load_from_config(kwargs)

        self.Net = model_cls
        self.num_gpus = int(os.getenv("WORLD_SIZE", "1"))
        self.call_by_mode()

    def call_by_mode(self):
        if self.opts.mode in ["train", "training"]:
            self.train()

    def train(self):
        device = torch.device("cuda")
        if self.num_gpus > 1:
            device = torch.device("cuda:{:d}".format(self.opts.rank))
            logging.info(
                "Init from distributed training rank: {}".format(self.opts.rank)
            )
            init_process_group(
                backend="nccl",
                init_method="env://",
                rank=self.opts.rank,
            )

        generator = self.Net().to(device)

        if self.opts.init_weight_model == "last":
            pattern = os.path.join(self.opts.checkpoint_path, "g_" + "????????")
            cp_list = glob.glob(pattern)
            if len(cp_list) != 0:
                self.opts.init_weight_model = sorted(cp_list)[-1]
            else:
                self.opts.init_weight_model = ""
        if self.opts.init_weight_model != "":
            init_weight_provided = torch.load(self.opts.init_weight_model)
            init_weight_provided_matched, unmatch = utils.torch.match_state_dict(
                generator.state_dict(), init_weight_provided
            )
            not_inited = generator.load_state_dict(
                init_weight_provided_matched, strict=False
            )
            logging.info(
                "Init from previous model {}, layers not initialized: {}: layers ommited (wrong shape): {}".format(
                    self.opts.init_weight_model,
                    str(not_inited),
                    str(unmatch.keys()),
                )
            )

        if self.opts.rank == 0:
            os.makedirs(self.opts.checkpoint_path, exist_ok=True)
            logging.info("checkpoints directory : " + str(self.opts.checkpoint_path))

        if self.num_gpus > 1:
            generator = DistributedDataParallel(
                generator, device_ids=[self.opts.rank]
            ).to(device)

        optim_g = torch.optim.AdamW(
            generator.parameters(),
            self.opts.lr,
            betas=[self.opts.adam_b1, self.opts.adam_b2],
        )

        steps = 0
        last_epoch = -1
        if self.opts.init_weight_model != "":
            optim_g.load_state_dict(init_weight_provided["optim_g"])
            steps = init_weight_provided["steps"] + 1
            last_epoch = init_weight_provided["epoch"]

        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optim_g, gamma=self.opts.lr_decay, last_epoch=last_epoch
        )

        def sample_interval(feats, lengths, filenames, f0s):
            print(feats.shape, f0s.shape, lengths)
            feats, f0s = dataset.sample_interval(
                [feats, f0s], seq_len=self.opts.segment_size
            )
            print(feats.shape, f0s.shape, lengths)
            lengths = torch.zeros_like(lengths) + self.opts.segment_size
            return feats, lengths, filenames, f0s

        dataloader = torch.utils.data.DataLoader(
            dataset.WavList(self.opts.train_utterances),
            batch_size=self.opts.minibatch_size,
            shuffle=False,
            num_workers=self.opts.num_workers,
            collate_fn=dataset.collate_fn_padd(
                self.opts.f0_stats, get_func=sample_interval
            ),
            persistent_workers=True,
        )

        if self.opts.rank == 0:
            dataloader_test = torch.utils.data.DataLoader(
                dataset.WavList(self.opts.test_utterances),
                batch_size=self.opts.minibatch_size,
                shuffle=False,
                num_workers=self.opts.num_workers,
                collate_fn=dataset.collate_fn_padd(self.opts.f0_stats),
                persistent_workers=True,
            )

        if self.opts.rank == 0:
            logging.info("Caching f0")
            for i, sample in enumerate(dataloader):
                pass
            for i, sample in enumerate(dataloader_test):
                pass
