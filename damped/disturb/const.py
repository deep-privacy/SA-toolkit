import torch

"""
constants used to send meta-data about existing toolkit and domain-tasks

[
 running: (0: no, 1:stop),
 dataset: (train: 0, eval: 1, test: 2),
 trainer_recv: (1: the disturb.fork_ will wait for the backward grad)
 # not used
 # not used
]
"""


def should_stop(meta_data: torch.Tensor) -> bool:
    return meta_data[0] == 1


def stop_signal() -> torch.Tensor:
    sig = torch.tensor([1, 0, 0, 0, 0], dtype=torch.int)
    return sig


def is_eval(meta_data: torch.Tensor) -> bool:
    return meta_data[1] == 1


def eval_signal() -> torch.Tensor:
    sig = torch.tensor([0, 1, 0, 0, 0], dtype=torch.int)
    return sig


def train_signal() -> torch.Tensor:
    sig = torch.tensor([0, 0, 0, 0, 0], dtype=torch.int)
    return sig


def wait_backward() -> torch.Tensor:
    sig = torch.tensor([-1, -1, 1, 0, 0], dtype=torch.int)
    return sig


def no_wait_backward() -> torch.Tensor:
    sig = torch.tensor([-1, -1, 0, 0, 0], dtype=torch.int)
    return sig


def is_wait_backward(meta_data: torch.Tensor) -> bool:
    return meta_data[2] == 1 and meta_data[1] == -1 and meta_data[0] == -1


def is_no_wait_backward(meta_data: torch.Tensor) -> bool:
    return meta_data[2] == 0 and meta_data[1] == -1 and meta_data[0] == -1
