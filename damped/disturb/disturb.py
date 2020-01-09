#!/usr/bin/env python

from damped import utils
from damped.utils import log_handler
from .managed_service import ManagedMemory
from .const import stop_signal, eval_signal, train_signal

import torch
import torch.distributed as dist

import logging

logger = logging.getLogger(__name__)
logger.propagate = False
logger.addHandler(log_handler)


def init(expected_domain_tasks=1, port=29500) -> None:
    """Initialize the damped distributed environment

    Args:
        expected_domain_tasks (int): The number of expected domain task.
        port (int): port on which the the tensor will be exchanged
    """
    logger.warning("Waiting for domain-task trainer connection")
    utils.init_distributedenv(0, world_size=expected_domain_tasks + 1, port=port)

    # init ManagedMemory
    ManagedMemory()


def stop(domain_tasks: int) -> None:
    """
    Send a stop signal to all domain tasks

    Args:
        domain_tasks (int): the number of domain_tasks used

    """
    for t in range(1, domain_tasks + 1):
        dist.send(
            torch.tensor(-1, dtype=torch.int), dst=t
        )  # indicate for meta-data exchange
        dist.send(stop_signal(), dst=t)


def eval(domain_tasks: int) -> None:
    """
    Put the trainer into evaluation mode

    Args:
        domain_tasks (int): the number of domain_tasks used
    """
    for t in range(1, domain_tasks + 1):
        dist.send(
            torch.tensor(-1, dtype=torch.int), dst=t
        )  # indicate for meta-data exchange
        dist.send(eval_signal(), dst=t)


def train(domain_tasks: int) -> None:
    """
    Put the trainer into training mode

    Args:
        domain_tasks (int): the number of domain_tasks used
    """
    for t in range(1, domain_tasks + 1):
        dist.send(
            torch.tensor(-1, dtype=torch.int), dst=t
        )  # indicate for meta-data exchange
        dist.send(train_signal(), dst=t)
