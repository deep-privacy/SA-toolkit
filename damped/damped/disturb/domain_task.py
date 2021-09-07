from dataclasses import dataclass
from typing import Tuple, Optional
import time
from threading import Lock
import os

import torch
import datetime
import torch.distributed as dist

import damped

import logging
from damped.utils import log_handler

from .const import wait_backward, no_wait_backward
from .managed_service import ManagedMemory

logger = logging.getLogger(__name__)
logger.propagate = False
logger.addHandler(log_handler)


INTERVAL_LOG_WAIT_TIME = 4000


@dataclass
class DomainTask(object):
    """
    Object that contains one domain task and the information about his worker.

    A DomainTask can send feature to a worker to perform
    speaker_identification, gender_estimation, age_estimation..


    Example::
        >>> import torch
        >>> from damped import disturb
        >>> disturb.init(expected_domain_tasks=1)  # one task ('speaker_identificaion')
        >>> task = disturb.DomainTask(name="speaker_identificaion", to_rank=1)
        >>> task.isend(torch.zeros((3,3)))
    """

    name: str
    to_rank: int

    def __post_init__(self):
        self._mutex_fork = Lock()  # for fork_detach
        self._mutex_fork_backward = Lock()  # for fork_recv_grad
        self._send_back_grad = False

    def fork_recv_grad(
        self,
        hidden_tensor: torch.Tensor,
        domain_label: torch.Tensor,
        dtype: Optional[Tuple[torch.dtype, torch.dtype]] = (
            torch.float32,
            torch.float32,
        ),
    ):
        """Sends a tensor for a DomainTask (the same way fork_detach works).
        But wait for the backward gradient from the DomainTask.

        (Sync operation as opposed to fork_detach)

        Args:
            hidden_tensor (torch.Tensor): tensor of features
            domain_label (torch.Tensor): tensor of y label
            dtype (Tuple(torch.dtype, torch.dtype), optional): the desired data
                type of sent tensor. The first dtype if for the feature, the
                second if for the label.

        https://discuss.pytorch.org/t/distributed-model-parallelism/10377/2

        Returns:
            The hidden_tensor.grad.data processed by the DomainTask() trainer.py
        """

        if int(os.getenv("DAMPED_N_DOMAIN", 1)) < self.to_rank:
            return work(None)

        with self._mutex_fork_backward:
            if not self._send_back_grad:
                dist.send(
                    torch.tensor(-1, dtype=torch.int), dst=self.to_rank
                )  # indicate for meta-data exchange
                dist.send(wait_backward(), dst=self.to_rank)
            self._send_back_grad = (
                False  # for fork_detach don't notify meta-data (fake)
            )
            req = self.fork_detach(hidden_tensor, domain_label, dtype=dtype)
            self._send_back_grad = True
            req.wait()

            recv_buff, meta_data = damped.utils.recv(rank=self.to_rank)
            assert not meta_data, "fork_recv_grad is not expected to receive meta_data"
            return recv_buff

    def fork_detach(
        self,
        hidden_tensor: torch.Tensor,
        domain_label: torch.Tensor,
        dtype: Optional[Tuple[torch.dtype, torch.dtype]] = (
            torch.float32,
            torch.float32,
        ),
    ):
        """Sends a tensor with a target label for a DomainTask trainer to learn

        Handles threading

        Args:
            hidden_tensor (torch.Tensor): tensor of features
            domain_label (torch.Tensor): tensor of y label
            dtype (Tuple(torch.dtype, torch.dtype), optional): the desired data
                type of sent tensor. The first dtype if for the feature, the
                second if for the label.

        Returns:
            A distributed request object. (call ``wait()`` to block the process
            until the operation is finished)
        """
        ManagedMemory().call_number.value += 1
        start_time = time.time()

        if int(os.getenv("DAMPED_N_DOMAIN", 1)) < self.to_rank:
            return work(None)

        with self._mutex_fork:
            if self._send_back_grad:
                dist.send(
                    torch.tensor(-1, dtype=torch.int), dst=self.to_rank
                )  # indicate for meta-data exchange
                dist.send(no_wait_backward(), dst=self.to_rank)
                self._send_back_grad = False
            self.isend(domain_label, dtype=dtype[1]).wait()
            req = self.isend(hidden_tensor, dtype=dtype[0])

        ManagedMemory().wait_time.value += time.time() - start_time
        return work(req)

    def isend(self, tensor: torch.Tensor, dtype: torch.dtype = torch.float32):
        """Sends a tensor asynchronously.

        Used to send batch of padded hidden state sequences.

        Args:
            tensor (torch.Tensor): Tensor to send to the task worker.
            Example: (B x Tmax x D).
                In speech field:
                    B: batch size
                    Tmax: Utterance
                    D: f-bank features
            dtype (torch.dtype, optional): the desired data type of sent tensor
        Returns:
            A distributed request object. (call ``wait()`` to block the process
            until the operation is finished)

        """
        return self._isend(self.to_rank, tensor, dtype=dtype)

    @staticmethod
    def _isend(dst, tensor: torch.Tensor, dtype: torch.dtype = torch.float32):
        """_isend staticmethod

        implement the logic behind send but can be used without an DomainTask
        object.
        """
        if tensor.is_cuda:
            logger.error("isend only support tensor that are allocated on the CPU!")

        shape = tensor.size()
        #  share the number of dimensions in the tensor (3 in B x Tmax x D)
        dist.send(torch.tensor(len(shape), dtype=torch.int), dst=dst)
        # send the tensor shape for correct a memory allocation on the worker side
        # can be (B x Tmax x D)
        dist.send(torch.tensor(shape, dtype=torch.int), dst=dst)
        req = dist.isend(tensor.to(dtype), dst=dst)
        return req


class work(object):
    """
    work overshadow torch.distributed.Work
    https://github.com/pytorch/pytorch/blob/master/torch/lib/c10d/ProcessGroup.hpp
    """

    def __init__(self, work):
        self._work = work

    def wait(self):
        """
        Waits until request completes. Blocking operation.
        """

        with ManagedMemory().wait_mutex:
            if (
                ManagedMemory().call_number.value % INTERVAL_LOG_WAIT_TIME == 1
                and ManagedMemory().call_number.value != 0
            ):
                wait_time = str(
                    datetime.timedelta(
                        seconds=ManagedMemory().wait_time.value
                        / ManagedMemory().call_number.value
                    )
                )
                logger.debug(f"Waited {wait_time} per fork")
                ManagedMemory().call_number.value = 0
                ManagedMemory().wait_time.value = 0

        if self._work is not None:
            ManagedMemory().call_number.value += 1
            start_time = time.time()
            self._work.wait()
            ManagedMemory().wait_time.value += time.time() - start_time
