from dataclasses import dataclass

import torch
import torch.distributed as dist

import logging
from damped.utils import log_handler
from threading import Lock

logger = logging.getLogger(__name__)
logger.propagate = False
logger.addHandler(log_handler)


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
        self._mutex_fork = Lock()

    def fork_detach(self, hidden_tensor: torch.Tensor, domain_label: torch.Tensor):
        """Sends a tensor with a target label for a DomainTask trainer to learn

        Handles threading

        Args:
            hidden_tensor (torch.Tensor): tensor of features
            domain_label (torch.Tensor): tensor of y label

        Returns:
            A distributed request object. (call ``wait()`` to block the process
            until the operation is finished)
        """
        with self._mutex_fork:
            self.isend(domain_label).wait()
            req = self.isend(hidden_tensor)

        return req

    def isend(self, tensor: torch.Tensor):
        """Sends a tensor asynchronously.

        Used to send batch of padded hidden state sequences.

        Args:
            tensor (torch.Tensor): Tensor to send to the task worker (B x Tmax x D).
                In speech field:
                    B: batch size
                    Tmax: Utterance
                    D: f-bank features
        Returns:
            A distributed request object. (call ``wait()`` to block the process
            until the operation is finished)

        """
        if tensor.is_cuda:
            logger.error("isend only support tensor that are allocated on the CPU!")

        shape = tensor.size()
        #  share the number of dimensions in the tensor (3 in B x Tmax x D)
        dist.send(torch.tensor(len(shape), dtype=torch.int), dst=self.to_rank)
        # send the tensor shape for correct a memory allocation on the worker side
        # can be (B x Tmax x D)
        dist.send(torch.tensor(shape, dtype=torch.int), dst=self.to_rank)
        req = dist.isend(tensor, self.to_rank)
        return req
