from dataclasses import dataclass

import torch
import torch.distributed as dist


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

    def isend(self, tensor: torch.Tensor):
        """Sends a tensor asynchronously.

        Usually used to send batch of padded hidden state sequences.

        Args:
            tensor (torch.Tensor): Tensor to send to the task worker (B x Tmax x D).
                In speech field:
                    B: batch size
                    Tmax: Utterance
                    D: f-bank features

        """
        shape = tensor.size()
        #  share the number of dimensions in the tensor (3 in B x Tmax x D)
        dist.send(torch.tensor(len(shape), dtype=torch.int), dst=self.to_rank)
        # send the tensor shape for correct a memory allocation on the worker side
        # can be (B x Tmax x D)
        dist.send(torch.tensor(shape, dtype=torch.int), dst=self.to_rank)
        self.send_req = dist.isend(tensor, self.to_rank)

    def wait(self):
        """Blocks the process until the operation previous isend is finished. """
        self.send_req.wait()
