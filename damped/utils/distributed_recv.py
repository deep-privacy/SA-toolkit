import torch
import torch.distributed as dist
from typing import Optional, Tuple


def fork_recv(
    rank: int,
    dtype: Optional[Tuple[torch.dtype, torch.dtype]] = (torch.float32, torch.float32),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get label and feature from forked task

    Args:
        rank (int): rank of the note in the distributed env
        dtype (Tuple(torch.dtype, torch.dtype), optional): the desired data
            type of sent tensor. The first dtype if for the feature, the
            second if for the label.

    Returns:
        Tuple(torch.Tensor, torch.Tensor): the related features and class label
    """
    label, is_meta_data = recv(rank=0, dtype=dtype[1])
    if is_meta_data:
        return (None, label, is_meta_data)
    features, _ = recv(rank=0, dtype=dtype[0])
    return (features, label, is_meta_data)


def recv(
    rank: int, dtype: Optional[torch.dtype] = torch.float32
) -> Tuple[torch.Tensor, bool]:
    """Receive a tensor from a DomainTask

    Args:
        rank (int): rank of the note in the distributed env
        dtype (torch.dtype, optional): the desired data type of received tensor

    Returns:
        Tuple(torch.Tensor, bool): [data value received, is meta-data]
    """
    exchange_dimensions = torch.zeros(1, dtype=torch.int)  # dimensions (eg: 3)
    dist.recv(exchange_dimensions, src=rank)

    # a negative value of exchange_dimensions indicate a meta-data exchange
    if exchange_dimensions[0] == -1:
        buff_meta_data = torch.zeros(5, dtype=torch.int)
        dist.recv(buff_meta_data, src=rank)
        return buff_meta_data, True

    exchange_size = torch.zeros(  # shape of (eg: B x Tmax X D)
        exchange_dimensions, dtype=torch.int
    )
    dist.recv(exchange_size, src=rank)

    recv_buff = torch.empty(  # value of (eg: B x Tmax x D)
        *exchange_size.tolist(), dtype=dtype,
    )  # random value in tensor
    dist.recv(recv_buff, src=rank)
    return recv_buff, False
