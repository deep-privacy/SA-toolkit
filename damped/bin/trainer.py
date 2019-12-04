#!/usr/bin/env python


import torch.distributed as dist
import torch

from damped import utils

shared_tensor_shape = (10, 600, 1024)

utils.init_distributedenv(rank=1)

while True:
    exchange_size = torch.tensor(shared_tensor_shape)

    recv_buff = torch.empty(  # value of (B x Tmax x D)
        tuple(map(lambda x: int(x), shared_tensor_shape))
    )  # random value in tensor
    dist.recv(recv_buff, src=0)
    print(recv_buff)
