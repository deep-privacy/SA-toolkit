#!/usr/bin/env python


import torch.distributed as dist
import torch

from damped import utils


utils.init_distributedenv(rank=1)

while True:
    exchange_dimensions = torch.zeros(1, dtype=torch.int)  # dimensions (3)
    dist.recv(exchange_dimensions, src=0)

    exchange_size = torch.zeros(  # shape of (B x Tmax X D)
        exchange_dimensions, dtype=torch.int
    )
    dist.recv(exchange_size, src=0)

    recv_buff = torch.empty(  # value of (B x Tmax x D)
        tuple(map(lambda x: int(x), exchange_size.tolist()))
    )  # random value in tensor
    dist.recv(recv_buff, src=0)
    print(recv_buff)
