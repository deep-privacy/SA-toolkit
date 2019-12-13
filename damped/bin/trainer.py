#!/usr/bin/env python


import torch
from damped import utils

utils.init_distributedenv(rank=1)

while True:
    recv_buff = utils.fork_recv(rank=0, dtype=(torch.float32, torch.uint8))
    print(recv_buff)
