#!/usr/bin/env python


from damped import utils

utils.init_distributedenv(rank=1)

while True:
    recv_buff = utils.fork_recv(rank=0)
    print(recv_buff)
