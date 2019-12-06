#!/usr/bin/env python


from damped import utils

utils.init_distributedenv(rank=1)

while True:
    recv_buff = utils.from_fork(rank=0)
    print(recv_buff)
