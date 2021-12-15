import torch.distributed as dist


def is_available():
    return dist.is_initialized()


def broadcast(tensor, src):
    if tensor.is_cuda and is_available():
        return dist.broadcast(tensor, src)


def all_reduce(tensor):
    if tensor.is_cuda and is_available():
        return dist.all_reduce(tensor, dist.ReduceOp.SUM)
