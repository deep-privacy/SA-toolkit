import logging
from pprint import pformat

import torch.distributed as dist

logger = logging.getLogger("damped.utils")


def init_distributedenv(rank: int, world_size=2, port=29500) -> None:
    """Initialize the distributed environment

    In other to enable weight sharing between main task and the domain task
    (executed in another process, and maybe in another node), the torch
    distributed backbend needs to be initialized beforehand.

    Only gloo backend is supported.
        Refer to https://pytorch.org/docs/stable/distributed.html#module-torch.distributed

    Args:
        rank (int): unique identifier for a DomainTask (0 if )
        expected_domain_tasks (int): The number of expected domain task.
        port (int): port on which the the tensor will be exchanged
    """

    init_param = {
        "backend": "gloo",
        "init_method": f"tcp://0.0.0.0:{port}",
        # The machine with rank 0 will be used to set up all connections
        "rank": rank,
        "world_size": world_size,
    }
    logger.info(f"Initialization of distributed env {pformat(init_param)}")
    dist.init_process_group(**init_param)
    logger.info("Distributed env inited!")
