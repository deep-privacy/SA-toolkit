import logging

import torch.distributed as dist
from .log import log_handler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)


def init_distributedenv(
    rank: int, world_size: int = 2, ip: str = "0.0.0.0", port: int = 29501
) -> None:
    """Initialize the distributed environment

    In other to enable weight sharing between main task and the domain task
    (executed in another process, and maybe in another node), the torch
    distributed backbend needs to be initialized beforehand.

    Only gloo backend is supported.
        Refer to https://pytorch.org/docs/stable/distributed.html#module-torch.distributed

    This function must be called on the main thread. (in the if-main)

    Args:
        rank (int): unique identifier for a DomainTask (0 if )
        world_size (int): The number of expected domain task.
        ip (str): The ipv4 or ipv6 cluster node address
        port (int): port on which the the tensor will be exchanged
    """

    init_param = {
        "backend": "gloo",
        "init_method": f"tcp://{ip}:{port}",
        # The machine with rank 0 will be used to set up all connections
        "rank": rank,
        "world_size": world_size,
    }
    logger.info(
        f"Initialization of distributed env... [init: {init_param['init_method']}, rank: {init_param['rank']}, world_size: {init_param['world_size']}]"  # noqa
    )
    dist.init_process_group(**init_param)
    dist.is_available()
    logger.info("Distributed env inited!")
