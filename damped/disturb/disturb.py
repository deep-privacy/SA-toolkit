#!/usr/bin/env python

from damped import utils

import logging
from damped.utils import log_handler
logger = logging.getLogger(__name__)
logger.propagate = False
logger.addHandler(log_handler)


def init(expected_domain_tasks=1, port=29500) -> None:
    """Initialize the damped distributed environment

    Args:
        expected_domain_tasks (int): The number of expected domain task.
        port (int): port on which the the tensor will be exchanged
    """
    logger.warning("Waiting for domain-task trainer connection")
    utils.init_distributedenv(0, world_size=expected_domain_tasks + 1, port=port)
