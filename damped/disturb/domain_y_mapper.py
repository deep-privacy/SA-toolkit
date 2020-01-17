from __future__ import annotations
from typing import Dict, Optional, Callable
from .managed_service import ManagedMemory

import torch

import logging
from damped.utils import log_handler
logger = logging.getLogger(__name__)
logger.propagate = False
logger.addHandler(log_handler)


class SingletonMetaDomain(type):
    """
    Internal metaclass
    The Singleton metaclass for DomainLabelMapper
    """

    def __call__(self, name: str) -> DomainLabelMapper:
        _instances = ManagedMemory().domain_label_mappers
        if name not in _instances:
            _instance = super().__call__(name)
            _instances[name] = _instance
            return _instance
        return _instances[name]


class DomainLabelMapper(metaclass=SingletonMetaDomain):
    """
    DomainLabelMapper is used to pass information in between a toolkit source
    code without having to change functions signature
    """

    _name: str
    map: Dict

    def __init__(self, name: str):
        """

        Args:
            name (str): The name of the this DomainLabelMapper
        """
        # TODO: There is no separation in between different DomainLabelMapper (name)
        self.map = ManagedMemory().domain_label_map  # must be init beforehand
        self._name = name

    def add(self, key: torch.Tensor, class_value: any) -> None:
        """Save a tensor class value Y associated with a key tensor

        Args:
            key (torch.Tensor): must me unique for the class_value (i.e.
            first 3 values of a X tensor)
            class_value (any): the domain target value
        """
        # neither torch.Tensor or list are hashable, using tuple as key
        self.map[tuple(key.tolist())] = class_value

    def get(
        self,
        key: torch.Tensor,
        default="-1",
        codec: Optional[Callable] = None,
        delete=True,
    ) -> any:
        """Get class label y from a key

        Args:
            key (torch.Tensor): The same key used in DomainLabelMapper.add()
            default (any, optional): The default value to use if the tensor isn't found in the map
            codec (Callable, optional): apply a transformation on the value found in the map
            delete (bool, optional): defualt True, remove value from hashable
        """
        key = tuple(key.tolist())
        if key in self.map:
            y = self.map[key]
            if delete:
                del self.map[key]
            if codec is not None:
                y = codec(y)
            return y
        logging.warning("Label not found for key!")
        return default
