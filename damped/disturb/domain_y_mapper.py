from __future__ import annotations
from typing import Optional
from .managed_service import ManagedMemory

import torch


class SingletonMetaDomain(type):
    """
    Internal metaclass
    The Singleton metaclass for DomainLabelMapper
    """

    _instance: Optional[DomainLabelMapper] = None

    def __call__(self) -> DomainLabelMapper:
        if self._instance is None:
            self._instance = super().__call__()
        return self._instance


class DomainLabelMapper(metaclass=SingletonMetaDomain):
    """
    DomainLabelMapper is used to pass information between a toolkit source code
    without having to change functions signature
    """

    def __init__(self):
        self.map = ManagedMemory().domain_label_map # must be init beforehand

    def add(self, key: torch.Tensor, class_value):
        # neither torch.Tensor or list are hashable, using tuple as key
        self.map[tuple(key.tolist())] = class_value

    def get(self, key: torch.Tensor, default="-1"):
        """Save a tensor class value Y associated with a key tensor

        Args:
            key (torch.Tensor): must me unique for the class_value (i.e. this
            first 3 value of a X tensor + the frist value of a task-Y tensor)
            class_value (any): the domain target value
        """
        """Get class label y from a key

        Args:
            key (torch.Tensor): The same key used in DomainLabelMapper().add()
        """
        key = tuple(key.tolist())
        return self.map.pop(key, default)
