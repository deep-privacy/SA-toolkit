"""
third party functions that complements pytorch
"""

import copy

import torch

def seed_worker(seed_val):
    """
    Function that initialize the random seed

    :param seed_val: not used
    """
    import torch
    import numpy
    import random
    worker_seed = torch.initial_seed() % 2 ** 32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def match_state_dict(state_dict_a, state_dict_b):
    """Filters state_dict_b to contain only states that are present in state_dict_a.

    Matching happens according to two criteria:
        - Is the key present in state_dict_a?
        - Does the state with the same key in state_dict_a have the same shape?

    Returns
        (matched_state_dict, unmatched_state_dict)

        States in matched_state_dict contains states from state_dict_b that are also
        in state_dict_a and unmatched_state_dict contains states that have no
        corresponding state in state_dict_a.

        In addition: state_dict_b = matched_state_dict U unmatched_state_dict.
    """
    matched_state_dict = {
        key: state
        for (key, state) in state_dict_b.items()
        if key in state_dict_a and state.shape == state_dict_a[key].shape
    }
    unmatched_state_dict = {
        key: state
        for (key, state) in state_dict_b.items()
        if key not in matched_state_dict
    }
    return matched_state_dict, unmatched_state_dict


def get_one_hot_str_from_tensor(tensor):
    index = torch.where(tensor == 1)
    index_str = str(index[0].item()) if index[0].numel() > 0 else "Value not found"
    return index_str


def fix_weight_norm_deepcopy(model):
    # Fix bug where deepcopy doesn't work with weightnorm.
    # Taken from https://github.com/pytorch/pytorch/issues/28594#issuecomment-679534348
    orig_deepcopy = getattr(model, "__deepcopy__", None)

    from torch.nn.utils.weight_norm import WeightNorm

    def __deepcopy__(model, memo):
        # save and delete all weightnorm weights on self
        weights = {}
        for hook in model._forward_pre_hooks.values():
            if isinstance(hook, WeightNorm):
                weights[hook.name] = getattr(model, hook.name)
                delattr(model, hook.name)
        # remove this deepcopy method, restoring the object's original one if necessary
        __deepcopy__ = model.__deepcopy__
        if orig_deepcopy:
            model.__deepcopy__ = orig_deepcopy
        else:
            del model.__deepcopy__
        # actually do the copy
        result = copy.deepcopy(model)
        # restore weights and method on self
        for name, value in weights.items():
            setattr(model, name, value)
        model.__deepcopy__ = __deepcopy__
        return result

    # bind __deepcopy__ to the weightnorm'd layer
    model.__deepcopy__ = __deepcopy__.__get__(model, model.__class__)


class WrappedTorchDDP(torch.nn.Module):
    """
    Wrap a DistributedDataParallel module and forward requests for missing
    attributes to the module wrapped by DDP (the twice-wrapped module).
    Also forward calls to :func:`state_dict` and :func:`load_state_dict`.
    Usage::
        module.xyz = "hello world"
        wrapped_module = DistributedDataParallel(module, **ddp_args)
        wrapped_module = WrappedTorchDDP(wrapped_module)
        assert wrapped_module.xyz == "hello world"
        assert wrapped_module.state_dict().keys() == module.state_dict().keys()
    Args:
        module (nn.Module): module to wrap

    """

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        assert hasattr(
            module, "module"
        ), "WrappedTorchDDP expects input to wrap another module"
        self.module = module

    def __getattr__(self, name):
        """Forward missing attributes to twice-wrapped module."""
        try:
            # defer to nn.Module's logic
            return super().__getattr__(name)
        except AttributeError:
            try:
                # forward to the once-wrapped module
                return getattr(self.module, name)
            except AttributeError:
                # forward to the twice-wrapped module
                return getattr(self.module.module, name)

    def state_dict(self, *args, **kwargs):
        """Forward to the twice-wrapped module."""
        return self.module.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        """Forward to the twice-wrapped module."""
        return self.module.module.load_state_dict(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

