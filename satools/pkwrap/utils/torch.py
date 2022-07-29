"""
third party functions that complements pytorch
"""


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


from torch.nn.utils.weight_norm import WeightNorm
import copy


def fix_weight_norm_deepcopy(model):
    # Fix bug where deepcopy doesn't work with weightnorm.
    # Taken from https://github.com/pytorch/pytorch/issues/28594#issuecomment-679534348
    orig_deepcopy = getattr(model, "__deepcopy__", None)

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
