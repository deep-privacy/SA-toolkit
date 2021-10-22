"""
third party functions that complements pytorch
"""
def match_state_dict(
    state_dict_a,
    state_dict_b
):
    """ Filters state_dict_b to contain only states that are present in state_dict_a.

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
