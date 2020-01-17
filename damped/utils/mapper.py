import damped
import torch
import os

import logging
from damped.utils import log_handler

logger = logging.getLogger(__name__)
logger.propagate = False
logger.addHandler(log_handler)

"""
Mappers for different task.
A damped.disturb domain task may use task independent y label.
Those y label might need to be mapped back to the real task label.

Example:
    The Librispeech egs sends y label corresponding to the user identifier.
    The user identifier needs to be mapped back to the gender class {f,m}
"""


def gender_mapper(dir_path):
    # Domain label
    spk2gender_lines = [
        line.rstrip("\n").split(" ")
        for line in open(os.path.join(dir_path, "..", "data", "spk2gender"))
    ]
    spk2gender = dict(map(lambda x: (x[0], x[1]), spk2gender_lines))
    print("spk2gender: ")
    for k, v in list(spk2gender.items())[:5]:
        print(f"  spk: {k} -> gender {v}")

    # sent y_mapper (from damped.disturb) to y label (for branch task training)
    def mapper(y_mapper):
        decoded_y_mapped_label = list(
            map(
                lambda x: damped.utils.codec.str_int_encoder.decode(x),
                y_mapper.tolist(),
            )
        )
        label = torch.zeros(len(y_mapper), dtype=torch.long)
        # gender 'f' for female, 'm' for male
        indice = {"f": 0, "m": 1}
        for i, x in enumerate(decoded_y_mapped_label):
            if x == "-1":
                logger.warning("Warn: y_mapper not found")
                continue
            label[i] = indice[spk2gender[x]]

        return label

    return mapper


def spkid_mapper(dir_path):
    # Domain label
    spk2gender_lines = [
        line.rstrip("\n").split(" ")
        for line in open(os.path.join(dir_path, "..", "data", "spk2id"))
    ]
    spk2id = dict(map(lambda x: (x[0], x[1]), spk2gender_lines))
    spk_number = len(spk2id.items())
    print(f"Total speaker: {spk_number}")

    # sent y_mapper (from damped.disturb) to y label (for branch task training)
    def mapper(y_mapper):
        decoded_y_mapped_label = list(
            map(
                lambda x: damped.utils.codec.str_int_encoder.decode(x),
                y_mapper.tolist(),
            )
        )
        label = torch.zeros(len(y_mapper), dtype=torch.long)
        for i, x in enumerate(decoded_y_mapped_label):
            if x == "-1":
                logger.warning("Warn: y_mapper not found")
                continue
            label[i] = int(spk2id[x])

        return label

    return mapper
