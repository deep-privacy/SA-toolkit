import damped
import torch
import os

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
        label = torch.zeros((len(y_mapper), 2))  # gender 'f' for female, 'm' for male
        for i, x in enumerate(decoded_y_mapped_label):
            indice = {"f": 0, "m": 1}
            label[i][indice[spk2gender[x]]] = 1
        return label

    return mapper
