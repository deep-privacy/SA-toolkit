from damped import utils
import torch


def test_init():
    a = utils.str_int_encoder.encode("test")
    assert 1952805748 == a
    assert torch.tensor(a).int() == a
    assert utils.str_int_encoder.decode(a) == "test"
