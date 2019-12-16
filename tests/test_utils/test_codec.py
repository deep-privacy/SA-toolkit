from damped import utils
import torch


def test_init():
    a = utils.StrIntEncoder.encode("test")
    assert 1952805748 == a
    assert torch.tensor(a).int() == a
    assert utils.StrIntEncoder.decode(a) == "test"
