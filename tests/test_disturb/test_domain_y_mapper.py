from damped import disturb
import torch


def test_domain_y_storage():
    xs_pad = torch.empty((3, 3, 3)).numpy()
    ys_pad = torch.empty((3, 2)).numpy()

    key_x = torch.tensor(xs_pad[0][0][:3], dtype=torch.float)
    key_y = torch.tensor(ys_pad[0][:2], dtype=torch.float)
    key = torch.cat((key_x, key_y))

    disturb.DomainLabelMapper().add(key, "TEST")

    assert disturb.DomainLabelMapper().get(key) == "TEST"
    assert len(disturb.DomainLabelMapper().map) == 0
