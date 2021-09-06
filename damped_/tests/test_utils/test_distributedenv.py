from damped import utils
import torch.distributed as dist


def test_init():
    utils.init_distributedenv(0, world_size=1, port=6223)
    assert dist.is_initialized()
