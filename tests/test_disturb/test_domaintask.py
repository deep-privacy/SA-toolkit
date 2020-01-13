import torch
from torch.multiprocessing import Process

from damped import utils
from damped import disturb


def test_domaintask_creation():
    task = disturb.DomainTask(name="speaker_identificaion", to_rank=1)
    assert task.to_rank == 1


def test_domaintask_sharetensor():
    task = disturb.DomainTask(name="speaker_identificaion", to_rank=1)

    def run(rank, size):
        if rank == task.to_rank:  # process disturb-ed
            disturb.init(port=12121)
            for _ in range(10):
                req = task.isend(torch.zeros(size))
                req.wait()

        else:  # Some server task running on another node
            utils.init_distributedenv(1, port=12121)

            for _ in range(10):
                recv_buff, _ = utils.recv(rank=0)
                assert torch.all(torch.eq(recv_buff, torch.zeros(size)))

    processes = []
    for rank in range(2):  # fork multiple processes for testing (single machine)
        p = Process(target=run, args=(rank, (30, 300, 80)))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        assert p.exitcode == 0  # something went wrong!


def test_domaintask_fork_detach():
    task = disturb.DomainTask(name="speaker_identificaion", to_rank=1)

    def run(rank, size):
        if rank == task.to_rank:  # process disturb-ed
            disturb.init(port=12123)
            for _ in range(10):
                req = task.fork_detach(torch.zeros(size), torch.zeros(size) + 1)
                req.wait()

        else:  # Some server task running on another node
            utils.init_distributedenv(1, port=12123)

            for _ in range(10):
                recv_buff_feat, recv_buff_label, _ = utils.fork_recv(rank=0)
                assert torch.all(torch.eq(recv_buff_feat, torch.zeros(size)))
                assert torch.all(torch.eq(recv_buff_label, torch.zeros(size) + 1))

    processes = []
    for rank in range(2):  # fork multiple processes for testing (single machine)
        p = Process(target=run, args=(rank, (30, 300, 80)))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        assert p.exitcode == 0  # something went wrong!


def test_domaintask_fork_detach_type():
    task = disturb.DomainTask(name="speaker_identificaion", to_rank=1)

    def run(rank, size):
        if rank == task.to_rank:  # process disturb-ed
            disturb.init(port=12123)
            for _ in range(10):
                req = task.fork_detach(
                    torch.zeros(size),
                    torch.zeros(size, dtype=torch.uint8) + 1,
                    dtype=(torch.float32, torch.uint8),
                )
                req.wait()

        else:  # Some server task running on another node
            utils.init_distributedenv(1, port=12123)

            for _ in range(10):
                recv_buff_feat, recv_buff_label, _ = utils.fork_recv(
                    rank=0, dtype=(torch.float32, torch.uint8)
                )
                assert torch.all(
                    torch.eq(recv_buff_feat, torch.zeros(size, dtype=torch.float32))
                )
                assert torch.all(
                    torch.eq(recv_buff_label, torch.zeros(size, dtype=torch.uint8) + 1)
                )
                assert recv_buff_label.dtype == torch.uint8
                assert recv_buff_feat.dtype == torch.float32

    processes = []
    for rank in range(2):  # fork multiple processes for testing (single machine)
        p = Process(target=run, args=(rank, (30, 300, 80)))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        assert p.exitcode == 0  # something went wrong!
