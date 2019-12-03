import torch
from damped import disturb

disturb.init(expected_domain_tasks=1)

task = disturb.DomainTask(name="speaker_identificaion", to_rank=1)
task.isend(torch.zeros((3, 6, 23, 3, 4, 1)))
