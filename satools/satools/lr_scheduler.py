import math
import warnings

import torch
from torch.optim.lr_scheduler import _LRScheduler


class OneCycleLR(torch.optim.lr_scheduler.OneCycleLR):
    """
    torch.optim.lr_scheduler.OneCycleLR, don't raise at total_steps
    """
    def __init__(self,
             optimizer,
             max_lr,
             total_steps=None,
             pct_start=0.3,
             anneal_strategy='cos',
             cycle_momentum=True,
             base_momentum=0.85,
             max_momentum=0.95,
             div_factor=25.,
             final_div_factor=1e4,
             three_phase=False,
             last_epoch=-1,
             verbose=False):
        super().__init__(optimizer=optimizer,
                         max_lr=max_lr,
                         total_steps=total_steps,
                         pct_start=pct_start,
                         anneal_strategy=anneal_strategy,
                         cycle_momentum=cycle_momentum,
                         base_momentum=base_momentum,
                         max_momentum=max_momentum,
                         div_factor=div_factor,
                         final_div_factor=final_div_factor,
                         three_phase=three_phase,
                         last_epoch=last_epoch,
                         verbose=verbose)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.step()

    def get_lr(self):
        step_num = self.last_epoch

        if step_num >= self.total_steps:
            if hasattr(self, "_last_lr"):
                return self._last_lr
            #  print("Usually raise here", flush=True)

        lrs = super().get_lr()

        return lrs


class CosineAnnealingWarmRestartsWithDecayAndLinearWarmup(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:
    """

    def __init__(self, optimizer, first_cycle_steps, cycle_mult=1, min_lr=0, last_epoch=-1, verbose=False, warmup_steps=350, decay=1):
        if first_cycle_steps <= 0 or not isinstance(first_cycle_steps, int):
            raise ValueError("Expected positive integer first_cycle_steps, but got {}".format(first_cycle_steps))
        if cycle_mult < 1 or not isinstance(cycle_mult, (float, int)):
            raise ValueError("Expected integer cycle_mult >= 1, but got {}".format(cycle_mult))
        self.first_cycle_steps = first_cycle_steps
        self.T_i = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.min_lr = min_lr
        self.T_cur = last_epoch

        # Decay attributes
        self.decay = decay

        # Warmup attributes
        self.warmup_steps = warmup_steps
        self.current_steps = 0

        super(CosineAnnealingWarmRestartsWithDecayAndLinearWarmup, self).__init__(optimizer, last_epoch, verbose)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.step()

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        return [
            (self.current_steps / self.warmup_steps) *
            (self.min_lr + (base_lr - self.min_lr) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2)
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        """Step could be called after every batch update"""
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if self.T_cur + 1 == self.T_i:
            if self.verbose:
                print("multiplying base_lrs by {:.4f}".format(self.decay))
            self.base_lrs = [base_lr * self.decay for base_lr in self.base_lrs]

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1

            if self.current_steps < self.warmup_steps:
                self.current_steps += 1

            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.cycle_mult

        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
