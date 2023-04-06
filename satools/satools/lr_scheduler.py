import torch
import warnings


class OneCycleLR(torch.optim.lr_scheduler.OneCycleLR):
    """
    torch.optim.lr_scheduler.OneCycleLR, Block total_steps
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
        self._old_get_lrs = None
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

        if self._old_get_lrs == None:
            step_num = self.total_steps

        if step_num > self.total_steps:
            return self._old_get_lrs
            #  print("Usually ends here", flush=True)

        lrs = super().get_lr()
        self._old_get_lrs = lrs
