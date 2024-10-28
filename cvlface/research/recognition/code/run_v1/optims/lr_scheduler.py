from torch.optim.lr_scheduler import _LRScheduler
import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
import numpy as np
from torch import nn

def param_groups_weight_decay(
        named_parameters,
        weight_decay=1e-5,
        no_weight_decay_list=(),
        no_weight_decay_value=0.0,
):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in named_parameters:
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': no_decay, 'weight_decay': no_weight_decay_value},
        {'params': decay, 'weight_decay': weight_decay}]


def make_scheduler(cfg, opt):

    if cfg.optims.scheduler == 'poly_2':
        print('poly_2 scheduler')
        lr_scheduler = PolyScheduler(
            optimizer=opt,
            base_lr=cfg.optims.lr,
            max_steps=cfg.trainers.total_step,
            warmup_steps=cfg.trainers.warmup_step,
            last_epoch=-1,
            power=2
        )
    elif cfg.optims.scheduler == 'poly_0':
        print('poly_0 scheduler')
        lr_scheduler = PolyScheduler(
            optimizer=opt,
            base_lr=cfg.optims.lr,
            max_steps=cfg.trainers.total_step,
            warmup_steps=cfg.trainers.warmup_step,
            last_epoch=-1,
            power=0
        )
    elif cfg.optims.scheduler == 'cosine':
        print('cosine scheduler')
        lr_scheduler = CosineLRScheduler(opt,
                                         t_initial=cfg.trainers.total_step - cfg.trainers.warmup_step,
                                         warmup_t=cfg.trainers.warmup_step,
                                         warmup_lr_init=0,
                                         warmup_prefix=True)
    elif cfg.optims.scheduler == 'step':
        print('step scheduler')
        steps_per_epoch = cfg.trainers.total_step // cfg.optims.num_epoch
        step_milestones = [mile * steps_per_epoch for mile in cfg.optims.lr_milestones]
        lr_scheduler = StepScheduler(optimizer=opt,
                                     base_lr=cfg.optims.lr,
                                     max_steps=cfg.trainers.total_step,
                                     warmup_steps=cfg.trainers.warmup_step,
                                     lr_milestones=step_milestones,
                                     lr_lambda=cfg.optims.lr_lambda)

    else:
        raise ValueError('')

    return lr_scheduler


def scheduler_step(scheduler, global_step):
    if isinstance(scheduler, _LRScheduler):
        scheduler.step()
    else:
        scheduler.step(global_step)

def get_last_lr(optimizer):
    lrs = [group['lr'] for group in optimizer.param_groups]
    return float(np.mean(lrs))


class PolyScheduler(_LRScheduler):
    def __init__(self, optimizer, base_lr, max_steps, warmup_steps, last_epoch=-1, power=2):
        self.base_lr = base_lr
        self.warmup_lr_init = 0.0001
        self.max_steps: int = max_steps
        self.warmup_steps: int = warmup_steps
        self.power = power
        super(PolyScheduler, self).__init__(optimizer, -1, False)
        self.last_epoch = last_epoch

    def get_warmup_lr(self):
        alpha = float(self.last_epoch) / float(self.warmup_steps)
        return [self.base_lr * alpha for _ in self.optimizer.param_groups]

    def get_lr(self):
        if self.last_epoch == -1:
            return [self.warmup_lr_init for _ in self.optimizer.param_groups]
        if self.last_epoch < self.warmup_steps:
            return self.get_warmup_lr()
        else:
            alpha = pow(
                1
                - float(self.last_epoch - self.warmup_steps)
                / float(self.max_steps - self.warmup_steps),
                self.power,
            )
            return [self.base_lr * alpha for _ in self.optimizer.param_groups]



class StepScheduler(_LRScheduler):
    def __init__(self, optimizer, base_lr, max_steps, warmup_steps, warmup_lr_init=0.0001, lr_milestones=[], lr_lambda=0.1, last_epoch=-1):
        self.base_lr = base_lr
        self.warmup_lr_init = warmup_lr_init
        self.max_steps: int = max_steps
        self.warmup_steps: int = warmup_steps
        super(StepScheduler, self).__init__(optimizer, -1, False)
        self.last_epoch = last_epoch
        self.lr_milestones = lr_milestones
        self.lr_lambda = lr_lambda

    def get_warmup_lr(self):
        alpha = float(self.last_epoch) / float(self.warmup_steps)
        return [self.base_lr * alpha for _ in self.optimizer.param_groups]

    def get_lr(self):
        if self.last_epoch == -1:
            return [self.warmup_lr_init for _ in self.optimizer.param_groups]
        if self.last_epoch < self.warmup_steps:
            return self.get_warmup_lr()
        else:
            alpha = 1.0
            for milestone in self.lr_milestones:
                if self.last_epoch > milestone:
                    alpha = alpha * self.lr_lambda
            return [self.base_lr * alpha for _ in self.optimizer.param_groups]




if __name__ == '__main__':
    print('')
    model = torch.nn.Linear(5,5)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
    base_lr = 0.001
    max_steps = 26
    warmup_steps = 10

    lr_milestones = [12, 20, 24]
    lr_lambda = 0.1
    scheduler = StepScheduler(optimizer, base_lr, max_steps, warmup_steps=warmup_steps, warmup_lr_init=0.0, lr_milestones=lr_milestones, lr_lambda=lr_lambda)
    lrs = []
    for step in range(max_steps):
        optimizer.step()
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        lrs.append(lr)

    from matplotlib import pyplot as plt
    plt.plot(lrs)
    plt.show()

    scheduler = PolyScheduler(optimizer, base_lr, max_steps, warmup_steps, power=0)
    lrs = []
    for step in range(max_steps):
        optimizer.step()
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        lrs.append(lr)

    from matplotlib import pyplot as plt
    plt.plot(lrs)
    plt.show()

    scheduler = CosineLRScheduler(optimizer,
                                  t_initial=max_steps - warmup_steps,
                                  warmup_t=warmup_steps,
                                  warmup_lr_init=0,
                                  warmup_prefix=True)

    lrs = []
    for step in range(max_steps):
        optimizer.step()
        scheduler_step(scheduler, step)
        lr = get_last_lr(optimizer)
        lrs.append(lr)

    from matplotlib import pyplot as plt
    plt.plot(lrs)
    plt.show()