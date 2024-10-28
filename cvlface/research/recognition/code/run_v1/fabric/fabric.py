from general_utils import dist_utils
import builtins as __builtin__
builtin_print = __builtin__.print
import torch
from functools import partial
from .sampler import worker_init_fn, DDPWithAttribute
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import os, sys
import traceback


def setup_dataloader_from_dataset(dataset,
                                  is_train,
                                  batch_size,
                                  num_workers,
                                  seed,
                                  fabric,
                                  collate_fn=None):

    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=fabric.local_rank, seed=seed)

    if is_train:
        sampler = DistributedSampler(dataset=dataset, num_replicas=fabric.world_size,
                                     rank=fabric.local_rank, shuffle=True, drop_last=True, seed=seed)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler,
                                collate_fn=collate_fn, worker_init_fn=init_fn,
                                drop_last=True, shuffle=(sampler is None), pin_memory=True, )

    else:
        sampler = DistributedSampler(dataset=dataset, num_replicas=fabric.world_size,
                                     rank=fabric.local_rank, shuffle=False, drop_last=False, seed=seed)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler,
                                collate_fn=collate_fn, worker_init_fn=init_fn,
                                drop_last=False, shuffle=(sampler is None), pin_memory=False, )
    dataloader = fabric.setup_dataloaders(dataloader, use_distributed_sampler=False)
    return dataloader

    # from functools import partial
    # from torch.utils.data.distributed import DistributedSampler
    # from torch.utils.data import DataLoader
    # import random
    # import numpy as np
    # def worker_init_fn(worker_id, num_workers, rank, seed):
    #     # The seed of each worker equals to
    #     # num_worker * rank + worker_id + user_seed
    #     worker_seed = num_workers * rank + worker_id + seed
    #     np.random.seed(worker_seed)
    #     random.seed(worker_seed)
    #     torch.manual_seed(worker_seed)
    #     # needed for speeding up dataloader in torch 2.0.1
    #     os.sched_setaffinity(0, range(os.cpu_count()))
    #
    # local_rank = fabric.local_rank
    # world_size = fabric.world_size
    # batch_size = cfg.trainers.batch_size
    # seed = cfg.trainers.seed
    # sampler = DistributedSampler(dataset=dataset, num_replicas=world_size,
    #                              rank=local_rank, shuffle=True, drop_last=True, seed=seed)
    # init_fn = partial(worker_init_fn, num_workers=num_workers, rank=local_rank, seed=seed)
    # dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler,
    #                         worker_init_fn=init_fn, drop_last=True, shuffle=(sampler is None), pin_memory=True,
    #                         collate_fn=collate_fn)
    # dataloader = fabric.setup_dataloaders(dataloader, use_distributed_sampler=False)



class Fabric():

    def __init__(self, local_rank, world_size, precision, grad_max_norm=5, loggers=(), seed=None, cfg=None):
        self.local_rank = local_rank
        self.world_size = world_size
        self.seed = seed


        self.all_gather =dist_utils.all_gather
        self.barrier = dist_utils.barrier
        self.broadcast = dist_utils.broadcast
        self.print = self.setup_print(local_rank, cfg)

        self.precision = precision
        assert self.precision in ['16-mixed', '32-true']

        if precision == '16-mixed':
            self.amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)
        elif precision == '32-true':
            self.amp = None
        else:
            raise ValueError(f'Unknown precision {precision}')
        self.grad_max_norm = grad_max_norm
        self.device = torch.device('cuda', local_rank)
        self.loggers = loggers
        self.cfg = cfg


    def log_dict(self, x):
        if self.local_rank == 0:
            for logger in self.loggers:
                logger.log_metrics(x)

    def backward(self, loss, optimizer, accumulate=False):

        param_gen = (param for param_group in optimizer.param_groups for param in param_group['params'])

        if not accumulate:
            if self.amp is None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(param_gen, max_norm=self.grad_max_norm)
                optimizer.step()
                optimizer.zero_grad()
            else:
                self.amp.scale(loss).backward()
                self.amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(param_gen, max_norm=self.grad_max_norm)
                self.amp.step(optimizer)
                self.amp.update()
                optimizer.zero_grad()


    def setup(self, model):
        model = model.to(self.device)
        if model.has_trainable_params():
            model_ddp = DDPWithAttribute(
                module=model, broadcast_buffers=False, device_ids=[self.local_rank], bucket_cap_mb=16,
                find_unused_parameters=False)
            return model_ddp
        else:
            return model

    def setup_dataloader_from_dataset(self, dataset, is_train, batch_size, num_workers, collate_fn=None):
        local_rank = self.local_rank
        world_size = self.world_size
        seed = self.seed

        if seed is None:
            init_fn = None
        else:
            init_fn = partial(worker_init_fn, num_workers=num_workers, rank=local_rank, seed=seed)

        if is_train:
            sampler = DistributedSampler(dataset=dataset, num_replicas=world_size,
                                         rank=local_rank, shuffle=True, drop_last=True, seed=seed)
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler,
                                    collate_fn=collate_fn, worker_init_fn=init_fn,
                                    drop_last=True, shuffle=(sampler is None), pin_memory=True, )

        else:
            sampler = DistributedSampler(dataset=dataset, num_replicas=world_size,
                                         rank=local_rank, shuffle=False, drop_last=False, seed=seed)
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler,
                                    collate_fn=collate_fn, worker_init_fn=init_fn,
                                    drop_last=False, shuffle=(sampler is None), pin_memory=False, )

        return dataloader

    def launch(self):
        pass

    def setup_print(self, rank, cfg):
        is_master = rank == 0
        if cfg is not None:
            save_path = os.path.join(cfg.trainers.output_dir, 'run_log.txt') if cfg is not None else './run_log.txt'
            is_debug = cfg.trainers.debug
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            original_print = __builtin__.print
            def custom_print(*args, **kwargs):
                force = kwargs.pop("force", False)
                if is_master or force:
                    builtin_print(*args, **kwargs)
                    if not is_debug:
                        with open(save_path, 'a') as f:
                            original_print(*args, file=f, **kwargs)

            __builtin__.print = custom_print
        else:
            custom_print = __builtin__.print
        return custom_print


