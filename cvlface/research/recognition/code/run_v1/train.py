import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)
import os, sys

sys.path.append(os.path.join(root))
import numpy as np
np.bool = np.bool_  # fix bug for mxnet 1.9.1
np.object = np.object_
np.float = np.float_

import pandas as pd
import torch
import config
from config import Config
from models import get_model
from classifiers import get_classifier
from aligners import get_aligner
from losses import get_margin_loss
from dataset import get_train_dataset, visualize_dataset, set_epoch
from evaluations import get_evaluator_by_name
from general_utils import random_utils, os_utils
from optims.optims import make_optimizer
from lightning.fabric.loggers import CSVLogger
from lightning.pytorch.loggers import WandbLogger
from optims.lr_scheduler import make_scheduler, scheduler_step, get_last_lr
from pipelines import pipeline_from_config, pipeline_from_name
import omegaconf
import lovely_tensors as lt
lt.monkey_patch()
from tqdm import tqdm
from evaluations import IsBestTracker, summary
import time
from lightning.fabric import Fabric
from pefts import apply_peft
from general_utils.dist_utils import verify_ddp_weights_equal
from functools import partial
from fabric.fabric import setup_dataloader_from_dataset


if __name__ == '__main__':


    cfg: Config = config.init(root)
    torch.set_float32_matmul_precision(cfg.trainers.float32_matmul_precision)
    print('matmul precision', cfg.trainers.float32_matmul_precision)
    print('precision', cfg.trainers.precision)

    random_utils.setup_seed(seed=cfg.trainers.seed, cuda_deterministic=False)

    loggers = []
    csv_logger = CSVLogger(root_dir=cfg.trainers.output_dir, flush_logs_every_n_steps=1)
    loggers.append(csv_logger)
    if cfg.trainers.using_wandb:
        wandb_logger = WandbLogger(project=cfg.trainers.task, save_dir=cfg.trainers.output_dir,
                                   name=os.path.basename(cfg.trainers.output_dir),
                                   log_model=False)
        loggers.append(wandb_logger)

    # grad_max_norm?
    fabric = Fabric(precision=cfg.trainers.precision,
                    loggers=loggers,
                    accelerator="auto",
                    strategy="ddp",
                    devices=cfg.trainers.num_gpu)
    fabric.seed_everything(cfg.trainers.seed)
    if cfg.trainers.num_gpu == 1:
        fabric.launch()
    fabric.setup_dataloader_from_dataset = partial(setup_dataloader_from_dataset, fabric=fabric, seed=cfg.trainers.seed)

    cfg.trainers.local_rank = fabric.local_rank
    cfg.trainers.world_size = fabric.world_size
    print = fabric.print

    # get model
    model = get_model(cfg.models, cfg.trainers.task)
    train_transform = model.make_train_transform()
    test_transform = model.make_test_transform()

    # get dataloader
    dataset, label_mapping = get_train_dataset(cfg.dataset, train_transform, cfg.data_augs, local_rank=cfg.trainers.local_rank)
    dataloader = fabric.setup_dataloader_from_dataset(dataset=dataset,
                                                      is_train=True,
                                                      batch_size=cfg.trainers.batch_size,
                                                      num_workers=cfg.trainers.num_workers)
    cfg.trainers.total_batch_size = cfg.trainers.batch_size * cfg.trainers.world_size
    batch_length = len(dataloader.dataset) // cfg.trainers.total_batch_size
    batch_length = batch_length if cfg.trainers.limit_num_batch <= 0 else cfg.trainers.limit_num_batch
    cfg.trainers.warmup_step = batch_length * cfg.optims.warmup_epoch
    cfg.trainers.total_step = batch_length * cfg.optims.num_epoch
    if fabric.local_rank == 0:
        visualize_dataset(dataloader, os.path.join(cfg.trainers.output_dir, 'train_data.png'))


    # get classifier
    margin_loss_fn = get_margin_loss(cfg.losses)

    extra_classes = 0
    classifier = get_classifier(cfg.classifiers,
                                margin_loss_fn=margin_loss_fn,
                                model_cfg=cfg.models,
                                num_classes=cfg.dataset.num_classes+extra_classes,
                                rank=fabric.local_rank,
                                world_size=fabric.world_size)

    # get aligner
    aligner = get_aligner(cfg.aligners)

    # apply peft if needed
    model, classifier = apply_peft(cfg.pefts, model=model, classifier=classifier, data_cfg=cfg.dataset, label_mapping=label_mapping)

    # get optimizer
    optimizer = make_optimizer(cfg, model, classifier, aligner)
    lr_scheduler = make_scheduler(cfg, optimizer)

    # prepare accelerator
    if model.has_trainable_params():
        model, optimizer = fabric.setup(model, optimizer)
    else:
        model = model.to(fabric.device)
        dummy_model = torch.nn.Linear(1, 1).to(fabric.device)
        dummy_model, optimizer = fabric.setup(dummy_model, optimizer)
    if classifier is not None:
        if classifier.apply_ddp:
            classifier = fabric.setup(classifier)
        else:
            classifier = classifier.to(fabric.device)  # no ddp as it divides fc into multiple GPUs
    if aligner.has_trainable_params():
        aligner = fabric.setup(aligner)
    elif aligner is not None:
        aligner = aligner.to(fabric.device)


    verify_ddp_weights_equal(model)
    if classifier is not None:
        verify_ddp_weights_equal(classifier)

    # make train pipe (after accelerator setup)
    train_pipeline = pipeline_from_config(cfg.pipelines, model, classifier, aligner, optimizer, lr_scheduler)
    train_pipeline.integrity_check(dataloader.dataset)

    # make inference pipe (after accelerator setup)
    eval_pipeline = pipeline_from_name(cfg.pipelines.eval_pipeline_name, model, aligner)
    eval_pipeline.integrity_check(dataloader.dataset.color_space)

    # evaluation callbacks
    evaluators = []
    for name, info in cfg.evaluations.per_epoch_evaluations.items():
        eval_data_path = os.path.join(cfg.evaluations.data_root, info.path)
        eval_type = info.evaluation_type
        eval_batch_size = info.batch_size * 4
        eval_num_workers = info.num_workers
        evaluator = get_evaluator_by_name(eval_type=eval_type, name=name, eval_data_path=eval_data_path,
                                          transform=eval_pipeline.make_test_transform(),
                                          fabric=fabric, batch_size=eval_batch_size, num_workers=eval_num_workers)
        evaluator.integrity_check(info.color_space, eval_pipeline.color_space)
        evaluator.config = info
        evaluators.append(evaluator)

    # copy project files
    if fabric.local_rank == 0:
        code_dir = os.path.dirname(os.path.abspath(__file__))
        os_utils.copy_project_files(code_dir, cfg.trainers.output_dir)
        omegaconf.OmegaConf.save(cfg, os.path.join(cfg.trainers.output_dir, 'config.yaml'))
        os.makedirs(os.path.join(cfg.trainers.output_dir, 'lightning_logs'), exist_ok=True)

    # train
    step = train_pipeline.step
    n_images_seen = train_pipeline.n_images_seen
    n_epochs = cfg.optims.num_epoch - train_pipeline.start_epoch
    print(f"start at {train_pipeline.start_epoch} and training for {n_epochs} epochs")
    is_best_tracker = IsBestTracker(fabric)
    tic = time.time()
    epoch = train_pipeline.start_epoch
    for epoch in range(train_pipeline.start_epoch, cfg.optims.num_epoch):
        epoch_start_time = time.time()
        train_pipeline.train()
        set_epoch(dataloader, epoch, cfg)
        batch_length = len(dataloader) if cfg.trainers.limit_num_batch <= 0 else cfg.trainers.limit_num_batch
        pbar = tqdm(total=batch_length, disable=fabric.local_rank != 0)
        if cfg.trainers.local_rank == 0:
            print('\nRun Name', os.path.basename(cfg.trainers.output_dir))
        for batch_idx, batch in enumerate(dataloader):

            if cfg.trainers.limit_num_batch > 0 and batch_idx >= cfg.trainers.limit_num_batch:
                break

            if cfg.trainers.mock_lr_run:
                loss = 0
            else:
                is_accumulating = batch_idx % cfg.trainers.gradient_acc != 0
                with fabric.no_backward_sync(model if model.has_trainable_params() else dummy_model,
                                             enabled=is_accumulating):
                    with fabric.autocast():
                        loss = train_pipeline(batch)
                        fabric.backward(loss)
                if not is_accumulating:
                    fabric.clip_gradients(model, optimizer, max_norm=cfg.optims.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

            scheduler_step(lr_scheduler, step)
            last_lr = get_last_lr(optimizer)

            n_images_seen += cfg.trainers.total_batch_size
            step += 1

            if batch_idx % 50 == 0:
                log_dict = {}
                log_dict['epoch'] = epoch
                log_dict['step'] = step
                log_dict['n_images_seen'] = n_images_seen
                log_dict['train/loss'] = loss
                log_dict['train/lr'] = last_lr
                log_dict['trainer/global_step'] = step
                log_dict['trainer/epoch'] = epoch
                fabric.log_dict(log_dict, step=step)

            speed = cfg.trainers.batch_size / (time.time() - tic)
            speed_total = speed * fabric.world_size
            pbar.set_description(f"Epoch {epoch} | Step {step} | Batch {batch_idx} | Speed {speed_total:.0f} | LR {last_lr:.5f} | Loss {loss:.4f}")
            pbar.update(1)
            tic = time.time()

        # validation
        if cfg.evaluations.eval_every_n_epochs > 0:
            print('Evaluation Started')
            eval_start_time = time.time()
            all_result = {}
            for evaluator in evaluators:
                if (epoch % cfg.evaluations.eval_every_n_epochs == 0  # every n epochs
                    or epoch == (cfg.optims.num_epoch - 1)  # last epoch
                    or epoch + 1 in cfg.optims.lr_milestones  # lr decay
                ):
                    print(f"Evaluating {evaluator.name}")
                    result = evaluator.evaluate(eval_pipeline, epoch=epoch, step=step, n_images_seen=n_images_seen)
                    all_result.update({evaluator.name + "/" + k: v for k, v in result.items()})
            eval_time = (time.time() - eval_start_time) / 60
            print(f'Evaluation Time: {eval_time:.2f} mins')

            if fabric.local_rank == 0:
                if all_result:
                    os.makedirs(os.path.join(cfg.trainers.output_dir, 'result'), exist_ok=True)
                    save_result = pd.DataFrame(pd.Series(all_result), columns=['val'])
                    save_result.to_csv(os.path.join(cfg.trainers.output_dir, f'result/eval_{epoch}_{step}.csv'))
                    mean, summary_dict = summary(save_result, epoch, step, n_images_seen)
                    fabric.log_dict(summary_dict)
                    summary_result = pd.DataFrame(pd.Series(summary_dict), columns=['val'])
                    summary_result.to_csv(os.path.join(cfg.trainers.output_dir, f'result/eval_summary_{epoch}_{step}.csv'))
                else:
                    print('Skipped evaluation. So best is not updated')
                    mean = is_best_tracker.prev_best_metric
            else:
                mean = -1.0
            is_best_tracker.set_is_best(mean)
            if fabric.local_rank == 0:
                fabric.log_dict({'is_best': float(is_best_tracker.is_best())})
                print(f'Epoch {epoch} | Step {step} | Best {is_best_tracker.is_best()}')
                if all_result:
                    print(summary_result.round(2).to_markdown())

            # save model
            train_pipeline.save(fabric, train_pipeline, cfg, epoch, step, n_images_seen,
                                is_best=is_best_tracker.is_best())
            print('Evaluation Finished and Model Saved')

        epoch_time = (time.time() - epoch_start_time) / 60
        print(f'Epoch Time: {epoch_time:.2f} mins')

    # load best model and do final eval
    is_best_path = os.path.join(cfg.trainers.output_dir, 'checkpoints', 'best')
    epoch = epoch + 1
    step = step + 1
    n_images_seen = n_images_seen + 1
    if os.path.exists(is_best_path) and cfg.trainers.skip_final_eval is False:
        fabric.barrier()
        time.sleep(fabric.local_rank * 5)  # prevent concurrent file access
        eval_pipeline.model.load_state_dict_from_path(os.path.join(is_best_path, 'model.pt'))
        print('Final Evaluation Started')

        # evaluation callbacks
        cfg.evaluations = config.load_yaml('final', directory='evaluations')
        evaluators = []
        for name, info in cfg.evaluations.per_epoch_evaluations.items():
            eval_data_path = os.path.join(cfg.evaluations.data_root, info.path)
            eval_type = info.evaluation_type
            eval_batch_size = info.batch_size
            eval_num_workers = info.num_workers
            evaluator = get_evaluator_by_name(eval_type=eval_type, name=name, eval_data_path=eval_data_path,
                                              transform=eval_pipeline.make_test_transform(),
                                              fabric=fabric, batch_size=eval_batch_size, num_workers=eval_num_workers)
            evaluator.integrity_check(info.color_space, eval_pipeline.color_space)
            evaluators.append(evaluator)


        all_result = {}
        for evaluator in evaluators:
            print(f"Evaluating {evaluator.name}")
            result = evaluator.evaluate(eval_pipeline, epoch=epoch, step=step, n_images_seen=n_images_seen)
            all_result.update({evaluator.name + "/" + k: v for k, v in result.items()})

        if fabric.local_rank == 0:
            os.makedirs(os.path.join(cfg.trainers.output_dir, 'result'), exist_ok=True)
            save_result = pd.DataFrame(pd.Series(all_result), columns=['val'])
            save_result.to_csv(os.path.join(cfg.trainers.output_dir, f'result/eval_best.csv'))
            mean, summary_dict = summary(save_result, epoch, step, n_images_seen)
            summary_dict = {k.replace('summary/', 'final/'): v for k, v in summary_dict.items()}
            # round to 2 decimal places
            summary_dict = {k: np.round(v, 2) for k, v in summary_dict.items()}
            fabric.log_dict(summary_dict)
            pd.DataFrame(pd.Series(summary_dict), columns=['val']).to_csv(
                os.path.join(cfg.trainers.output_dir, f'result/eval_summary_best.csv'))
    else:
        print('Skip final evaluation')

    # close
    print('done')