import os
import torch

from .verification_evaluator import VerificationEvaluator
from .ijbbc_evaluator import IJBBCEvaluator
from .tinyface_evaluator import TinyFaceEvaluator

def get_evaluator_by_name(eval_type, name, eval_data_path, transform, fabric, batch_size, num_workers):

    assert os.path.isdir(eval_data_path), ('Evaluation Dataset does not exist. Check that cvlface/.env file is set correctly '
                                           'and the dataset is downloaded.')

    if eval_type == 'verification':
        return VerificationEvaluator(name, eval_data_path, transform, fabric, batch_size, num_workers)
    elif eval_type == 'ijbbc':
        return IJBBCEvaluator(name, eval_data_path, transform, fabric, batch_size, num_workers)
    elif eval_type == 'tinyface':
        return TinyFaceEvaluator(name, eval_data_path, transform, fabric, batch_size, num_workers)
    else:
        raise ValueError('Unknown evaluation type: %s' % eval_type)


def summary(save_result, epoch, step, n_images_seen):
    key_metrics = ['cfpfp/acc', 'agedb_30/acc', 'lfw/acc',
                   'cplfw/acc', 'calfw/acc',
                   'tinyface/rank-1', 'tinyface/rank-5',
                   'IJBB_gt_aligned/Norm:False_Det:True_tpr_at_fpr_0.0001',
                   'IJBC_gt_aligned/Norm:False_Det:True_tpr_at_fpr_0.0001']
    key_metrics_in_save_result = [k for k in key_metrics if k in save_result.index]
    if key_metrics_in_save_result:
        summary = save_result.loc[key_metrics_in_save_result]
        summary.index = ['summary/'+k.replace('/', '_') for k in summary.index]
        summary.index = [k.replace('Norm:False_Det:True_tpr_at_fpr_0.0001', 'TPR@FPR0.01') for k in summary.index]
        summary.index = [k.replace('_gt_aligned', '') for k in summary.index]
        mean = summary['val'].mean()

        summary_dict = summary['val'].to_dict()
        summary_dict['epoch'] = epoch
        summary_dict['step'] = step
        summary_dict['n_images_seen'] = n_images_seen
        summary_dict['trainer/global_step'] = step
        summary_dict['trainer/epoch'] = epoch

    else:
        mean = save_result['val'].mean()
        summary_dict = save_result['val'].to_dict()
        summary_dict['epoch'] = epoch
        summary_dict['step'] = step
        summary_dict['n_images_seen'] = n_images_seen
        summary_dict['trainer/global_step'] = step
        summary_dict['trainer/epoch'] = epoch
    return mean, summary_dict


class IsBestTracker():

    def __init__(self, fabric):
        self._is_best = True
        self.prev_best_metric = -1
        self.fabric = fabric


    def set_is_best(self, metric):
        metric_tensor = torch.tensor(metric, device=self.fabric.device)
        self.fabric.barrier()
        self.fabric.broadcast(metric_tensor, 0)
        metric = metric_tensor.item()

        if metric > self.prev_best_metric:
            self.prev_best_metric = metric
            self._is_best = True
        else:
            self._is_best = False


    def is_best(self):
        return self._is_best