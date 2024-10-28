import torch
import random
import string
import os
import numpy as np


def preprocess_transform(examples, image_transforms):
    images = [image.convert("RGB") for image in examples['image']]
    images = [image_transforms(image) for image in images]
    examples["pixel_values"] = images
    return examples


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    indexes = torch.tensor([example["index"] for example in examples], dtype=torch.int)
    is_sames = torch.tensor([example["is_same"] for example in examples], dtype=torch.bool)

    return {
        "pixel_values": pixel_values,
        "index": indexes,
        "is_same": is_sames,
    }

def repeat_tensor_along_dim(tensor, dim, repeats):
    # Create the shape for repeating using list comprehension.
    # For all dimensions other than the specified 'dim', it will just have a '1' (i.e., no repeat).
    repeat_shape = [repeats if i == dim else 1 for i in range(tensor.dim())]
    return tensor.repeat(*repeat_shape)


def flatten_first_two_dims(tensor):
    flattened_shape = [-1] + list(tensor.shape[2:])
    # Reshape the tensor
    flattened_tensor = tensor.reshape(*flattened_shape)
    return flattened_tensor


def flatten_first_two_dims_numpy(array):
    flattened_shape = (-1,) + array.shape[2:]
    # Reshape the tensor
    flattened_array = array.reshape(*flattened_shape)
    return flattened_array

def first_unique_index(array):
    unique, idx, counts = torch.unique(array, dim=0, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0], device=cum_sum.device, dtype=cum_sum.dtype), cum_sum[:-1]))
    first_indicies = ind_sorted[cum_sum]
    return first_indicies

class BaseEvaluator():
    def __init__(self, name, fabric, batch_size):
        self.name = name
        self.fabric = fabric
        self.batch_size = batch_size

    def integrity_check(self, eval_color_space, pipeline_color_space):
        raise NotImplementedError('extract method must be implemented in subclass')

    def extract(self, pipeline):
        raise NotImplementedError('extract method must be implemented in subclass')

    def compute_metric(self, gathered_collection):
        raise NotImplementedError('extract method must be implemented in subclass')


    def evaluate(self, pipeline, epoch=0, step=0, n_images_seen=0):
        raise NotImplementedError('extract method must be implemented in subclass')

    def log(self, result, epoch, step, n_images_seen):
        # append the name of the evaluation to the key
        log_result = {f'val/{self.name}/{k}': v for k, v in result.items()}
        log_result['epoch'] = epoch
        log_result['step'] = step
        log_result['n_images_seen'] = n_images_seen
        # log
        self.fabric.log_dict(log_result)

    def complete_batch(self, batch):
        batch_keys = batch.keys()
        for key in batch_keys:
            if len(batch[key]) != self.batch_size:
                if isinstance(batch[key], torch.Tensor):
                    num_missing = self.batch_size - len(batch[key])
                    last_example = batch[key][-1].unsqueeze(0)
                    additional_examples = repeat_tensor_along_dim(last_example, dim=0, repeats=num_missing)
                    batch[key] = torch.cat([batch[key], additional_examples], dim=0)
                elif isinstance(batch[key], list):
                    batch[key] = batch[key] + [batch[key][-1]] * (self.batch_size - len(batch[key]))
        return batch


    def gather_collection(self, method='cpu', per_gpu_collection={}):
        # gathers dictionary across all gpus

        if method == 'cpu':
            runname = os.getcwd().split('/')[-1]
            if hasattr(self.fabric, 'loggers') and len(self.fabric.loggers) > 0 and hasattr(self.fabric.loggers[0], 'root_dir'):
                runname = runname + '_' + self.fabric.loggers[0].root_dir.split('/')[-1]

            cache_dir = os.path.join(os.path.expanduser('~/.cache'), runname, 'temporary_cpu_communication')
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(per_gpu_collection, os.path.join(cache_dir, f'per_gpu_collection_rank{self.fabric.local_rank}.pt'))
            self.fabric.barrier()

            if self.fabric.local_rank == 0:
                # load per gpu collection from cache
                gathered_collection = []
                world_size = self.fabric.world_size
                for rank in range(world_size):
                    per_gpu_collection = torch.load(os.path.join(cache_dir, f'per_gpu_collection_rank{rank}.pt'))
                    gathered_collection.append(per_gpu_collection)
                collection = {}
                for key in gathered_collection[0].keys():
                    concat = [per_gpu_collection[key] for per_gpu_collection in gathered_collection]
                    if isinstance(concat[0], list):
                        stacked = np.array(concat).transpose(1, 0)
                        stacked = flatten_first_two_dims_numpy(stacked)
                    else:
                        # stacked = torch.stack(concat, dim=0).transpose(0, 1)
                        assert isinstance(concat[0], torch.Tensor)
                        stacked = torch.stack(concat, dim=1)
                        stacked = flatten_first_two_dims(stacked)
                    collection[key] = stacked
                collection = self.remove_duplicates(collection)
                self.check_index_order(collection)
                # erase cache_dir
                os.system(f'rm -rf {cache_dir}')
            else:
                collection = None
            self.fabric.barrier()
        else:
            # gpu based gathering
            gathered_collection = self.fabric.all_gather(per_gpu_collection)
            collection = self.flatten_collection(gathered_collection)
            collection = self.remove_duplicates(collection)
            self.check_index_order(collection)
        return collection

    def flatten_collection(self, gathered_collection):

        # flatten collection by sorting by index
        # gathered_collection['index'] = torch.tensor([[2,3],[0,1],[4,5]])
        collection_order = torch.argsort(gathered_collection['index'].min(dim=1)[0])
        for key, val in gathered_collection.items():
            gathered_collection[key] = val[collection_order].transpose(0, 1)
        flattened_collection = {k:flatten_first_two_dims(v) for k, v, in gathered_collection.items()}

        return flattened_collection

    def remove_duplicates(self, collection):
        # find duplicate index and drop except the first one
        unique_idx = first_unique_index(collection['index'])
        for key, val in collection.items():
            collection[key] = val[unique_idx]
        return collection


    def check_index_order(self, collection):
        index_to_check = collection['index'].to(self.fabric.device)
        assert (index_to_check == torch.arange(index_to_check.shape[0],
                                                              dtype=index_to_check.dtype,
                                                              device=index_to_check.device)).all()

    def is_debug_run(self):
        try:
            if self.fabric.cfg.trainers.debug:
                return True
            else:
                return False
        except:
            return False