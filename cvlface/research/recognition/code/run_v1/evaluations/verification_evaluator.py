from datasets import Dataset
import torch
from functools import partial
from .base_evaluator import BaseEvaluator
from .verifications.verification import evaluate
import sklearn
from tqdm import tqdm
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


class VerificationEvaluator(BaseEvaluator):
    def __init__(self, name, data_path, transform, fabric, batch_size, num_workers):
        super().__init__(name, fabric, batch_size)
        self.name = name
        self.data_path = data_path
        dataset = Dataset.load_from_disk(data_path)
        preprocess = partial(preprocess_transform, image_transforms=transform)
        dataset = dataset.with_transform(preprocess)
        self.dataloader = fabric.setup_dataloader_from_dataset(dataset,
                                                               is_train=False,
                                                               batch_size=batch_size,
                                                               num_workers=num_workers,
                                                               collate_fn=collate_fn)


    def integrity_check(self, eval_color_space, pipeline_color_space):
        assert eval_color_space == pipeline_color_space


    @torch.no_grad()
    def evaluate(self, pipeline, epoch=0, step=0, n_images_seen=0):
        pipeline.eval()
        collection = self.extract(pipeline)
        collection_flip = self.extract(pipeline, flip_images=True)
        if self.fabric.local_rank == 0:
            result = self.compute_metric(collection, collection_flip)
            self.log(result, epoch, step, n_images_seen)
        else:
            result = {}
        return result


    def extract(self, pipeline, flip_images=False):
        all_features = []
        all_is_sames = []
        all_index = []
        for batch_idx, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader),
                                     desc=f'Verification {self.name}',
                                     disable=self.fabric.local_rank != 0):
            batch = self.complete_batch(batch)  # needed for last batch to be gather compatible

            if self.is_debug_run():
                if batch_idx > 10:
                    break

            images = batch['pixel_values']
            is_sames = batch['is_same']
            index = batch['index']

            if flip_images:
                images = torch.flip(images, dims=[3])
            features = pipeline(images)
            all_features.append(features.cpu().detach())
            all_is_sames.append(is_sames.cpu().detach())
            all_index.append(index.cpu().detach())

        # aggregate across all gpus
        per_gpu_collection = {"index": torch.cat(all_index, dim=0),
                              'is_same': torch.cat(all_is_sames, dim=0),
                              'features': torch.cat(all_features, dim=0)}

        # cpu based gathering just in case we have a lot of data
        collection = self.gather_collection(method='cpu', per_gpu_collection=per_gpu_collection)
        return collection


    def compute_metric(self, collection, collection_flip):
        if self.is_debug_run():
            print('Debug run, skipping metric computation')
            return {'acc': 0, 'std': 0}

        embeddings = (collection['features'] + collection_flip['features']).numpy()
        embeddings = sklearn.preprocessing.normalize(embeddings)
        issame_list = collection['is_same'].numpy()[::2]
        _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=10)
        accuracy = accuracy * 100
        acc, std = np.mean(accuracy), np.std(accuracy)
        result = {'acc': acc, 'std': std}
        return result

