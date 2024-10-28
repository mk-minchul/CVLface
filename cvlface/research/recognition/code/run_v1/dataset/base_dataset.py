import numbers
import os
import mxnet as mx
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import atexit
import pandas as pd
from tqdm import tqdm

def iterate_record(imgidx, record):
    # make one yourself
    record_info = []
    for idx in tqdm(imgidx, total=len(imgidx), desc='Iterating Dataset for extracting info (done only once)'):
        s = record.read_idx(idx)
        header, _ = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = int(label)
        row = {'_idx':idx, 'path': f'{label}/{idx}.jpg', 'label': label}
        record_info.append(row)
    record_info = pd.DataFrame(record_info)
    return record_info


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.to_PIL = transforms.ToPILImage()
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        self.imgrec.close = lambda: None
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

        info_path = os.path.join(root_dir, 'train.tsv')
        if os.path.isfile(info_path):
            self.info = pd.read_csv(os.path.join(root_dir, 'train.tsv'), sep='\t', header=None)
            self.info.columns = ['_idx', 'path', 'label']
        else:
            self.info = iterate_record(self.imgidx, self.imgrec)
            self.info.to_csv(info_path, sep='\t', header=False, index=False)
        self.label_info = {k: v for k, v in self.info.groupby('label')}

        atexit.register(self.dispose)

    def __getitem__(self, index):
        sample, label = self.read_sample(index)
        sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.info)

    def read_sample(self, index):
        info_index = self.info.index[index]
        idx = self.imgidx[info_index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        sample = self.to_PIL(sample)
        return sample, label


    def dispose(self):
        self.imgrec.close()

    def __del__(self):
        self.dispose()



class SyntheticDataset(Dataset):
    def __init__(self, num_class, num_sample):
        super(SyntheticDataset, self).__init__()
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).squeeze(0).float()
        img = ((img / 255) - 0.5) / 0.5
        self.img = img
        self.num_class = num_class
        self.num_sample = num_sample
        print("SyntheticDataset: num_class: {}, num_sample: {}".format(num_class, num_sample))

    def __getitem__(self, index):
        label = random.randint(0, self.num_class - 1)
        return self.img, label

    def __len__(self):
        return self.num_sample

