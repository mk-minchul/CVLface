import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)
import os, sys

sys.path.append(os.path.join(root))

import queue
import argparse
import numpy as np
np.bool = np.bool_  # fix bug for mxnet 1.9.1
np.object = np.object_
np.float = np.float_
import mxnet as mx
import pandas as pd
from general_utils.os_utils import get_all_files, natural_sort
from tqdm import tqdm
from PIL import Image

class Writer():

    def __init__(self, save_root, prefix='train'):
        record, q_out, list_writer, mark_done_writer = self.prepare_record_saver(save_root, prefix=prefix)
        self.record = record
        self.list_writer = list_writer
        self.mark_done_writer = mark_done_writer  # needed for continuing
        self.q_out = q_out
        self.image_index = 0

    @staticmethod
    def prepare_record_saver(save_root, prefix='train'):

        os.makedirs(save_root, exist_ok=True)
        q_out = queue.Queue()
        fname_rec = f'{prefix}.rec'
        fname_idx = f'{prefix}.idx'
        fname_list = f'{prefix}.tsv'
        done_list = 'done_list.txt'

        if os.path.isfile(os.path.join(save_root, fname_idx)):
            os.remove(os.path.join(save_root, fname_idx))
        if os.path.isfile(os.path.join(save_root, fname_rec)):
            os.remove(os.path.join(save_root, fname_rec))
        if os.path.isfile(os.path.join(save_root, fname_list)):
            os.remove(os.path.join(save_root, fname_list))
        if os.path.isfile(os.path.join(save_root, done_list)):
            os.remove(os.path.join(save_root, done_list))

        record = mx.recordio.MXIndexedRecordIO(os.path.join(save_root, fname_idx),
                                               os.path.join(save_root, fname_rec), 'w')
        list_writer = open(os.path.join(save_root, fname_list), 'w')
        mark_done_writer = open(os.path.join(save_root, done_list), 'w')

        return record, q_out, list_writer, mark_done_writer

    def write(self, rgb_pil_img, save_path, label, bgr=False):
        assert isinstance(label, int)
        header = mx.recordio.IRHeader(0, label, self.image_index, 0)
        if bgr:
            # this saves in bgr
            s = mx.recordio.pack_img(header, np.array(rgb_pil_img), quality=100, img_fmt='.jpg')
        else:
            # this saves in rgb
            s = mx.recordio.pack_img(header, np.array(rgb_pil_img)[:,:,::-1], quality=100, img_fmt='.jpg')
        item = [self.image_index, save_path, label]

        self.q_out.put((item[0], s, item))
        _, s, _ = self.q_out.get()
        self.record.write_idx(item[0], s)
        line = f'{self.image_index}\t{save_path}\t{label}\n'
        self.list_writer.write(line)
        self.image_index = self.image_index + 1

    def close(self):
        self.record.close()
        self.list_writer.close()
        self.mark_done_writer.close()

    def mark_done(self, context, name):
        line = '%s\t' % context + '%s\n' % name
        self.mark_done_writer.write(line)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Bundle images into a record file. Images should be stored in a directory structure where each '
                    'subfolder is named after the label and contains images for that label, '
                    'e.g., label1/image1.png, label2/image2.png.')
    parser.add_argument('--source_dir', type=str,
                        help='Directory containing labeled image folders. '
                             'Example structure: label1/image.png, label2/image1.png, label2/image2.png')
    parser.add_argument('--remove_images', action='store_true',)

    args = parser.parse_args()
    source_dir = args.source_dir
    if source_dir.endswith('/'):
        source_dir = source_dir[:-1]
    save_dir = source_dir

    # find all images
    print('Finding all images in', source_dir)
    all_image_paths = get_all_files(source_dir, extension_list=['.jpg', '.png'], sort=True)
    print(f'Found {len(all_image_paths)} images in {source_dir}')

    # parse labels from directory structure
    paths = pd.Series(all_image_paths)
    dataset = pd.DataFrame(paths, columns=['path'])
    dataset['rel_path'] = dataset['path'].apply(lambda x: x.replace(source_dir+'/', ''))
    dataset['label'] = dataset['rel_path'].apply(lambda x: x.split('/')[0])
    dataset['image_name'] = dataset['rel_path'].apply(lambda x: x.split('/')[1])
    print('Num unique labels:', len(dataset['label'].unique()))

    unique_subject_ids = natural_sort(dataset['label'].unique().tolist())
    label_mapping = {i: sid for sid, i in zip(range(len(unique_subject_ids)), unique_subject_ids)}

    writer = Writer(save_dir, prefix='train')

    num_done = -1
    for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc='Writing images to record file'):
        orig_rgb_pil_img = Image.open(row['path'])
        label = label_mapping[row['label']]
        save_path = f'{label}/{row["image_name"]}'
        writer.write(rgb_pil_img=orig_rgb_pil_img, save_path=save_path, label=label, bgr=False)
        writer.mark_done(i, save_path)
        num_done += 1
        if num_done % 10000 == 0:
            os.makedirs(os.path.join(save_dir, 'examples'), exist_ok=True)
            orig_rgb_pil_img.save(os.path.join(save_dir, 'examples', f'{num_done}.jpg'))

    writer.close()

    # remove source image dir
    if args.remove_images:
        import shutil
        for d in os.listdir(source_dir):
            if os.path.isdir(os.path.join(source_dir, d)) and d != 'examples':
                shutil.rmtree(os.path.join(source_dir, d))