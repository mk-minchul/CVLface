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

from tqdm import tqdm
from torch.utils.data import DataLoader
sys.path.append(os.path.join(root, 'research/recognition/code/run_v1'))
from research.recognition.code.run_v1.dataset.base_dataset import MXFaceDataset
from general_utils.huggingface_model_utils import load_model_by_repo_id
from general_utils.img_utils import visualize


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default='path_to_rec')
    parser.add_argument('--save_name', type=str, default='ldmk_5points.csv')
    parser.add_argument('--padding_ratio_override', type=float, default=0.215)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()
    source_dir = args.source_dir
    save_name = args.save_name
    padding = args.padding_ratio_override

    # load model
    HF_TOKEN = os.environ['HF_TOKEN']
    assert HF_TOKEN
    # convert ~/ to absolute path
    aligner = load_model_by_repo_id(repo_id='minchul/cvlface_DFA_mobilenet',
                                    save_path=os.path.expanduser('~/cache/cvlface_DFA_mobilenet'),
                                    HF_TOKEN=HF_TOKEN).to('cuda')

    # load dataset
    transform = aligner.model.make_test_transform()
    assert os.path.isdir(source_dir), f'{source_dir} is not a directory'
    dataset = MXFaceDataset(root_dir=source_dir, local_rank=0)
    dataset.transform = transform
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
    )

    f = open(os.path.join(source_dir, f'{save_name}'), 'w')
    f.write('idx,ldmk_0,ldmk_1,ldmk_2,ldmk_3,ldmk_4,ldmk_5,ldmk_6,ldmk_7,ldmk_8,ldmk_9\n')
    print('Saving result at ', os.path.join(source_dir, f'{save_name}'))

    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='Predicting landmarks'):
        x, target = batch
        x = x.to('cuda')
        aligned_x, orig_pred_ldmks, aligned_ldmks, score, thetas, bbox = aligner(x, padding_ratio_override=padding)
        for i, ldmk in enumerate(orig_pred_ldmks):
            ldmk = ldmk.reshape(-1)
            index = idx * args.batch_size + i
            f.write(f'{index},{ldmk[0]:.5f},{ldmk[1]:.5f},{ldmk[2]:.5f},{ldmk[3]:.5f},{ldmk[4]:.5f},{ldmk[5]:.5f},{ldmk[6]:.5f},{ldmk[7]:.5f},{ldmk[8]:.5f},{ldmk[9]:.5f}\n')

        if idx % 1000 == 0:
            vis = visualize(x[:4].cpu(), orig_pred_ldmks[:4].cpu())
            os.makedirs(os.path.join(source_dir, 'aligner_result_vis'), exist_ok=True)
            vis.save(os.path.join(source_dir, 'aligner_result_vis', f'{idx}.jpg'))

    f.close()


if __name__ == '__main__':

    main()