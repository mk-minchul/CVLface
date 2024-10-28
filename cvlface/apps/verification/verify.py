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

import argparse
from general_utils.huggingface_model_utils import load_model_by_repo_id
from general_utils.img_utils import visualize
from general_utils.img_utils import prepare_text_img
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
import pandas as pd
import torch
import inspect


def pil_to_input(pil_image, device='cuda'):
    # input is a rgb image normalized.
    trans = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    input = trans(pil_image).unsqueeze(0).to(device)  # torch.randn(1, 3, 112, 112)
    return input


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--recognition_model_id', type=str, default='minchul/cvlface_adaface_ir101_webface12m')
    parser.add_argument('--recognition_model_id', type=str, default='minchul/cvlface_adaface_vit_base_kprpe_webface4m')
    parser.add_argument('--aligner_id', type=str, default='minchul/cvlface_DFA_mobilenet')
    parser.add_argument('--data_root', type=str, default='./example')
    parser.add_argument('--threshold', type=float, default=0.3)
    args = parser.parse_args()

    # load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fr_model = load_model_by_repo_id(repo_id=args.recognition_model_id,
                                     save_path=os.path.expanduser(f'~/.cvlface_cache/{args.recognition_model_id}'),
                                     HF_TOKEN=os.environ['HF_TOKEN'], ).to(device)
    aligner = load_model_by_repo_id(repo_id=args.aligner_id,
                                    save_path=os.path.expanduser(f'~/.cvlface_cache/{args.aligner_id}'),
                                    HF_TOKEN=os.environ['HF_TOKEN'], ).to(device)

    result = []
    pairs = pd.read_csv(os.path.join(args.data_root,'pairs.csv'), index_col=0)
    for i, row in pairs.iterrows():

        # read images
        img1 = Image.open(os.path.join(args.data_root, 'images', row['A']))
        img2 = Image.open(os.path.join(args.data_root, 'images', row['B']))
        input1 = pil_to_input(img1, device)
        input2 = pil_to_input(img2, device)

        # align
        aligned_x1, orig_pred_ldmks1, aligned_ldmks1, score1, thetas1, normalized_bbox1 = aligner(input1)
        aligned_x2, orig_pred_ldmks2, aligned_ldmks2, score2, thetas2, normalized_bbox2 = aligner(input2)

        # recognize
        input_signature = inspect.signature(fr_model.model.net.forward)
        if input_signature.parameters.get('keypoints') is not None:
            feat1 = fr_model(aligned_x1, aligned_ldmks1)
            feat2 = fr_model(aligned_x2, aligned_ldmks2)
        else:
            feat1 = fr_model(aligned_x1)
            feat2 = fr_model(aligned_x2)

        # compute cosine similarity
        cossim = torch.nn.functional.cosine_similarity(feat1, feat2).item()
        is_same = cossim > args.threshold
        result.append({'index': i, 'A': row['A'], 'B': row['B'], 'is_same': is_same, 'cossim': cossim})

        # save aligned images
        vis = visualize(torch.cat([aligned_x1, aligned_x2], dim=0).cpu())
        text_img = prepare_text_img(f'cossim: {cossim:.3f}\n Is Same: {is_same}', height=vis.size[1], width=vis.size[0])
        concat = np.concatenate([np.array(vis), text_img], axis=1)
        vis = Image.fromarray(concat)
        os.makedirs(os.path.join(args.data_root, 'visualization'), exist_ok=True)
        vis.save(os.path.join(args.data_root, 'visualization', f'{i}.png'))

    df = pd.DataFrame(result)
    df.to_csv(os.path.join(args.data_root, 'result.csv'), index=False)
    print(df)