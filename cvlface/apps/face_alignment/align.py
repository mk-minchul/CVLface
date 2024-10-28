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
import torch
from general_utils.os_utils import get_all_files


def pil_to_input(pil_image, device='cuda'):
    # input is a rgb image normalized.
    trans = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    input = trans(pil_image).unsqueeze(0).to(device)  # torch.randn(1, 3, 112, 112)
    return input


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--aligner_id', type=str, default='minchul/cvlface_DFA_mobilenet')
    parser.add_argument('--data_root', type=str, default='./example/images')
    parser.add_argument('--save_root', type=str, default='./example/aligned_images')
    args = parser.parse_args()

    # load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    aligner = load_model_by_repo_id(repo_id=args.aligner_id,
                                    save_path=os.path.expanduser(f'~/.cvlface_cache/{args.aligner_id}'),
                                    HF_TOKEN=os.environ['HF_TOKEN'], ).to(device)

    all_image_paths = get_all_files(args.data_root, extension_list=['.jpg', '.png'])

    for i, path in enumerate(all_image_paths):

        img1 = Image.open(path)
        input1 = pil_to_input(img1, device)

        # align
        aligned_x1, orig_pred_ldmks1, aligned_ldmks1, score1, thetas1, normalized_bbox1 = aligner(input1)

        # save aligned images
        vis1 = visualize(aligned_x1.cpu().clone())
        vis2 = visualize(aligned_x1.cpu().clone(), aligned_ldmks1.cpu())
        save_path = path.replace(args.data_root, args.save_root)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vis1.save(save_path)
        vis2.save(os.path.splitext(save_path)[0] + '_ldmks.png')
