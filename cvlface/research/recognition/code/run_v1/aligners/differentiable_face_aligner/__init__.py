from ..base import BaseAligner
from torchvision import transforms
from .dfa import get_landmark_predictor, get_preprocessor
from . import aligner_helper
import torch
import torch.nn.functional as F
import numpy as np


class DifferentiableFaceAligner(BaseAligner):

    '''
    A differentiable face aligner that aligns the image with one face to a canonical position.
    The aligner is based on the following paper (check out supplementary material for more details):
    @inproceedings{kim2024kprpe,
       title={{KeyPoint Relative Position Encoding for Face Recognition},
       author={Kim, Minchul and Su, Yiyang and Liu, Feng and Liu, Xiaoming},
       booktitle={CVPR},
       year={2024}
    }
    '''

    def __init__(self, net, prior_box, preprocessor, config):
        super(DifferentiableFaceAligner, self).__init__()
        self.net = net
        self.prior_box = prior_box
        self.preprocessor = preprocessor
        self.config = config

    @classmethod
    def from_config(cls, config):
        net, prior_box = get_landmark_predictor(network=config.arch,
                                                use_aggregator=True,
                                                input_size=config.input_size)

        preprocessor = get_preprocessor(output_size=config.input_size,
                                        padding=config.input_padding_ratio,
                                        padding_val=config.input_padding_val)
        if config.freeze:
            for param in net.parameters():
                param.requires_grad = False
        model = cls(net, prior_box, preprocessor, config)
        model.eval()
        return model

    def forward(self, x, padding_ratio_override=None):

        # input size check
        assert x.shape[1] == 3
        assert x.ndim == 4
        assert isinstance(x, torch.Tensor)
        is_square = x.shape[2] == x.shape[3]

        x = self.preprocessor(x, padding_ratio_override=padding_ratio_override)
        assert self.prior_box.image_size == x.shape[2:]

        # make image into BGR
        x_bgr = x.flip(1)
        result = self.net(x_bgr, self.prior_box)
        orig_pred_ldmks, bbox, cls = aligner_helper.split_network_output(result)
        score = torch.nn.Softmax(dim=-1)(cls)[:,1:]

        reference_ldmk = aligner_helper.reference_landmark()
        input_size = self.config.input_size
        output_size = self.config.output_size
        cv2_tfms = aligner_helper.get_cv2_affine_from_landmark(orig_pred_ldmks, reference_ldmk, input_size, input_size)
        thetas = aligner_helper.cv2_param_to_torch_theta(cv2_tfms, input_size, input_size, output_size, output_size)
        thetas = thetas.to(orig_pred_ldmks.device)

        output_size = torch.Size((len(thetas), 3, output_size, output_size))
        grid = F.affine_grid(thetas, output_size, align_corners=True)
        aligned_x = F.grid_sample(x + 1, grid, align_corners=True) - 1  # +1, -1 for making padding pixel 0
        aligned_ldmks = aligner_helper.adjust_ldmks(orig_pred_ldmks.view(-1, 5, 2), thetas)

        orig_pred_ldmks = orig_pred_ldmks.view(-1, 5, 2)
        # bbox (xmin, ymin, xmax, ymax)
        normalized_bbox = bbox / torch.tensor([[x_bgr.size(3), x_bgr.size(2)] * 2]).to(bbox.device)


        if padding_ratio_override is None:
            padding_ratio = self.preprocessor.padding
        else:
            padding_ratio = padding_ratio_override
        if padding_ratio > 0:
            # unpad the landmark so that it is in the original image coordinate
            scale = 1 / (1 + (2 * padding_ratio))
            pad_inv_theta = torch.from_numpy(np.array([[1 / scale, 0, 0], [0, 1 / scale, 0]]))
            pad_inv_theta = pad_inv_theta.unsqueeze(0).float().to(self.device).repeat(orig_pred_ldmks.size(0), 1, 1)
            unpad_ldmk_pred = torch.concat([orig_pred_ldmks.view(-1, 5, 2),
                                            torch.ones((orig_pred_ldmks.size(0), 5, 1)).to(self.device)], dim=-1)
            unpad_ldmk_pred = (((unpad_ldmk_pred) * 2 - 1) @ pad_inv_theta.mT) / 2 + 0.5
            unpad_ldmk_pred = unpad_ldmk_pred.view(orig_pred_ldmks.size(0), -1).detach()
            unpad_ldmk_pred = unpad_ldmk_pred.view(-1, 5, 2)
            if not is_square:
                unpad_ldmk_pred = None  # cannot use this if the input is not square becaouse preprocessor changes input
                normalized_bbox = None  # cannot use this if the input is not square becaouse preprocessor changes input
            return aligned_x, unpad_ldmk_pred, aligned_ldmks, score, thetas, normalized_bbox

        if not is_square:
            orig_pred_ldmks = None  # cannot use this if the input is not square becaouse preprocessor changes input
            normalized_bbox = None  # cannot use this if the input is not square becaouse preprocessor changes input
        return aligned_x, orig_pred_ldmks, aligned_ldmks, score, thetas, normalized_bbox

    def make_train_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        return transform

    def make_test_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        return transform

