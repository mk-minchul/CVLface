from ..base import BaseAligner
from torchvision import transforms
from .retinaface import get_landmark_predictor, get_preprocessor
from . import aligner_helper
import torch
import torch.nn.functional as F
import numpy as np

class RetinaFaceAligner(BaseAligner):

    """
    A non-differentiable face aligner that aligns the image with one face to a canonical position.
    The aligner is based on the following paper:

    ```
    @inproceedings{deng2020retinaface,
      title={Retinaface: Single-shot multi-level face localisation in the wild},
      author={Deng, Jiankang and Guo, Jia and Ververas, Evangelos and Kotsia, Irene and Zafeiriou, Stefanos},
      booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
      pages={5203--5212},
      year={2020}
    }
    ```
    """

    def __init__(self, net, prior_box, preprocessor, config):
        super(RetinaFaceAligner, self).__init__()
        self.net = net
        self.prior_box = prior_box
        self.preprocessor = preprocessor
        self.config = config

    @classmethod
    def from_config(cls, config):
        net, prior_box = get_landmark_predictor(network=config.arch,
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
        input_img = normalize_for_net(unnormalize(x_bgr))

        result = self.net(input_img, self.prior_box)
        batch_loc, batch_conf, batch_landms = result
        batch_loc = torch.split(batch_loc, 1, dim=0)
        batch_conf = torch.split(batch_conf, 1, dim=0)
        batch_landms = torch.split(batch_landms, 1, dim=0)

        nms_ldmks = []
        nms_scores = []
        nms_bbox = []
        for loc, conf, landms, in zip(batch_loc, batch_conf, batch_landms):
            dets = postprocess(self.prior_box, loc, conf, landms, confidence_threshold=0.0, nms_threshold=0.4)
            bbox, score, ldmks = parse_one_det_result(dets)
            ldmks = ldmks / np.array( [self.prior_box.image_size[0], self.prior_box.image_size[1]] * 5)
            nms_ldmks.append(ldmks)
            nms_scores.append(score)
            nms_bbox.append(bbox)

        orig_pred_ldmks = torch.from_numpy(np.array(nms_ldmks)).to(self.device).float()
        score = torch.from_numpy(np.array(nms_scores)).to(self.device).float().unsqueeze(-1)
        bbox = torch.from_numpy(np.array(nms_bbox)).to(self.device).float()


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
        normalized_bbox = bbox / torch.tensor([[input_img.size(3), input_img.size(2)] * 2]).to(bbox.device)


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


def normalize(image):
    image = image / 255.
    image = (image - 0.5) / 0.5
    return image

def unnormalize(image):
    image = image * 0.5 + 0.5
    image = image * 255.
    return image

def normalize_for_net(bgr_image_0_255):
    # bgr_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return bgr_image_0_255 - torch.tensor([104, 117, 123])[None, :, None, None].to(bgr_image_0_255.device)


def postprocess(priorbox, loc, conf, landms, confidence_threshold, nms_threshold):

    device = loc.device
    im_height, im_width = priorbox.image_size

    scale = torch.Tensor([im_width, im_height, im_width, im_height])
    scale = scale.to(device)

    boxes = priorbox.decode(loc.data.squeeze(0))
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = priorbox.decode_landm(landms.data.squeeze(0))
    scale1 = torch.Tensor([im_width, im_height, im_width, im_height,
                           im_width, im_height, im_width, im_height,
                           im_width, im_height])
    scale1 = scale1.to(device)
    landms = landms * scale1
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    if len(inds) == 0:
        inds = np.where(scores >= 0)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1]
    # order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    # dets = dets[:args.keep_top_k, :]
    # landms = landms[:args.keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)
    return dets


def py_cpu_nms(dets,
               thresh):
    """
    Pure Python NMS baseline.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def parse_one_det_result(dets):
    dets_sorted = dets[dets[:, 4].argsort()[::-1]]
    result = dets_sorted[0]
    bbox = result[:4]
    score = result[4]
    ldmks = result[5:]
    return bbox, score, ldmks
