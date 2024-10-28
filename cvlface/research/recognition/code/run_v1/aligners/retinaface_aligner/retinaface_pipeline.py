import torch
import numpy as np
import cv2
from .retinaface.utils.model_utils import load_model
from .retinaface.layers.functions.prior_box import PriorBox
from .retinaface.models.retinaface import RetinaFace
import torch.nn.functional as F


cfg_mnet = {
    'name': 'mobilenet0.25',
    'gpu_train': True,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    # 'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}


cfg_re50 = {
    'name': 'Resnet50',
    'gpu_train': True,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    # 'image_size': 840,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}


def load_retinface_model(network='resnet50', trained_model_path=''):
    cfg = None
    if network == "mobile0.25":
        cfg = cfg_mnet
    elif network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, trained_model_path, True)
    net.eval()
    # freeze grad
    for param in net.parameters():
        param.requires_grad = False

    return net


class RetinaFacePipeline(torch.nn.Module):

    def __init__(self, net, priorbox, input_size, device='cuda'):
        super().__init__()
        self.net = net
        self.priorbox = priorbox
        self.input_size = input_size
        self.output_size = 112
        self.device = device


    def normalize(self, image):
        image = image / 255.
        image = (image - 0.5) / 0.5
        return image

    def unnormalize(self, image):
        image = image * 0.5 + 0.5
        image = image * 255.
        return image

    def normalize_for_net(self, bgr_image_0_255):
        # bgr_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        return bgr_image_0_255 - torch.tensor([104, 117, 123])[None, :, None, None].to(self.device)

    def prealign_preprocess(self, images, value=0.0):
        # pad to input_size
        assert isinstance(images, torch.Tensor)
        assert images.ndim == 4 or images.ndim == 3
        input_size = self.input_size

        data_width = images.shape[-1]
        data_height = images.shape[-2]
        if data_width > input_size or data_height > input_size:
            # image is biggert than the input size
            # resize such that the larger side becomes the input_size without changing the aspect ratio
            if data_width > data_height:
                scale = input_size / data_width
            else:
                scale = input_size / data_height
            if images.ndim == 4:
                images = F.interpolate(input=images, scale_factor=scale,
                                        mode='bilinear', align_corners=False)
            else:
                images = F.interpolate(input=images.unsqueeze(0), scale_factor=scale,
                                        mode='bilinear', align_corners=False).squeeze(0)

        data_width = images.shape[-1]
        data_height = images.shape[-2]
        padding_width1 = (input_size - data_width) // 2
        padding_width2 = (input_size - data_width) - padding_width1
        padding_height1 = (input_size - data_height) // 2
        padding_height2 = (input_size - data_height) - padding_height1

        result = torch.nn.functional.pad(input=images,
                                         pad=(padding_width1, padding_width2,
                                              padding_height1, padding_height2),
                                         value=value)
        assert result.shape[-1] == input_size
        assert result.shape[-2] == input_size
        return result

    def forward(self, rgb_images):

        # cv2.imwrite('/mckim/temp/temp.jpg', self.unnormalize(rgb_images[0]).cpu().numpy().transpose(1,2,0))

        assert rgb_images.shape[1] == 3
        assert rgb_images.ndim == 4
        assert isinstance(rgb_images, torch.Tensor)
        assert self.priorbox.image_size == rgb_images.shape[2:]
        rgb_images = rgb_images.to(self.device)

        # make image into BGR
        bgr_images = rgb_images.flip(1)
        input_img = self.normalize_for_net(self.unnormalize(bgr_images))
        batch_loc, batch_conf, batch_landms = self.net(input_img)
        batch_loc = torch.split(batch_loc, 1, dim=0)
        batch_conf = torch.split(batch_conf, 1, dim=0)
        batch_landms = torch.split(batch_landms, 1, dim=0)

        all_ldmks = []
        for loc, conf, landms, in zip(batch_loc, batch_conf, batch_landms):
            dets = postprocess(self.priorbox, loc, conf, landms, confidence_threshold=0.0, nms_threshold=0.4)
            bbox, score, ldmks = parse_one_det_result(dets)
            ldmks = ldmks / np.array( [self.priorbox.image_size[0], self.priorbox.image_size[1]] * 5)
            all_ldmks.append(ldmks)
        all_ldmks = torch.from_numpy(np.array(all_ldmks)).to(self.device).float()
        return all_ldmks


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


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
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


def load_retinaface_pipeline(network, trained_model_path, input_size, device):
    net = load_retinface_model(network='resnet50', trained_model_path=trained_model_path)
    net = net.to(device)
    priorbox = PriorBox(image_size=(input_size, input_size),
                        min_sizes=[[16, 32], [64, 128], [256, 512]],
                        steps=[8,16,32], clip=False,
                        variances=[0.1, 0.2],
                        device=device)
    pipeline = RetinaFacePipeline(net, priorbox, input_size, device=device)
    pipeline.cuda()
    return pipeline
