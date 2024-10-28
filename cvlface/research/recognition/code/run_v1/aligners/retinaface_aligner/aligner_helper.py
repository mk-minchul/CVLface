import torch
import numpy as np
import cv2
from skimage import transform as trans
import cv2


def split_network_output(align_out):
    anchor_bbox_pred, anchor_cls_pred, anchor_ldmk_pred, merged, _ = align_out
    bbox, cls, ldmk = torch.split(merged, [4, 2, 10], dim=1)
    return ldmk, bbox, cls


def get_cv2_affine_from_landmark(ldmks, reference_ldmk, image_width, image_height, ):
    assert ldmks.ndim == 2  # batchdim
    assert ldmks.shape[1] == 10
    assert isinstance(ldmks, torch.Tensor)

    assert reference_ldmk.ndim == 2
    assert reference_ldmk.shape[0] == 5
    assert reference_ldmk.shape[1] == 2
    assert isinstance(reference_ldmk, np.ndarray)

    to_img_size = np.array([[[image_width, image_height]]])
    ldmks = ldmks.view(ldmks.shape[0], 5, 2).detach().cpu().numpy()
    ldmks = ldmks * to_img_size
    transforms = []
    for ldmk in ldmks:
        tform = trans.SimilarityTransform()
        tform.estimate(ldmk, reference_ldmk)
        M = tform.params[0:2, :]
        transforms.append(M)
    transforms = np.stack(transforms, axis=0)
    return transforms


def cv2_param_to_torch_theta(cv2_tfms, image_width, image_height, output_width, output_height):
    # https://github.com/wuneng/WarpAffine2GridSample
    """4.Affine Transformation Matrix to theta"""
    assert cv2_tfms.ndim == 3  # N, 2, 3
    assert cv2_tfms.shape[1] == 2
    assert cv2_tfms.shape[2] == 3

    srcs = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.float32)
    srcs = np.expand_dims(srcs, axis=0).repeat(cv2_tfms.shape[0], axis=0)
    dsts = np.matmul(srcs, cv2_tfms[:, :, :2].transpose(0, 2, 1)) + cv2_tfms[:, :, 2:3].transpose(0, 2, 1)

    # normalize to [-1, 1]
    srcs = srcs / np.array([[[image_width, image_height]]]) * 2 - 1
    dsts = dsts / np.array([[[output_width, output_height]]]) * 2 - 1

    thetas = []
    for src, dst in zip(srcs, dsts):
        theta = trans.estimate_transform("affine", src=dst, dst=src).params[:2]
        thetas.append(theta)
    thetas = np.stack(thetas, axis=0)
    thetas = torch.from_numpy(thetas).float()
    return thetas


def adjust_ldmks(ldmks, thetas):
    inv_thetas = inv_matrix(thetas).to(ldmks.device).float()
    _ldmks = torch.cat([ldmks, torch.ones((ldmks.shape[0], 5, 1)).to(ldmks.device)], dim=2)
    ldmk_aligned = (((_ldmks) * 2 - 1) @ inv_thetas.permute(0,2,1)) / 2 + 0.5
    return ldmk_aligned


def inv_matrix(theta):
    # torch batched version
    assert theta.ndim == 3
    a, b, t1 = theta[:, 0,0], theta[:, 0,1], theta[:, 0,2]
    c, d, t2 = theta[:, 1,0], theta[:, 1,1], theta[:, 1,2]
    det = a * d - b * c
    inv_det = 1.0 / det
    inv_mat = torch.stack([
        torch.stack([d * inv_det, -b * inv_det, (b * t2 - d * t1) * inv_det], dim=1),
        torch.stack([-c * inv_det, a * inv_det, (c * t1 - a * t2) * inv_det], dim=1)
    ], dim=1)
    return inv_mat

def reference_landmark():
    return np.array([[38.29459953, 51.69630051],
                     [73.53179932, 51.50139999],
                     [56.02519989, 71.73660278],
                     [41.54930115, 92.3655014],
                     [70.72990036, 92.20410156]])


def draw_ldmk(img, ldmk):
    if ldmk is None:
        return img
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    img = img.copy()
    for i in range(5):
        color = colors[i]
        cv2.circle(img, (int(ldmk[i*2] * img.shape[1]), int(ldmk[i*2+1] * img.shape[0])), 1, color, 4)
    return img