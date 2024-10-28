import numpy as np

from .base import BasePipeline
from models.base import BaseModel
from aligners.base import BaseAligner
from PIL import Image
import torch
from math import sqrt


class InferAlignerKeypointModelNMEScorePipeline(BasePipeline):

    def __init__(self,
                 aligner:BaseAligner,
                 model:BaseModel,
                 ):
        super(InferAlignerKeypointModelNMEScorePipeline, self).__init__()

        self.aligner = aligner
        self.model = model
        self.eval()

        self._reference_landmark = torch.from_numpy(np.array([[38.29459953, 51.69630051],
                                                              [73.53179932, 51.50139999],
                                                              [56.02519989, 71.73660278],
                                                              [41.54930115, 92.3655014],
                                                              [70.72990036, 92.20410156]]) / 112)
        self.reference_landmark = None

    @property
    def module_names_list(self):
        return ['aligner', 'model', ]

    def integrity_check(self, dataset_color_space):
        # color space check
        assert dataset_color_space == self.model.config.color_space
        self.color_space = dataset_color_space
        assert dataset_color_space == self.aligner.config.color_space
        self.make_test_transform()

    def make_test_transform(self):
        # check that aligner and model have the same transform
        aligner_transform = self.aligner.make_test_transform()
        model_transform = self.model.make_test_transform()
        x = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
        x = Image.fromarray(x)
        assert (aligner_transform(x) == model_transform(x)).all()
        return model_transform

    def __call__(self, batch):
        inputs = batch

        # if inputs.size(2) == 112:
        #     # we can pad it to be 160
        #     padding_ratio_override = 0.215
        # else:
        #     padding_ratio_override = None
        padding_ratio_override = None

        aligner_result = self.aligner(inputs.to(self.aligner.device), padding_ratio_override=padding_ratio_override)
        aligned_x, orig_pred_ldmks, aligned_ldmks, score, thetas, normalized_bbox = aligner_result
        assert inputs.size(2) == inputs.size(3)  # we can only use orig_pred_ldmks if the image is not altered
        feats = self.model(inputs, orig_pred_ldmks)

        if self.reference_landmark is None:
            self.reference_landmark = self._reference_landmark.to(orig_pred_ldmks.device).unsqueeze(0)

        nme = self.calc_nme_batched(self.reference_landmark, orig_pred_ldmks, bbox=[0, 0, 1, 1])
        ldmk_score = torch.clip((0.2 - torch.clip(nme, 0, 0.2)) / 0.2 + 1e-6, 0, 1)
        norm_norms = ldmk_score * score
        norm_norms = torch.clip(norm_norms, 1e-6, 1)
        norms = torch.norm(feats, p=2, dim=1, keepdim=True)
        feats = feats / norms * norm_norms

        return feats


    def train(self):
        raise NotImplementedError('InferAlignerKeypointModelPipeline does not support train mode')


    def eval(self):
        self.aligner.eval()
        self.model.eval()

    def calc_nme_batched(self, ldmk_gt, ldmk_pred, bbox):
        minx, miny, maxx, maxy = bbox
        llength = sqrt((maxx - minx) * (maxy - miny))

        assert ldmk_gt.ndim == 3
        assert ldmk_pred.ndim == 3

        # nme
        dis = (ldmk_pred - ldmk_gt) ** 2
        dis = torch.sqrt(torch.sum(dis, 2))
        dis = torch.mean(dis, 1, keepdim=True)
        nme = dis / llength
        return nme
