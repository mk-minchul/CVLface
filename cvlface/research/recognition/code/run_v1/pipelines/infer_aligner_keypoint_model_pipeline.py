import numpy as np

from .base import BasePipeline
from models.base import BaseModel
from aligners.base import BaseAligner
from PIL import Image

class InferAlignerKeypointModelPipeline(BasePipeline):

    def __init__(self,
                 aligner:BaseAligner,
                 model:BaseModel,
                 ):
        super(InferAlignerKeypointModelPipeline, self).__init__()

        self.aligner = aligner
        self.model = model
        self.eval()

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
        return feats


    def train(self):
        raise NotImplementedError('InferAlignerKeypointModelPipeline does not support train mode')


    def eval(self):
        self.aligner.eval()
        self.model.eval()


