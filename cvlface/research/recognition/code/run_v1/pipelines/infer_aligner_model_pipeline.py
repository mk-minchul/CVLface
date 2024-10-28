import numpy as np

from .base import BasePipeline
from models.base import BaseModel
from aligners.base import BaseAligner
from PIL import Image

class InferAlignerModelPipeline(BasePipeline):

    def __init__(self,
                 aligner:BaseAligner,
                 model:BaseModel,
                 ):
        super(InferAlignerModelPipeline, self).__init__()

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
        alinged_inputs = self.aligner(inputs.to(self.aligner.device))[0]
        feats = self.model(alinged_inputs)
        return feats


    def train(self):
        raise NotImplementedError('InferAlignerModelPipeline does not support train mode')


    def eval(self):
        self.aligner.eval()
        self.model.eval()


