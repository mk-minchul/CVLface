

from .base import BasePipeline
from models.base import BaseModel

class InferModelPipeline(BasePipeline):

    def __init__(self,
                 model:BaseModel,
                 ):
        super(InferModelPipeline, self).__init__()

        self.model = model
        self.eval()

    @property
    def module_names_list(self):
        return ['model', ]

    def integrity_check(self, dataset_color_space):
        # color space check
        assert dataset_color_space == self.model.config.color_space
        self.color_space = dataset_color_space
        self.make_test_transform()

    def make_test_transform(self):
        return self.model.make_test_transform()

    def __call__(self, batch):
        inputs = batch
        feats = self.model(inputs)
        return feats


    def train(self):
        raise NotImplementedError('InferModelPipeline does not support train mode')


    def eval(self):
        self.model.eval()


