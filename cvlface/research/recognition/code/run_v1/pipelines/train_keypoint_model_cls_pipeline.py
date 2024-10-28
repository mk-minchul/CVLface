from .base import BasePipeline
from models.base import BaseModel
import torch

class TrainKeypointModelClsPipeline(BasePipeline):

    def __init__(self,
                 model:BaseModel,
                 classifier:BaseModel,
                 optimizer,
                 lr_scheduler):
        super(TrainKeypointModelClsPipeline, self).__init__()

        self.model = model
        self.classifier = classifier
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    @property
    def module_names_list(self):
        return ['model', 'classifier', 'optimizer', 'lr_scheduler']

    def integrity_check(self, dataset):
        # color space check
        dataset_color_space = dataset.color_space
        assert dataset_color_space == self.model.config.color_space
        self.color_space = dataset_color_space
        self.make_train_transform()

    def make_train_transform(self):
        return self.model.make_train_transform()

    def __call__(self, batch):
        if len(batch) == 7:
            sample1, targets, ldmk1, theta1, sample2, ldmk2, theta2 = batch
        else:
            raise ValueError('not supported batch format')
        if sample2.ndim == 1:
            inputs = sample1
            ldmks = ldmk1
        else:
            inputs = torch.cat([sample1, sample2], dim=0)
            ldmks = torch.cat([ldmk1, ldmk2], dim=0)
            targets = torch.cat([targets, targets], dim=0)

        feats = self.model(inputs, ldmks)
        loss = self.classifier(feats, targets.to(self.classifier.device))
        return loss


    def train(self):
        if not self.model.config.freeze:
            self.model.train()
        if not self.classifier.config.freeze:
            self.classifier.train()


    def eval(self):
        self.model.eval()
        self.classifier.eval()


