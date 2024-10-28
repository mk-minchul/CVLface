from ..base import BaseAligner


class NoneAligner(BaseAligner):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @classmethod
    def from_config(cls, aligner_config):
        return cls(aligner_config)

    def make_train_transform(self):
        return lambda x:x

    def make_test_transform(self):
        return lambda x:x

    def forward(self, x):
        return x
