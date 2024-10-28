from ..base import BaseClassifier
from .partial_fc import PartialFC_V2


class PartialFCClassifier(BaseClassifier):

    def __init__(self, classifier, config, rank, world_size):
        super(PartialFCClassifier, self).__init__()
        self.partial_fc = classifier
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.apply_ddp = False

    @classmethod
    def from_config(cls, classifier_cfg, margin_loss_fn, model_cfg, num_classes, rank, world_size):
        if classifier_cfg.name == 'partial_fc':
            classifier = PartialFC_V2(
                rank=rank,
                world_size=world_size,
                margin_loss=margin_loss_fn,
                embedding_size=model_cfg.output_dim,
                num_classes=num_classes,
                sample_rate=classifier_cfg.sample_rate,
            )
        else:
            raise NotImplementedError

        model = cls(classifier, classifier_cfg, rank, world_size)
        model.eval()
        return model

    def forward(self, local_embeddings, local_labels):
        loss = self.partial_fc(local_embeddings, local_labels)
        return loss




