from ..base import BaseModel
from .vit import VisionTransformer
from torchvision import transforms
from .part_fvit import PartFVIT

class PartFViTModel(BaseModel):

    """
    A PartFViT Model integrating a Vision Transformer (ViT) with additional functionality for part-based feature vision transformer (PartFVIT).
    Sun, Zhonglin, and Georgios Tzimiropoulos. "Part-based face recognition with vision transformers." [arXiv preprint arXiv:2212.00057 (2022)](https://arxiv.org/abs/2212.00057).
    """

    def __init__(self, net, config):
        super(PartFViTModel, self).__init__(config)
        self.net = net

    @classmethod
    def from_config(cls, config):

        if config.name == 'small':
            net = VisionTransformer(img_size=112, patch_size=8, num_classes=config.output_dim, embed_dim=512, depth=12,
                                    mlp_ratio=5, num_heads=8, drop_path_rate=0.1, norm_layer="ln",
                                    mask_ratio=config.mask_ratio)
            fvit = PartFVIT(net, num_patch=196, patch_size=8)
        elif config.name == 'base':
            net = VisionTransformer(img_size=112, patch_size=8, num_classes=config.output_dim, embed_dim=512, depth=24,
                                    mlp_ratio=3, num_heads=16, drop_path_rate=0.1, norm_layer="ln",
                                    mask_ratio=config.mask_ratio)
            fvit = PartFVIT(net, num_patch=196, patch_size=8)
        else:
            raise NotImplementedError

        model = cls(fvit, config)
        model.eval()
        return model

    def forward(self, x):

        if self.input_color_flip:
            x = x.flip(1)
        return self.net(x)

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

def load_model(model_config):
    model = PartFViTModel.from_config(model_config)
    return model