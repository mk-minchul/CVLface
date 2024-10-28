from ..base import BaseModel
from .vit import VisionTransformer
from torchvision import transforms


class ViTModel(BaseModel):

    """
    A class representing a Vision Transformer (ViT) model that inherits from the BaseModel class.

    This model applies the transformer architecture to image analysis, utilizing patches of images as input sequences,
    allowing for attention-based processing of visual elements.
    https://arxiv.org/abs/2010.11929
    ```
    @article{dosovitskiy2020image,
      title={An image is worth 16x16 words: Transformers for image recognition at scale},
      author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
      journal={arXiv preprint arXiv:2010.11929},
      year={2020}
    }
    ```
    """

    def __init__(self, net, config):
        super(ViTModel, self).__init__(config)
        self.net = net


    @classmethod
    def from_config(cls, config):

        if config.name == 'small':
            net = VisionTransformer(img_size=112, patch_size=8, num_classes=config.output_dim, embed_dim=512, depth=12,
                                    mlp_ratio=5, num_heads=8, drop_path_rate=0.1, norm_layer="ln",
                                    mask_ratio=config.mask_ratio)
        elif config.name == 'base':
            net = VisionTransformer(img_size=112, patch_size=8, num_classes=config.output_dim, embed_dim=512, depth=24,
                                    mlp_ratio=3, num_heads=16, drop_path_rate=0.1, norm_layer="ln",
                                    mask_ratio=config.mask_ratio)
        else:
            raise NotImplementedError

        model = cls(net, config)
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
    model = ViTModel.from_config(model_config)
    return model