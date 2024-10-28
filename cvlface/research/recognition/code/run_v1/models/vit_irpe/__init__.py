from ..base import BaseModel
from .vit import VisionTransformerWithiRPE
from torchvision import transforms


class ViTiRPEModel(BaseModel):


    """
    Vision Transformer for face recognition model with image Relative Position Encoding (ViT-iRPE) model.

    ```
    @inproceedings{wu2021rethinking,
      title={Rethinking and improving relative position encoding for vision transformer},
      author={Wu, Kan and Peng, Houwen and Chen, Minghao and Fu, Jianlong and Chao, Hongyang},
      booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
      pages={10033--10041},
      year={2021}
    }
    ```
    """

    def __init__(self, net, config):
        super(ViTiRPEModel, self).__init__(config)
        self.net = net


    @classmethod
    def from_config(cls, config):

        if config.name == 'small':
            net = VisionTransformerWithiRPE(img_size=112, patch_size=8, num_classes=config.output_dim, embed_dim=512, depth=12,
                                    mlp_ratio=5, num_heads=8, drop_path_rate=0.1, norm_layer="ln",
                                    mask_ratio=config.mask_ratio, rpe_config=config.rpe_config)
        elif config.name == 'base':
            net = VisionTransformerWithiRPE(img_size=112, patch_size=8, num_classes=config.output_dim, embed_dim=512, depth=24,
                                    mlp_ratio=3, num_heads=16, drop_path_rate=0.1, norm_layer="ln",
                                    mask_ratio=config.mask_ratio, rpe_config=config.rpe_config)
        else:
            raise NotImplementedError

        model = cls(net, config)
        model.eval()
        return model

    def forward(self, x, *args, **kwargs):
        if self.input_color_flip:
            x = x.flip(1)
        return self.net(x, *args, **kwargs)

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
    model = ViTiRPEModel.from_config(model_config)
    return model