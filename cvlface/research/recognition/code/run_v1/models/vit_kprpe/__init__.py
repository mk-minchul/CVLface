from ..base import BaseModel
from .vit import VisionTransformerWithKPRPE
from torchvision import transforms


class ViTKPRPEModel(BaseModel):


    """
    Vision Transformer for face recognition model with KeyPoint Relative Position Encoding (KP-RPE).

    ```
    @article{kim2024keypoint,
      title={KeyPoint Relative Position Encoding for Face Recognition},
      author={Kim, Minchul and Su, Yiyang and Liu, Feng and Jain, Anil and Liu, Xiaoming},
      journal={CVPR},
      year={2024}
    }
    ```
    """
    def __init__(self, net, config):
        super(ViTKPRPEModel, self).__init__(config)
        self.net = net


    @classmethod
    def from_config(cls, config):

        if config.name == 'small':
            net = VisionTransformerWithKPRPE(img_size=112, patch_size=8, num_classes=config.output_dim, embed_dim=512, depth=12,
                                    mlp_ratio=5, num_heads=8, drop_path_rate=0.1, norm_layer="ln",
                                    mask_ratio=config.mask_ratio, rpe_config=config.rpe_config)
        elif config.name == 'base':
            net = VisionTransformerWithKPRPE(img_size=112, patch_size=8, num_classes=config.output_dim, embed_dim=512, depth=24,
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
    model = ViTKPRPEModel.from_config(model_config)
    return model