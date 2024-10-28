from ..base import BaseModel
from torchvision import transforms
from .swin.names import swin_v2_b, swin_v2_s

class SWINModel(BaseModel):


    """
    A modified version of the Swin Transformer, tailored for facial recognition with an input dimension of 112x112 pixels.

    This model inherits from the BaseModel class and utilizes the smaller variants of the Swin Transformer architecture,
    such as `swin_v2_b` and `swin_v2_s`.

    The Swin Transformer uses shifted windows to bring greater efficiency and flexibility to the transformer architecture,
    allowing for attention mechanisms that adapt to the hierarchical nature of visual data.

    References:
        - Swin Transformer paper: https://arxiv.org/abs/2103.14030
        ```
        @inproceedings{liu2021swin,
          title={Swin transformer: Hierarchical vision transformer using shifted windows},
          author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
          booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
          pages={10012--10022},
          year={2021}
        }
        ```
    """

    def __init__(self, net, config):
        super(SWINModel, self).__init__(config)
        self.net = net


    @classmethod
    def from_config(cls, config):

        if config.name == 'small':
            net = swin_v2_s()
        elif config.name == 'base':
            net = swin_v2_b()
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
    model = SWINModel.from_config(model_config)
    return model