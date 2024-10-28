from ..base import BaseModel
from torchvision import transforms
from .swin.names import swin_v2_b, swin_v2_s
from .rpe_options import make_kprpe_shared, make_kprpe_bias
from torch import nn
from .RPE import build_rpe


class SWINKPRPEModelWithKPRPE(BaseModel):

    def __init__(self, net, config):
        super(SWINKPRPEModelWithKPRPE, self).__init__(config)
        self.net = net

        self.rpe_config = config.rpe_config
        self.keypoint_linear, self.num_buckets, self.swin_config = make_kprpe_shared(self.rpe_config, self.net)
        kprpes = []
        for feature_size in self.swin_config.keys():
            num_heads = self.swin_config[feature_size][0]['num_heads']
            _, rpe_k, _ = build_rpe(self.rpe_config, head_dim=None, num_heads=num_heads)
            kprpes.append(rpe_k)
        self.kprpes = nn.ModuleList(kprpes)

        # remove unused params in swin
        for mod in self.net.features:
            if hasattr(mod, 'attn'):
                if hasattr(mod.attn, 'relative_position_bias_table'):
                    del mod.attn.relative_position_bias_table
                if hasattr(mod.attn, 'cpb_mlp'):
                    del mod.attn.cpb_mlp


            assert config.mask_ratio == 0

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

    def forward(self, x, keypoints):
        if self.input_color_flip:
            x = x.flip(1)

        if self.rpe_config is None:
            extra_ctx = None
        else:
            extra_ctx = make_kprpe_bias(keypoints, x,
                                        self.keypoint_linear, self.rpe_config, self.swin_config, self.num_buckets,
                                        self.kprpes)

        return self.net(x, extra_ctx)

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
    model = SWINKPRPEModelWithKPRPE.from_config(model_config)
    return model

