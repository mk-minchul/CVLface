from .model import _swin_transformer
from .modules_v2 import SwinTransformerBlockV2, PatchMergingV2

def swin_t():
    return _swin_transformer(img_size=[112,112], patch_size=[3, 3], embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                             window_size=[5, 5], stochastic_depth_prob=0.2, num_classes=512, )


def swin_s():
    return _swin_transformer(img_size=[112,112], patch_size=[3, 3], embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24],
                             window_size=[5, 5], stochastic_depth_prob=0.3, num_classes=512, )


def swin_b():
    return _swin_transformer(img_size=[112,112], patch_size=[3, 3], embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                             window_size=[5, 5], stochastic_depth_prob=0.5, num_classes=512, )


def swin_v2_t():
    return _swin_transformer(img_size=[112,112], patch_size=[3, 3], embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                             window_size=[5, 5], stochastic_depth_prob=0.2, num_classes=512,
                             block=SwinTransformerBlockV2, downsample_layer=PatchMergingV2, )


def swin_v2_s():
    return _swin_transformer(img_size=[112,112], patch_size=[3, 3], embed_dim=128, depths=[2, 2, 12, 2], num_heads=[4, 8, 16, 32],
                             window_size=[5, 5], stochastic_depth_prob=0.3, num_classes=512,
                             block=SwinTransformerBlockV2, downsample_layer=PatchMergingV2, )


def swin_v2_b():
    return _swin_transformer(img_size=[112,112], patch_size=[3, 3], embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                             window_size=[5, 5], stochastic_depth_prob=0.5, num_classes=512,
                             block=SwinTransformerBlockV2, downsample_layer=PatchMergingV2, )
