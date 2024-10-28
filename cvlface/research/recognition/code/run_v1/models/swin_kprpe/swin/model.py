from functools import partial
from typing import Any, Callable, List, Optional
from torch import nn, Tensor
from torchvision.ops.misc import MLP, Permute
from torchvision.utils import _log_api_usage_once
from .modules_v1 import PatchMerging, ShiftedWindowAttention, SwinTransformerBlock
from .modules_v2 import PatchMergingV2, ShiftedWindowAttentionV2, SwinTransformerBlockV2
import torch

class SwinTransformer(nn.Module):
    """
    Implements Swin Transformer from the `"Swin Transformer: Hierarchical Vision Transformer using
    Shifted Windows" <https://arxiv.org/abs/2103.14030>`_ paper.
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        num_classes (int): Number of classes for classification head. Default: 1000.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        downsample_layer (nn.Module): Downsample layer (patch merging). Default: PatchMerging.
    """

    def __init__(
        self,
        img_size: List[int],
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block: Optional[Callable[..., nn.Module]] = SwinTransformerBlockV2,
        downsample_layer: Callable[..., nn.Module] = PatchMergingV2,
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.num_classes = num_classes

        if block is None:
            block = SwinTransformerBlock
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        layers: List[nn.Module] = []
        # split image into non-overlapping patches
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    3, embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])
                ),
                Permute([0, 2, 3, 1]),
                norm_layer(embed_dim),
            )
        )

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                layers.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, norm_layer))
        self.features = nn.ModuleList(layers)

        num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(num_features)
        self.flatten = nn.Flatten(1)

        # features head
        if img_size[0] == 112 and patch_size[0] == 3:
            final_patch_size = [5, 5]
        else:
            # just calculate
            raise NotImplementedError("Not implemented for this img_size and patch_size")
        num_patches = final_patch_size[0] * final_patch_size[1]
        self.embed_dim = embed_dim        # basis feature dim for building blocks
        self.num_features = num_features  # C for final intermediate feature
        self.num_patches = num_patches    # N for output feature
        self.num_classes = num_classes    # C for return feature after flattening and mapping
        self.feature = nn.Sequential(
            nn.Linear(in_features=num_features * num_patches, out_features=num_features, bias=False),
            nn.BatchNorm1d(num_features=num_features, eps=2e-5),
            nn.Linear(in_features=num_features, out_features=num_classes, bias=False),
            nn.BatchNorm1d(num_features=num_classes, eps=2e-5)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, extra_ctx=None):
        for module in self.features:
            if isinstance(module, SwinTransformerBlock) or isinstance(module, SwinTransformerBlockV2):
                x = module(x, extra_ctx)
            else:
                x = module(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.feature(x)
        return x


def _swin_transformer(
        img_size: List[int],
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        stochastic_depth_prob: float,
        num_classes=512,
        **kwargs: Any
) -> SwinTransformer:

    model = SwinTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        stochastic_depth_prob=stochastic_depth_prob,
        num_classes=num_classes,
        **kwargs
    )

    return model

