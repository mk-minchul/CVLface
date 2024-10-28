from torchvision.models import mobilenet_v3_small
import torch
from torch import nn as nn
import torch.nn.functional as F



class PartFVIT(nn.Module):

    def __init__(self, vit, num_patch, patch_size=8):
        super(PartFVIT, self).__init__()
        self.num_patch = num_patch
        self.patch_size = patch_size
        self.mobilenet = mobilenet_v3_small(weights=None, num_classes=num_patch * 2)
        self.mobilenet.classifier = nn.Identity()
        self.mobilenet.avgpool = nn.Identity()
        out = self.mobilenet(torch.randn(1, 3, 112, 112))
        c_in_features = out.shape[1]
        fc_loc = nn.Linear(c_in_features, num_patch * 2)

        num_step = int(self.num_patch ** 0.5)
        linspace = torch.linspace(-1, 1, num_step)
        grid_x, grid_y = torch.meshgrid(linspace, linspace, indexing='ij')
        grid_x = grid_x.reshape(-1)
        grid_y = grid_y.reshape(-1)
        bias = torch.stack([grid_x, grid_y], dim=1).view(-1)
        fc_loc.weight.data.zero_()
        fc_loc.bias.data.copy_(bias)

        self.mobilenet.classifier = fc_loc
        self.mobilenet(torch.randn(1, 3, 112, 112))

        self.patch_emb = nn.Linear(patch_size * patch_size * 3, vit.embed_dim)
        self.vit = vit

    def forward_patch_stn(self, x):
        coord_pred = self.mobilenet(x)
        coord_pred = coord_pred.view(-1, self.num_patch, 2)

        image_side = x.shape[-1]
        scale = (self.patch_size - 1) / (image_side - 1)
        batch_size = coord_pred.shape[0]
        all_patches = []
        for center in coord_pred.transpose(0, 1):
            theta = torch.tensor([[[scale, 0, 0], [0, scale, 0]]],
                                 requires_grad=True, dtype=coord_pred.dtype, device=coord_pred.device)
            theta = theta.repeat(batch_size, 1, 1)
            theta[:, :, -1] = center
            grid = F.affine_grid(theta, [batch_size, 3, self.patch_size, self.patch_size], align_corners=True)
            patches = F.grid_sample(x, grid, align_corners=True)
            all_patches.append(patches)
        all_patches = torch.stack(all_patches, 1).view(batch_size, self.num_patch, -1)
        return all_patches

    def forward(self, x):
        all_patches = self.forward_patch_stn(x)
        all_patches = self.patch_emb(all_patches)
        return self.vit(all_patches)


if __name__ == '__main__':
    model = PartFVIT()
    x = torch.randn(3, 3, 112, 112, requires_grad=True)
    out = model(x)

