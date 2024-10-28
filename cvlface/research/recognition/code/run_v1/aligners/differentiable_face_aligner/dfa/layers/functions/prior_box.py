import torch
from itertools import product as product
from math import ceil


class PriorBox(object):

    def __init__(self,
                 image_size,
                 min_sizes=[[64, 80], [96, 112], [128, 144]],
                 steps=[8,16,32],
                 clip=False,
                 variances=[0.1, 0.2],
                 ):
        super(PriorBox, self).__init__()
        self.min_sizes = min_sizes
        self.steps = steps
        self.clip = clip
        self.variances = variances
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        with torch.no_grad():
            self.priors = self.forward()

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        # import pandas as pd
        # pd.DataFrame(output.numpy()).to_csv('/mckim/temp/temp.csv')
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

    def encode(self, matched):
        """Encode the variances from the priorbox layers into the ground truth boxes
        we have matched (based on jaccard overlap) with the prior boxes.
        """
        self.priors = self.priors.to(matched.device)

        # dist b/t match center and prior's center
        g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - self.priors[:, :2]
        # encode variance
        g_cxcy /= (self.variances[0] * self.priors[:, 2:])
        # match wh / prior wh
        g_wh = (matched[:, 2:] - matched[:, :2]) / self.priors[:, 2:]
        g_wh = torch.log(g_wh) / self.variances[1]
        # return target for smooth_l1_loss
        return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]

    def encode_landm(self, matched):
        """Encode the variances from the priorbox layers into the ground truth boxes
        we have matched (based on jaccard overlap) with the prior boxes.
        """
        self.priors = self.priors.to(matched.device)

        # dist b/t match center and prior's center
        matched = torch.reshape(matched, (matched.size(0), 5, 2))
        priors_cx = self.priors[:, 0].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
        priors_cy = self.priors[:, 1].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
        priors_w = self.priors[:, 2].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
        priors_h = self.priors[:, 3].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
        priors = torch.cat([priors_cx, priors_cy, priors_w, priors_h], dim=2)
        g_cxcy = matched[:, :, :2] - priors[:, :, :2]
        # encode variance
        g_cxcy /= (self.variances[0] * priors[:, :, 2:])
        # g_cxcy /= priors[:, :, 2:]
        g_cxcy = g_cxcy.reshape(g_cxcy.size(0), -1)
        # return target for smooth_l1_loss
        return g_cxcy


    # Adapted from https://github.com/Hakuyume/chainer-ssd
    def decode(self, loc):
        """Decode locations from predictions using priors to undo
        the encoding we did for offset regression at train time.
        """
        self.priors = self.priors.to(loc.device)

        boxes = torch.cat((
            self.priors[:, :2] + loc[:, :2] * self.variances[0] * self.priors[:, 2:],
            self.priors[:, 2:] * torch.exp(loc[:, 2:] * self.variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def decode_landm(self, pre):
        """Decode landm from predictions using priors to undo
        the encoding we did for offset regression at train time.
        """
        self.priors = self.priors.to(pre.device)
        landms = torch.cat((self.priors[:, :2] + pre[:, :2] * self.variances[0] * self.priors[:, 2:],
                            self.priors[:, :2] + pre[:, 2:4] * self.variances[0] * self.priors[:, 2:],
                            self.priors[:, :2] + pre[:, 4:6] * self.variances[0] * self.priors[:, 2:],
                            self.priors[:, :2] + pre[:, 6:8] * self.variances[0] * self.priors[:, 2:],
                            self.priors[:, :2] + pre[:, 8:10] * self.variances[0] * self.priors[:, 2:],
                            ), dim=1)
        return landms


    def decode_batch(self, loc):
        """Decode locations from predictions using priors to undo
        the encoding we did for offset regression at train time.
        """
        self.priors = self.priors.to(loc.device)
        assert loc.ndim == 3
        priors = self.priors.unsqueeze(0).expand(loc.size(0), -1, -1)
        boxes = torch.cat((
            priors[:, :, :2] + loc[:, :, :2] * self.variances[0] * priors[:, :, 2:],
            priors[:, :, 2:] * torch.exp(loc[:, :, 2:] * self.variances[1])), -1)
        boxes[:, :, :2] -= boxes[:, :, 2:] / 2
        boxes[:, :, 2:] += boxes[:, :, :2]
        return boxes


    def decode_landm_batch(self, prediction):
        """Decode landm from prediction using priors to undo
        the encoding we did for offset regression at train time.
        """
        assert prediction.ndim == 3
        self.priors = self.priors.to(prediction.device)
        priors = self.priors.unsqueeze(0).expand(prediction.size(0), -1, -1)
        landms = torch.cat((priors[:, :, :2] + prediction[:, :, :2] * self.variances[0] * priors[:, :, 2:],
                            priors[:, :, :2] + prediction[:, :, 2:4] * self.variances[0] * priors[:, :, 2:],
                            priors[:, :, :2] + prediction[:, :, 4:6] * self.variances[0] * priors[:, :, 2:],
                            priors[:, :, :2] + prediction[:, :, 6:8] * self.variances[0] * priors[:, :, 2:],
                            priors[:, :, :2] + prediction[:, :, 8:10] * self.variances[0] * priors[:, :, 2:],
                            ), dim=-1)
        return landms