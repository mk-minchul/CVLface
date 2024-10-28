import torch
import math


class AdaFaceLoss(torch.nn.Module):
    def __init__(self,
                 s,
                 m,
                 h,
                 t_alpha,
                 interclass_filtering_threshold=0):
        super().__init__()
        self.s = s
        self.m = m
        self.h = h
        self.t_alpha = t_alpha
        self.interclass_filtering_threshold = interclass_filtering_threshold
        self.eps = 1e-3

    def forward(self, logits, labels, norms, batch_mean, batch_std):
        index_positive = torch.where(labels != -1)[0]

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty
            logits = tensor_mul * logits

        safe_norms = torch.clip(norms, min=0.001, max=100)  # for stability
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * batch_mean
            batch_std = std * self.t_alpha + (1 - self.t_alpha) * batch_std

        margin_scaler = (safe_norms - batch_mean) / (batch_std + self.eps)  # 66% between -1, 1
        margin_scaler = margin_scaler * self.h  # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1).view(-1)
        margin_scaler = margin_scaler[index_positive]

        target_logit = logits[index_positive, labels[index_positive].view(-1)]


        #########
        with torch.no_grad():
            # g_angular
            target_logit.arccos_()
            margin_final_logit = target_logit + (self.m * margin_scaler * -1)
            margin_final_logit.cos_()
            # g_additive
            margin_final_logit = margin_final_logit - (self.m + (self.m * margin_scaler))
            # make margin_final_logit as same dtype as logits
            margin_final_logit = margin_final_logit.type(logits.dtype)
            logits[index_positive, labels[index_positive].view(-1)] = margin_final_logit

        # scale
        logits = logits * self.s

        return logits, batch_mean, batch_std
