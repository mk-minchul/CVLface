import torch
from .lr_scheduler import param_groups_weight_decay


def make_optimizer(cfg, model, classifier, aligner):

    params = []
    num_total_params = 0
    num_trainable_params = 0

    if cfg.optims.filter_bias_and_bn:
        no_weight_decay_value = 0.0
    else:
        no_weight_decay_value = cfg.optims.weight_decay

    # get backbone param groups
    if model.has_trainable_params():
        no_weight_decay_list = []
        for name, param in model.named_parameters():
            if ('emb' in name and 'patch_embed' not in name) or ('token' in name):
                no_weight_decay_list.append(name)
        num_total_params += sum([p.numel() for p in model.parameters()])
        num_trainable_params += sum([p.numel() for p in model.parameters() if p.requires_grad])
        model_param_groups = param_groups_weight_decay(model.named_parameters(),
                                                       weight_decay=cfg.optims.weight_decay,
                                                       no_weight_decay_value=no_weight_decay_value,
                                                       no_weight_decay_list=no_weight_decay_list)
        params = params + model_param_groups

    # get classifier param groups
    if classifier is not None and classifier.has_trainable_params():
        num_total_params += sum([p.numel() for p in classifier.parameters()])
        num_trainable_params += sum([p.numel() for p in classifier.parameters() if p.requires_grad])
        cls_param_groups = [{"params": [p for p in classifier.parameters() if p.requires_grad],
                             'weight_decay': cfg.optims.weight_decay}]
        params = params + cls_param_groups

    # get aligner param groups
    if aligner.has_trainable_params():
        num_total_params += sum([p.numel() for p in aligner.parameters()])
        num_trainable_params += sum([p.numel() for p in aligner.parameters() if p.requires_grad])
        aligner_param_groups = [{"params": [p for p in aligner.parameters() if p.requires_grad],
                                 'weight_decay': cfg.optims.weight_decay}]
        params = params + aligner_param_groups

    # print number of params
    print(f"Total params: {num_total_params}")
    print(f"Trainable params: {num_trainable_params}")
    print(f"Percentage of trainable params: {num_trainable_params / num_total_params * 100:.2f}%")

    if cfg.optims.optimizer == "sgd":
        # TODO the params of partial fc must be last in the params list
        opt = torch.optim.SGD(params=params, lr=cfg.optims.lr, momentum=cfg.optims.momentum, weight_decay=cfg.optims.weight_decay)
    elif cfg.optims.optimizer == "adamw":
        opt = torch.optim.AdamW(params=params, lr=cfg.optims.lr, weight_decay=cfg.optims.weight_decay)
    else:
        raise

    return opt


def split_backbone_aligner(backbone):
    backbone_named_params = []
    aligner_named_params = []
    for name, param in backbone.named_parameters():
        if 'mobilenet' in name or 'fc_loc' in name:
            aligner_named_params.append((name, param))
        else:
            backbone_named_params.append((name, param))
    return backbone_named_params, aligner_named_params