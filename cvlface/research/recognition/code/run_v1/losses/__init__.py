from .margin_loss import CombinedMarginLoss
from .adaface import AdaFaceLoss

def get_margin_loss(loss_config):
    if loss_config.margin_loss_name == 'margin':
        margin_loss = CombinedMarginLoss(
            64,
            loss_config.margin_list[0],
            loss_config.margin_list[1],
            loss_config.margin_list[2],
            loss_config.interclass_filtering_threshold
        )
    elif loss_config.margin_loss_name == 'adaface':
        margin_loss = AdaFaceLoss(
            64,
            m=loss_config.m,
            h=loss_config.h,
            t_alpha=loss_config.t_alpha,
            interclass_filtering_threshold=loss_config.interclass_filtering_threshold
        )
    elif loss_config.margin_loss_name == 'none':
        margin_loss = None
    else:
        raise ValueError("Not implemented loss margin_loss_name")
    return margin_loss

