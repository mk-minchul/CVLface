from .models.retinaface import RetinaFace
from .utils.model_utils import load_model
from .config import cfg_mnet, cfg_re50
from .layers.functions.prior_box import PriorBox
from .preprocessor import Preprocessor

def get_landmark_predictor(network='mobile0.25', input_size=160):

    cfg = None
    if network == "mobile0.25":
        cfg = cfg_mnet
    elif network == "resnet50":
        cfg = cfg_re50
    net = RetinaFace(cfg=cfg, phase = 'test')
    priorbox = PriorBox(image_size=(input_size, input_size),
                        # min_sizes=[[64, 80], [96, 112], [128, 144]],
                        min_sizes=[[16, 32], [64, 128], [256, 512]],
                        steps=[8, 16, 32],
                        clip=False,
                        variances=[0.1, 0.2],)

    # aligner = Aligner(net, priorbox, input_size, output_size=output_size)
    # return aligner
    return net, priorbox


def get_preprocessor(output_size=160, padding=0.0, padding_val='zero'):
    return Preprocessor(output_size=output_size, padding=padding, padding_val=padding_val)