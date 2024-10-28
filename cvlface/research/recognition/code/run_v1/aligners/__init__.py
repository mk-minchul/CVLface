from .base import BaseAligner


def get_aligner(aligner_cfg):

    if aligner_cfg.name == 'none':
        from .none import NoneAligner
        aligner = NoneAligner.from_config(aligner_cfg)
    elif aligner_cfg.name == 'retinaface_aligner':
        from .retinaface_aligner import RetinaFaceAligner
        aligner = RetinaFaceAligner.from_config(aligner_cfg)
    elif aligner_cfg.name == 'differentiable_face_aligner':
        from .differentiable_face_aligner import DifferentiableFaceAligner
        aligner = DifferentiableFaceAligner.from_config(aligner_cfg)
    else:
        raise ValueError(f"Unknown classifier: {aligner_cfg.name}")

    if aligner_cfg.start_from:
        aligner.load_state_dict_from_path(aligner_cfg.start_from)

    if aligner_cfg.freeze:
        for param in aligner.parameters():
            param.requires_grad = False
    return aligner

