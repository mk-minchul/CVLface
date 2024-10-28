from . import partial_fc
from . import fc

def get_classifier(classifier_cfg, margin_loss_fn, model_cfg, num_classes, rank, world_size):

    if margin_loss_fn is None:
        classifier = None
        print("No margin loss function provided, classifier will not be created")
        return classifier

    if classifier_cfg.name == 'partial_fc':
        classifier = partial_fc.PartialFCClassifier.from_config(classifier_cfg, margin_loss_fn,
                                                                model_cfg, num_classes,
                                                                rank, world_size)
    elif classifier_cfg.name == 'fc':
        classifier = fc.FCClassifier.from_config(classifier_cfg, margin_loss_fn,
                                                 model_cfg, num_classes,
                                                 rank, world_size)

    else:
        raise ValueError(f"Unknown classifier: {classifier_cfg.name}")

    if classifier_cfg.start_from:
        classifier.load_state_dict_from_path(classifier_cfg.start_from)

    if classifier_cfg.freeze:
        for param in classifier.parameters():
            param.requires_grad = False

    return classifier

