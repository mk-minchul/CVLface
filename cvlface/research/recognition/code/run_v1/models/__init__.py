

def get_model(model_config, task=''):

    if '/vit/' in model_config.yaml_path:
        from .vit import load_model as load_vit_model
        model = load_vit_model(model_config)
        print('Loaded ViT model')
    elif '/vit_irpe/' in model_config.yaml_path:
        from .vit_irpe import load_model as load_vit_irpe_model
        model = load_vit_irpe_model(model_config)
        print('Loaded ViT model with iRPE')
    elif '/vit_kprpe/' in model_config.yaml_path:
        from .vit_kprpe import load_model as load_vit_kprpe_model
        model = load_vit_kprpe_model(model_config)
        print('Loaded ViT model with KPRPE')
    elif '/iresnet/' in model_config.yaml_path:
        from .iresnet import load_model as load_iresnet_model
        model = load_iresnet_model(model_config)
        print('Loaded iResNet model')
    elif '/iresnet_insightface/' in model_config.yaml_path:
        from .iresnet_insightface import load_model as load_iresnet_insightface_model
        model = load_iresnet_insightface_model(model_config)
        print('Loaded iResNet model')
    elif '/part_fvit/' in model_config.yaml_path:
        from .part_fvit import load_model as load_part_fvit_model
        model = load_part_fvit_model(model_config)
        print('Loaded PartFVIT model')
    elif '/swin/' in model_config.yaml_path:
        from .swin import load_model as load_swin_model
        model = load_swin_model(model_config)
        print('Loaded Swin model')
    elif '/swin_kprpe/' in model_config.yaml_path:
        from .swin_kprpe import load_model as load_swin_kprpe_model
        model = load_swin_kprpe_model(model_config)
        print('Loaded Swin model with KPRPE')
    else:
        raise NotImplementedError(f"Model {model_config.yaml_path} not implemented")
    if model_config.start_from:
        model.load_state_dict_from_path(model_config.start_from)

    if model_config.freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model
