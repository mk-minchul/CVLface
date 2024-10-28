from peft import LoraConfig, LoraModel
import torch
import os

def apply_peft(peft_config, model, classifier, data_cfg, label_mapping=None):

    if peft_config.name == 'none':
        return model, classifier

    print('Apply peft')

    if peft_config.model_ckpt_dir:
        print('load model from', peft_config.model_ckpt_dir)
        model.load_state_dict_from_path(os.path.join(peft_config.model_ckpt_dir, 'model.pt'))
    if peft_config.classifier_ckpt_dir and classifier is not None:
        print('load classifier from', peft_config.classifier_ckpt_dir)
        classifier.load_state_dict_from_path(os.path.join(peft_config.classifier_ckpt_dir, 'classifier.pt'))

    peft_model = apply_peft_to_model(peft_config, model)
    classifier = load_center(classifier, peft_config, data_cfg, label_mapping)
    return peft_model, classifier


def apply_peft_to_model(peft_config, model):


    if peft_config.name == 'lora':
        target_modules_mapping = {
            'att_qkv': ['qkv'],
            'att_qkv_keypoint_linear': ['qkv', 'keypoint_linear'],
            'att_qkv_feature': ['qkv', 'feature.0', 'feature.2'],
            'att_qkv_feature_keypoint_linear': ['qkv', 'feature.0', 'feature.2', 'keypoint_linear'],
        }
        target_modules = target_modules_mapping[peft_config.target_modules]
        peft_config = LoraConfig(r=peft_config.lora_rank, lora_alpha=peft_config.lora_rank,
                                 target_modules=target_modules,
                                 lora_dropout=0.1, bias="none")
        perf_model = LoraModel(model, peft_config, adapter_name='default')


    elif peft_config.name == 'part_freeze':
        target_modules_mapping = {}
        for k in range(24):
            target_modules_mapping[f'blocks.{k}'] = [f'blocks.{i}' for i in range(k, 24)]
            target_modules_mapping[f'blocks.{k}_feature'] = [f'blocks.{i}' for i in range(k, 24)] + ['feature']
            target_modules_mapping[f'blocks.{k}_keypoint_linear'] = [f'blocks.{i}' for i in range(k, 24)] + ['keypoint_linear']
            target_modules_mapping[f'blocks.{k}_feature_keypoint_linear'] = [f'blocks.{i}' for i in range(k, 24)] + ['feature'] + ['keypoint_linear']
        for k in range(49):
            target_modules_mapping[f'body.{k}'] = [f'body.{i}' for i in range(k, 49)] + ['output_layer']
        target_modules = target_modules_mapping[peft_config.target_modules]

        for key, param in model.named_parameters():
            is_train = False
            for target in target_modules:
                if target in key:
                    is_train = True
                    break
            if is_train:
                param.requires_grad = True
            else:
                param.requires_grad = False
        perf_model = model
    elif peft_config.name == 'full':
        # full train
        perf_model = model

    elif peft_config.name == 'freeze':
        for key, param in model.named_parameters():
            param.requires_grad = False
        perf_model = model
    else:
        raise ValueError(f"peft_config.name: {peft_config.name}")

    trainables, untrainables = print_trainable_parameters(perf_model)
    return perf_model

def load_center(classifier, peft_config, data_cfg, label_mapping=None):

    # center is a pre-computed center for each class for custom dataset

    if not peft_config.center_paths or classifier is None:
        print('skip_center loading')
        return classifier

    print('loading center')
    center = None

    if data_cfg.rec:
        if data_cfg.rec == 'webface260m/WebFace4M':
            main_center_path = 'WebFace4M'
        elif data_cfg.rec == 'webface260m/WebFace12M':
            main_center_path = 'WebFace12M'
        elif data_cfg.rec == 'webface260m/WebFace42M':
            main_center_path = 'WebFace42M'
        else:
            raise NotImplementedError(f"data_cfg.rec: {data_cfg.rec}")

        main_center_path_full = os.path.join(peft_config.model_ckpt_dir, 'centers', main_center_path, 'center.pth')
        if os.path.isfile(main_center_path_full):
            main_center_st = torch.load(main_center_path_full)
        else:
            main_center_path = main_center_path.lower()
            main_center_path_full = os.path.join(peft_config.model_ckpt_dir, 'centers', main_center_path, 'center.pth')
            main_center_st = torch.load(main_center_path_full)

        center = main_center_st['center']

        if label_mapping is not None:
            if len(label_mapping) < len(center):
                print('load label_mapping for center')
                if list(label_mapping.values()) == list(range(len(label_mapping))):
                    # then we can just use key as index
                    center = center[list(label_mapping.keys()), :]
                else:
                    raise NotImplementedError('label_mapping is not a simple index mapping')

    print('peft_config.center_paths', peft_config.center_paths)
    for center_path in peft_config.center_paths:
        _center = torch.load(os.path.join(peft_config.model_ckpt_dir, 'centers', center_path, 'center.pth'))['center']
        if center is None:
            center = _center
        else:
            center = torch.cat([center, _center], dim=0)

    if classifier.world_size == 1:
        assert len(center) == classifier.partial_fc.num_local

    class_start = classifier.partial_fc.class_start
    num_sample = classifier.partial_fc.num_local
    sub_center = center[class_start:class_start + num_sample, :]
    if classifier.partial_fc.weight.shape[0] != sub_center.shape[0]:
        print('Rank', classifier.rank,
              'classifier.partial_fc.weight.shape[0] != sub_center.shape[0]'
              'classifier.partial_fc.weight.shape[0]', classifier.partial_fc.weight.shape[0],
              'sub_center.shape[0]', sub_center.shape[0],
              'center.shape[0]', center.shape[0],
              'class_start', class_start,
              'num_sample', num_sample,
              'class_start+num_sample', class_start+num_sample)
        extra = classifier.partial_fc.weight.shape[0] - sub_center.shape[0]
        extra_center = sub_center[-extra:, :]
        sub_center = torch.cat([sub_center, extra_center], dim=0)
    classifier.partial_fc.weight.data.copy_(sub_center)

    if hasattr(classifier.partial_fc, 'batch_mean'):
        print('load batch_mean and batch_std')
        try:
            print(main_center_st['batch_mean'], main_center_st['batch_std'])
            batch_mean = main_center_st['batch_mean']
            batch_std = main_center_st['batch_std']
        except:
            print('no batch_mean and batch_std, using default from webface12m')
            batch_mean = torch.tensor([22.099])
            batch_std = torch.tensor([3.86])

        classifier.partial_fc.batch_mean.data.copy_(batch_mean)
        classifier.partial_fc.batch_std.data.copy_(batch_std)
    return classifier

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    trainable_names = []
    untrainable_names = []
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            trainable_names.append(name)
        else:
            untrainable_names.append(name)
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
    return trainable_names, untrainable_names