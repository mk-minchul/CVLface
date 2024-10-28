from .names import swin_v2_b, swin_v2_s
import torch


if __name__ == '__main__':

    model = swin_v2_b()
    model.eval()
    inputs_shape = (1, 3, 112, 112)
    x = torch.randn(*inputs_shape)
    y = model(x)
    print(x.shape)
    print(y.shape)

    model.eval()
    from fvcore.nn import flop_count
    import numpy as np
    res = flop_count(model, inputs=torch.randn(inputs_shape), supported_ops={})
    fvcore_flop = np.array(list(res[0].values())).sum()
    print(f'FLOPs: {fvcore_flop:.2f}', 'G')
    print('Num Params: ', sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 'M')


    model = swin_v2_s()
    model.eval()
    inputs_shape = (1, 3, 112, 112)
    x = torch.randn(*inputs_shape)
    y = model(x)
    print(x.shape)
    print(y.shape)

    model.eval()
    from fvcore.nn import flop_count
    import numpy as np
    res = flop_count(model, inputs=torch.randn(inputs_shape), supported_ops={})
    fvcore_flop = np.array(list(res[0].values())).sum()
    print(f'FLOPs: {fvcore_flop:.2f}', 'G')
    print('Num Params: ', sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 'M')