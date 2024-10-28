import torch
import torch.nn.functional as F

class Preprocessor():

    def __init__(self, output_size=160, padding=0.0, padding_val='zero'):
        self.output_size = output_size
        self.padding = padding
        self.padding_val = padding_val

    def preprocess_batched(self, imgs, padding_ratio_override=None):

        # check img is of float
        if imgs.dtype == torch.float32:
            if self.padding_val == 'zero':
                padding_val = -1.0
            elif self.padding_val == 'mean':
                padding_val = imgs.mean()
            else:
                raise ValueError('padding_val must be "zero" or "mean"')
        elif imgs.dtype == torch.uint8:
            if self.padding_val == 'zero':
                padding_val = 0
            elif self.padding_val == 'mean':
                padding_val = imgs.mean()
            else:
                raise ValueError('padding_val must be "zero" or "mean"')
        else:
            raise ValueError('imgs.dtype must be torch.float32 or torch.uint8')

        square_imgs = self.make_square_img_batched(imgs, padding_val=padding_val)

        if padding_ratio_override is not None:
            padding = padding_ratio_override
        else:
            padding = self.padding
        padded_imgs = self.make_padded_img_batched(square_imgs, padding=padding, padding_val=padding_val)

        size=(self.output_size, self.output_size)
        if imgs.dtype == torch.float32:
            resized_imgs = F.interpolate(padded_imgs, size=size, mode='bilinear', align_corners=True)
        elif imgs.dtype == torch.uint8:
            padded_imgs = padded_imgs.to(torch.float32)
            resized_imgs = F.interpolate(padded_imgs, size=size, mode='bilinear', align_corners=True)
            resized_imgs = torch.clip(resized_imgs, 0, 255)
            resized_imgs = resized_imgs.to(torch.uint8)
        else:
            raise ValueError('imgs.dtype must be torch.float32 or torch.uint8')
        return resized_imgs


    def make_square_img_batched(self, imgs, padding_val):
        assert imgs.ndim == 4
        # squarify the image
        h, w = imgs.shape[2:]
        if h > w:
            diff = (h - w)
            pad_left = diff // 2
            pad_right = diff - pad_left
            imgs = F.pad(imgs, (pad_left, pad_right, 0, 0), value=padding_val)
        elif w > h:
            diff = (w - h)
            pad_top = diff // 2
            pad_bottom = diff - pad_top
            imgs = F.pad(imgs, (0, 0, pad_top, pad_bottom), value=padding_val)
        assert imgs.shape[2] == imgs.shape[3]
        return imgs


    def make_padded_img_batched(self, imgs, padding, padding_val):
        if padding == 0:
            return imgs
        assert imgs.ndim == 4


        # pad the image
        h, w = imgs.shape[2:]
        pad_h = int(h * padding)
        pad_w = int(w * padding)
        imgs = F.pad(imgs, (pad_w, pad_w, pad_h, pad_h), value=padding_val)
        return imgs


    def __call__(self, input, padding_ratio_override=None):
        if input.ndim == 3:
            assert input.shape[0] == 3
            batch_input = input.unsqueeze(0)
            return self.preprocess_batched(batch_input, padding_ratio_override=padding_ratio_override)[0]
        elif input.ndim == 4:
            assert input.shape[1] == 3
            return self.preprocess_batched(input, padding_ratio_override=padding_ratio_override)
        else:
            raise ValueError(f'Invalid input shape: {input.shape}')