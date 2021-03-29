import torch
import torchvision
import cv2
import numpy as np


class Scale:
    def __init__(self, size):
        self.size = tuple(size)

    def __call__(self, img, mask=None):
        resize_img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            resize_mask = cv2.resize(mask, self.size,
                                     interpolation=cv2.INTER_LINEAR)
            return resize_img, resize_mask
        else:
            return resize_img


class Normalize:
    def __call__(self, img):
        img = img.astype(np.float32) / 255
        return img


class ToTensor:
    def __call__(self, arr):
        arr = torch.from_numpy(arr)
        return arr


class AddChannel:
    def __call__(self, arr):
        arr = arr.unsqueeze(0)
        return arr


class MoveChannels:
    def __init__(self, to_channels_first=True):
        self.to_channels_first = to_channels_first

    def __call__(self, image):
        if self.to_channels_first:
            return np.moveaxis(image, -1, 0)
        else:
            return np.moveaxis(image, 0, -1)


def get_transforms(size):
    transforms = torchvision.transforms.Compose([
        Scale(size),
        Normalize(),
        MoveChannels(),
        ToTensor()
    ])
    return transforms
