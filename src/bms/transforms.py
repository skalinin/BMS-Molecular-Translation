import torch
import torchvision
import cv2
import random
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


class MoveChannels:
    def __init__(self, to_channels_first=True):
        self.to_channels_first = to_channels_first

    def __call__(self, image):
        if self.to_channels_first:
            return np.moveaxis(image, -1, 0)
        else:
            return np.moveaxis(image, 0, -1)


class UseWithProb:
    def __init__(self, transform, prob=0.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            image = self.transform(image)
        return image


def img_rotate90(img, k=1):
    """Rotate image.

    np.rot90() corrupts an opencv image
    https://stackoverflow.com/questions/20843544/np-rot90-corrupts-an-opencv-image
    https://github.com/opencv/opencv/issues/18120
    np.rot90 works a lot faster then
    cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) but .copy() workaround slow
    down np-realization to 4-5 times, so use cv2.rotate instead
    """
    assert k in [1, 2, 3], "k must be 1, 2 or 3"

    if k == 1:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif k == 2:
        return cv2.rotate(img, cv2.ROTATE_180)
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # 270


class Flip(object):
    def __init__(self, flip_code):
        assert flip_code == 0 or flip_code == 1
        self.flip_code = flip_code

    def __call__(self, img, mask=None):
        img = cv2.flip(img, self.flip_code)
        if mask is not None:
            mask = cv2.flip(mask, self.flip_code)
            return img, mask
        return img


class HorizontalFlip(Flip):
    def __init__(self):
        super().__init__(1)


class VerticalFlip(Flip):
    def __init__(self):
        super().__init__(0)


class RandomGaussianBlur:
    """Apply Gaussian blur with random kernel size

    Args:
        max_ksize (int): maximal size of a kernel to apply, should be odd
        sigma_x (int): Standard deviation
    """

    def __init__(self, max_ksize=5, sigma_x=20):
        assert max_ksize % 2 == 1, "max_ksize should be odd"
        self.max_ksize = max_ksize // 2 + 1
        self.sigma_x = sigma_x

    def __call__(self, image, mask=None):
        kernal_size = (1, 1)
        while kernal_size == (1, 1):
            kernal_size = tuple(2*np.random.randint(0, self.max_ksize, 2) + 1)
        blured_image = cv2.GaussianBlur(image, kernal_size, self.sigma_x)

        if mask is None:
            return blured_image
        return blured_image, mask


class MakeHorizontal:
    """Rotate image if its height greater than width.
    """
    def __call__(self, img):
        h, w, _ = img.shape
        if h > w:
            img = img_rotate90(img)
        return img


class Transpose:
    def __call__(self, img):
        return cv2.transpose(img)


class RandomTransposeAndFlip:
    """Rotate image by randomly apply transpose, vertical or horizontal flips.
    """
    def __init__(self):
        self.transpose = Transpose()
        self.vertical_flip = VerticalFlip()
        self.horizontal_flip = HorizontalFlip()

    def __call__(self, img):
        if random.random() < 0.5:
            img = self.transpose(img)
        if random.random() < 0.5:
            img = self.vertical_flip(img)
        if random.random() < 0.5:
            img = self.horizontal_flip(img)
        return img


def get_train_transforms(output_height, output_width, prob):
    transforms = torchvision.transforms.Compose([
        Scale((output_height, output_width)),
        RandomTransposeAndFlip(),
        UseWithProb(RandomGaussianBlur(max_ksize=3), prob=prob),
        Normalize(),
        MoveChannels(),
        ToTensor()
    ])
    return transforms


def get_val_transforms(output_height, output_width):
    transforms = torchvision.transforms.Compose([
        MakeHorizontal(),
        Scale((output_height, output_width)),
        Normalize(),
        MoveChannels(),
        ToTensor()
    ])
    return transforms
