import torch
import torchvision
import cv2
import random
import numpy as np
from scipy.stats import multivariate_normal


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


class UseWithProb:
    def __init__(self, transform, prob=0.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            image = self.transform(image)
        return image


class Rotate:
    def __init__(self, n):
        self.n = n

    def __call__(self, img, mask=None):
        img = np.rot90(img, k=self.n)
        if mask is not None:
            mask = np.rot90(mask, k=self.n)
            return img, mask
        return img


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


class RandomCrop:
    def __init__(self, rnd_crop_min, rnd_crop_max=1):
        self.factor_max = rnd_crop_max
        self.factor_min = rnd_crop_min

    def __call__(self, img, mask=None):
        factor = random.uniform(self.factor_min, self.factor_max)
        size = (
            int(img.shape[1]*factor),
            int(img.shape[0]*factor)
        )
        img, x1, y1 = random_crop(img, size)
        if mask is None:
            return img
        mask = img_crop(mask, (x1, y1, x1 + size[0], y1 + size[1]))
        return img, mask


def img_crop(img, box):
    return img[box[1]:box[3], box[0]:box[2]]


def img_size(image: np.ndarray):
    return image.shape[1], image.shape[0]


def grayscale_handle(image, check, gray_img=False):
    if check:
        gray_img = False
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            gray_img = True
        return image, gray_img
    else:
        if gray_img:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image


def random_crop(img, size):
    tw = size[0]
    th = size[1]
    w, h = img_size(img)
    if ((w - tw) > 0) and ((h - th) > 0):
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
    else:
        x1 = 0
        y1 = 0
    img_return = img_crop(img, (x1, y1, x1 + tw, y1 + th))
    return img_return, x1, y1


# Image saturation fix
def satur_img(img):
    return np.clip(img, 0, 255)


class RandomShadow:
    def __init__(self):
        pass

    def __call__(self, image, mask=None):
        image, gray_img = grayscale_handle(image, True)
        row, col, ch = image.shape
        # We take a random point at the top for the x coordinate and then
        # another random x-coordinate at the bottom and join them to create
        # a shadow zone on the image.
        top_y = col * np.random.uniform()
        top_x = 0
        bot_x = row
        bot_y = col * np.random.uniform()
        img_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        shadow_mask = 0 * img_hls[:, :, 1]
        X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
        Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]

        shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1

        random_bright = .25 + .7 * np.random.uniform()
        cond0 = shadow_mask == 0
        cond1 = shadow_mask == 1

        if np.random.randint(2) == 1:
            img_hls[:, :, 1][cond1] = img_hls[:, :, 1][cond1] * random_bright
        else:
            img_hls[:, :, 1][cond0] = img_hls[:, :, 1][cond0] * random_bright
        image = cv2.cvtColor(img_hls, cv2.COLOR_HLS2RGB)

        image = satur_img(image)
        image = image.astype(np.uint8)
        image = grayscale_handle(image, False, gray_img)

        if mask is not None:
            return image, mask
        else:
            return image


class ImageGlare:
    def __init__(self, glare_mean, glare_deviation):
        self.glare_mean = glare_mean
        self.glare_deviation = glare_deviation

    def __call__(self, img, mask=None):
        img, gray_img = grayscale_handle(img, True)
        self.glare_bright = int(random.gauss(
            self.glare_mean,
            self.glare_deviation)
        )
        # sometimes error because of random cov matrix:
        # "the input matrix must be positive semidefinite"
        while True:
            try:
                size_blob = random.randint(1, 4)
                n1 = np.random.exponential(scale=0.5)
                n2 = np.random.exponential(scale=5.0)
                n3 = np.random.uniform(low=-0.99, high=0.99)

                x, y = np.meshgrid(np.linspace(-size_blob, size_blob, img.shape[1]),
                                   np.linspace(-size_blob, size_blob, img.shape[0]))
                pos = np.dstack((x, y))

                rv = multivariate_normal(
                    [random.randint(-2, 2), random.randint(-2, 2)],
                    [
                        [n1, 0],
                        [n3, n2]
                    ]
                )
                eps = rv.pdf(pos).max() + 10**(-8)
                img_blob = rv.pdf(pos) * (self.glare_bright / eps)

                # make "rgb" glare by stacking layer three times
                img_blob = np.stack((img_blob, img_blob, img_blob), axis=2)
                img_glare = img + img_blob

                img_glare = satur_img(img_glare)
                img_glare = img_glare.astype(np.uint8)
                img_glare = grayscale_handle(img_glare, False, gray_img)

                if mask is not None:
                    return img_glare, mask
                else:
                    return img_glare
            except:
                pass


class RandomGaussianBlur:
    '''Apply Gaussian blur with random kernel size
    Args:
        max_ksize (int): maximal size of a kernel to apply, should be odd
        sigma_x (int): Standard deviation
    '''
    def __init__(self, max_ksize=5, sigma_x=20):
        assert max_ksize % 2 == 1, "max_ksize should be odd"
        self.max_ksize = max_ksize // 2 + 1
        self.sigma_x = sigma_x

    def __call__(self, image, mask=None):
        image, gray_img = grayscale_handle(image, True)
        kernal_size = (1, 1)
        while kernal_size == (1, 1):
            kernal_size = tuple(2 * np.random.randint(0, self.max_ksize, 2) + 1)
        blured_image = cv2.GaussianBlur(image, kernal_size, self.sigma_x)

        blured_image = grayscale_handle(blured_image, False, gray_img)

        if mask is None:
            return blured_image
        return blured_image, mask


# Compute linear image transformation img*s+m
def lin_img(img, s=1.0, m=0.0):
    img = img.astype(np.int)
    img = img * s + m
    img = satur_img(img)
    img = img.astype(np.uint8)
    return img


def glare(img, n_vert, contr, bright):
    vert = []
    w, h = img_size(img)
    for i in range(random.randint(3, n_vert)):  # Create random vertices
        new_vert = (random.randint(0, w), random.randint(0, h))
        vert.append(new_vert)
    vert = np.array([vert], dtype=np.int32)
    mask = np.zeros_like(img)
    ignore_mask_color = (255,) * 3
    cv2.fillPoly(mask, vert, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    img[mask > 0] = lin_img(masked_image[mask > 0], contr, bright)
    return img


def gauss_noise(img, sigma_squared):
    w, h = img_size(img)
    gauss = np.random.normal(0, sigma_squared, (h, w, 3))
    gauss = gauss.reshape(h, w, 3)
    img = img + gauss
    return img


class RandomTransform(object):
    def __init__(self, contr=0.0, bright=0, sigma_squared=0, n_vert=None):
        self.contr = contr
        self.bright = bright
        self.sigma_squared = sigma_squared
        self.n_vert = n_vert
        if self.n_vert is not None:
            if self.n_vert < 4:
                self.n_vert = 4

    def __call__(self, img, mask=None):
        img, gray_img = grayscale_handle(img, True)

        if self.n_vert is not None:
            self.n_vert_rnd = np.random.randint(3, self.n_vert)
            bright_f = random.randint(0, 3 * self.bright)  # Brightness
            img = glare(img, self.n_vert_rnd, 1.0, bright_f)
            img = satur_img(img)
        contr_f = random.uniform(1 - self.contr, 1 + self.contr)  # Contrast
        bright_f = random.uniform(-self.bright, self.bright)  # Brightness
        img = lin_img(img, contr_f, bright_f)
        blur = random.randint(0, 1)  # Blur kernel type
        if blur == 1:
            img = cv2.GaussianBlur(img, (3, 3), 20)

        img = gauss_noise(img, random.uniform(0, self.sigma_squared))
        img = satur_img(img)
        img = np.uint8(img)
        img = grayscale_handle(img, False, gray_img)

        if mask is None:
            return img
        else:
            return img, mask


class GaussNoise:
    def __init__(self, sigma_sq):
        self.sigma_sq = sigma_sq

    def __call__(self, img, mask=None):
        if self.sigma_sq > 0.0:
            img = self._gauss_noise(img,
                                    np.random.uniform(0, self.sigma_sq))
        if mask is None:
            return img
        return img, mask

    def _gauss_noise(self, img, sigma_sq):
        img, gray_img = grayscale_handle(img, True)

        img = img.astype(np.uint32)
        h, w, c = img.shape
        gauss = np.random.normal(0, sigma_sq, (h, w))
        gauss = gauss.reshape(h, w)
        img = img + np.stack([gauss for i in range(c)], axis=2)
        img = satur_img(img)
        img = img.astype(np.uint8)

        img = grayscale_handle(img, False, gray_img)
        return img


def get_train_transforms(size, prob=0.2):
    transforms = torchvision.transforms.Compose([
        UseWithProb(GaussNoise(20), prob=prob),
        UseWithProb(
            RandomTransform(contr=0.5, bright=30, sigma_squared=20, n_vert=15),
            prob=prob
        ),
        UseWithProb(RandomGaussianBlur(max_ksize=3), prob=prob),
        UseWithProb(ImageGlare(70, 30), prob=prob),
        UseWithProb(RandomShadow(), prob=prob),
        UseWithProb(RandomCrop(0.8), prob=prob),
        UseWithProb(HorizontalFlip(), prob=prob),
        UseWithProb(VerticalFlip(), prob=prob),
        UseWithProb(Rotate(1), prob=prob),
        Scale(size),
        Normalize(),
        MoveChannels(),
        ToTensor()
    ])
    return transforms


def get_val_transforms(size):
    transforms = torchvision.transforms.Compose([
        Scale(size),
        Normalize(),
        MoveChannels(),
        ToTensor()
    ])
    return transforms
