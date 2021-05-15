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


# source https://www.kaggle.com/michaelwolff/bms-inchi-cropped-img-sizes-for-best-resolution
class AdaptiveCrop:
    """Crop mol from image to increase it overall resolution."""

    def __init__(self, contour_min_size=2, small_stuff_size=2,
                 small_stuff_dist=5, pad_pixels=3):
        self.contour_min_size = contour_min_size
        self.small_stuff_size = small_stuff_size
        self.small_stuff_dist = small_stuff_dist
        self.pad_pixels = pad_pixels

    def __call__(self, img_init):
        # preprocess image
        img = 255 - cv2.cvtColor(img_init, cv2.COLOR_BGR2GRAY)
        _, thresh = \
            cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        small_stuff = []
        x_min0, y_min0, x_max0, y_max0 = np.inf, np.inf, 0, 0
        for cnt in contours:
            # ignore contours under contour_min_size pixels
            if len(cnt) < self.contour_min_size:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # collect position of small contours starting with contour_min_size pixels
            if w <= self.small_stuff_size and h <= self.small_stuff_size:
                small_stuff.append([x, y, x+w, y+h])
                continue
            x_min0 = min(x_min0, x)
            y_min0 = min(y_min0, y)
            x_max0 = max(x_max0, x + w)
            y_max0 = max(y_max0, y + h)

        x_min, y_min, x_max, y_max = x_min0, y_min0, x_max0, y_max0
        # enlarge the found crop box if it cuts out small stuff that is very close by
        for i in range(len(small_stuff)):
            if (
                small_stuff[i][0] < x_min0
                and small_stuff[i][0] + self.small_stuff_dist >= x_min0
            ):
                x_min = small_stuff[i][0]
            if (
                small_stuff[i][1] < y_min0
                and small_stuff[i][1] + self.small_stuff_dist >= y_min0
            ):
                y_min = small_stuff[i][1]
            if (
                small_stuff[i][2] > x_max0
                and small_stuff[i][2] - self.small_stuff_dist <= x_max0
            ):
                x_max = small_stuff[i][2]
            if (
                small_stuff[i][3] > y_max0
                and small_stuff[i][3] - self.small_stuff_dist <= y_max0
            ):
                y_max = small_stuff[i][3]

        # make sure we get the crop within a valid range
        if self.pad_pixels > 0:
            y_min = max(0, y_min-self.pad_pixels)
            y_max = min(img.shape[0], y_max+self.pad_pixels)
            x_min = max(0, x_min-self.pad_pixels)
            x_max = min(img.shape[1], x_max+self.pad_pixels)

        img_cropped = img_init[y_min:y_max, x_min:x_max]
        return img_cropped


def get_train_transforms(output_height, output_width, prob):
    transforms = torchvision.transforms.Compose([
        # AdaptiveCrop(),
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
        # AdaptiveCrop(),
        Scale((output_height, output_width)),
        Normalize(),
        MoveChannels(),
        ToTensor()
    ])
    return transforms
