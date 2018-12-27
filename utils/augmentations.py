# matlab code:
# https://github.com/hellbell/ADNet/blob/3a7955587b5d395401ebc94a5ab067759340680d/utils/get_extract_regions.m
# other reference: https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

import numpy as np
import types
import torch
from torchvision import transforms
import cv2
import types
from numpy import random


class ToTensor(object):
    def __call__(self, cvimage, box=None, action_label=None, conf_label=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), box, action_label, conf_label


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, box=None, action_label=None, conf_label=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), box, action_label, conf_label


class CropRegion(object):
    def __call__(self, image, box, action_label=None, conf_label=None):
        image = np.array(image)
        box = np.array(box)
        if box is not None:
            center = box[0:2] + 0.5 * box[2:4]
            wh = box[2:4] * 1.4  # multiplication = 1.4
            box_lefttop = center - 0.5 * wh
            box_rightbottom = center + 0.5 * wh
            box_ = [
                max(0, box_lefttop[0]),
                max(0, box_lefttop[1]),
                min(box_rightbottom[0], image.shape[1]),
                min(box_rightbottom[1], image.shape[0])
            ]

            im = image[int(box_[1]):int(box_[3]), int(box_[0]):int(box_[2]), :]
        else:
            im = image[:, :, :]

        return im.astype(np.float32), box, action_label, conf_label


# crop "multiplication" times of the box width and height
class CropRegion_withContext(object):
    def __init__(self, multiplication=None):
        if multiplication is None:
            multiplication = 1.4  # same with default CropRegion
        assert multiplication >= 1, "multiplication should more than 1 so the object itself is not cropped"
        self.multiplication = multiplication

    def __call__(self, image, box, action_label=None, conf_label=None):
        image = np.array(image)
        box = np.array(box)
        if box is not None:
            center = box[0:2] + 0.5 * box[2:4]
            wh = box[2:4] * self.multiplication
            box_lefttop = center - 0.5 * wh
            box_rightbottom = center + 0.5 * wh
            box_ = [
                max(0, box_lefttop[0]),
                max(0, box_lefttop[1]),
                min(box_rightbottom[0], image.shape[1]),
                min(box_rightbottom[1], image.shape[0])
            ]

            im = image[int(box_[1]):int(box_[3]), int(box_[0]):int(box_[2]), :]
        else:
            im = image[:, :, :]

        return im.astype(np.float32), box, action_label, conf_label

# crop with context and multiplication, but keep the final square based on the lowest dimension
# e.g. dimension with context is 5x7. Then add more context on the first dimension, so the final dimension become 7x7
# add padding if the box is in the border and the edge is outside of the image
class CropRegion_withContext_square(object):
    def __init__(self, multiplication=None, border_padding="black"):
        assert border_padding in ['replicate', 'black']
        self.border_padding = border_padding

        if multiplication is None:
            multiplication = 1.4  # same with default CropRegion
        assert multiplication >= 1, "multiplication should more than 1 so the object itself is not cropped"
        self.multiplication = multiplication

    def __call__(self, image, box, action_label=None, conf_label=None):
        image = np.array(image)
        box = np.array(box)
        if box is not None:
            center = box[0:2] + 0.5 * box[2:4]
            wh = box[2:4] * self.multiplication
            if wh[0] < wh[1]:
                wh[0] = wh[1]
            elif wh[1] < wh[0]:
                wh[1] = wh[0]

            box_lefttop = center - 0.5 * wh
            box_rightbottom = center + 0.5 * wh
            box_ = [  # box for cropping the image... not the box of object
                max(0, box_lefttop[0]),
                max(0, box_lefttop[1]),
                min(box_rightbottom[0], image.shape[1]),
                min(box_rightbottom[1], image.shape[0])
            ]

            im = image[int(box_[1]):int(box_[3]), int(box_[0]):int(box_[2]), :]

            if box_lefttop[0] < 0:
                left_border = 0 - box_lefttop[0]
            else:
                left_border = 0
            if box_lefttop[1] < 0:
                top_border = 0 - box_lefttop[1]
            else:
                top_border = 0
            if box_rightbottom[0] > image.shape[1]:
                right_border = box_rightbottom[0] - image.shape[1]
            else:
                right_border = 0
            if box_rightbottom[1] > image.shape[0]:
                bottom_border = box_rightbottom[1] - image.shape[0]
            else:
                bottom_border = 0

            if self.border_padding == 'replicate':
                im = cv2.copyMakeBorder(im, int(top_border), int(bottom_border), int(left_border), int(right_border), cv2.BORDER_REPLICATE)
            else:  # elif self.border_padding == 'replicate':
                im = cv2.copyMakeBorder(im, int(top_border), int(bottom_border), int(left_border), int(right_border), cv2.BORDER_CONSTANT, 0)

        else:
            im = image[:, :, :]

        return im.astype(np.float32), box, action_label, conf_label


class ResizeImage(object):
    def __init__(self, inputSize):
        self.inputSize = inputSize  # network's input size (which is the output size of this function)

    def __call__(self, image, box, action_label=None, conf_label=None):
        im = cv2.resize(image, dsize=tuple(self.inputSize[:2]))
        return im.astype(np.float32), box, action_label, conf_label


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        # >>> augmentations.Compose([
        # >>>     transforms.CenterCrop(10),
        # >>>     transforms.ToTensor(),
        # >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, box=None, action_label=None, conf_label=None):
        for t in self.transforms:
            img, box, action_label, conf_label = t(img, box, action_label, conf_label)
        return img, box, action_label, conf_label


class ADNet_Augmentation(object):
    def __init__(self, opts):
        self.augment = Compose([
            SubtractMeans(opts['means']),
            CropRegion(),
            ResizeImage(opts['inputSize']),
            ToTensor()
        ])

    def __call__(self, img, box, action_label=None, conf_label=None):
        return self.augment(img, box, action_label, conf_label)


class ADNet_Augmentation_KeepAspectRatio(object):
    def __init__(self, opts, multiplication=None, border_padding="black"):
        self.augment = Compose([
            SubtractMeans(opts['means']),
            CropRegion_withContext_square(multiplication, border_padding),
            ResizeImage(opts['inputSize']),
            ToTensor()
        ])

    def __call__(self, img, box, action_label=None, conf_label=None):
        return self.augment(img, box, action_label, conf_label)